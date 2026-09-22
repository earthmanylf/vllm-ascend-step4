# SPDX-License-Identifier: Apache-2.0
"""Ascend Step-4 model fork.

Sliding-window layers reuse the upstream dense ``Step5Attention`` unchanged
(sparse_config=None, model_has_dsa_layers=False) and run on the
platform-default Ascend attention backend, which supports sliding windows.

Full-attention (DSA) layers use ``AscendStep5DSAAttention``, a
minimax_m3-style custom attention module whose impl runs the step5-hf
reference triton kernels through triton-ascend: region summaries are
self-managed per layer while K/V live in the standard paged KV pool.

Weight loading, MoE, norms and the causal-LM shell are inherited from the
upstream vLLM implementation, so module/parameter names here mirror it
exactly (qkv_indexer_proj, sparse_indexer_*, ssmax_s, ...).
"""

from __future__ import annotations

import copy
import os
import re

from typing import Any

import torch
from torch import nn
from torch.nn.parameter import Parameter

from vllm.config import VllmConfig
from vllm.distributed import (
    get_dp_group,
    get_ep_group,
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.linear import RowParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.utils import (
    PPMissingLayer,
    extract_layer_index,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
    sequence_parallel_chunk,
)
from vllm.sequence import IntermediateTensors
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheSpec

from vllm.models.step5.kernels import (
    Step5SparseConfig,
    checkpoint_has_step5_sparse_config,
    get_step5_sparse_config,
)
from vllm.models.step5.layernorm import (
    Step5IndexerLayerNorm,
    get_step5_norm_dtype,
    make_step5_rms_norm,
)
from vllm.models.step5.model import (
    Step5Attention,
    Step5DecoderLayer,
    Step5ForCausalLM,
    Step5FP32RouterLinear,
    Step5FusedQKVIndexerLinear,
    Step5MLP,
    Step5MoE,
    Step5Model,
    Step5SparseIndexerIndexTPLinear,
    _get_step5_layer_types,
    _get_step5_moe_layer_indices,
    _is_step5_full_attention_layer,
    _per_layer_value,
    _require_resolved_valid_vocab_size,
    _set_step5_moe_protocol_metadata,
    _step5_moe_reduce_policy,
    _validate_step5_dsa_parallel_geometry,
)

import vllm_ascend.ops  # noqa: F401  (loads before device_op to avoid its import cycle)
from vllm_ascend import envs
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.models.step5.dsa_attention import (
    AscendStep5DSABackend,
    AscendStep5DSAImpl,
)

from vllm_ascend.models.step5.fused_add_rms_norm import fused_add_rms_norm_cast
from vllm_ascend.models.step5.fused_qknorm_rope import (
    fused_indexer_qkz_kv_norm_rope,
    fused_qknorm_rope,
)
from vllm_ascend.models.step5.summary_cache import AscendStep5SparseSummaryCacheLayer
from vllm_ascend.quantization.configs.fp8_config import AscendFp8Config, Step5Fp8BlockConfig

logger = init_logger(__name__)

def _kv_cache_torch_dtype(cache_config, vllm_config) -> torch.dtype:
    cache_dtype = cache_config.cache_dtype if cache_config is not None else "auto"
    if cache_dtype in ("auto", None, ""):
        return vllm_config.model_config.dtype
    if cache_dtype in ("bfloat16", "bf16"):
        return torch.bfloat16
    if cache_dtype in ("float16", "fp16", "half"):
        return torch.float16
    if cache_dtype in ("float", "fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported step5-ascend KV cache dtype: {cache_dtype}")


def _step5_fp8_expert_layout(config: Any) -> tuple[set[int], int]:
    """Return (bf16 expert layer indices, quant block size).

    A Step5 checkpoint either declares ``quantization_config.fp8`` with
    128x128 weight blocks -- in which case every MoE layer is fp8 except
    those under ``modules_to_not_convert`` -- or declares nothing, in which
    case every expert is bf16.
    """
    quant_config = getattr(config, "quantization_config", None)
    if not quant_config:
        return set(), 0
    quant_method = quant_config.get("quant_method")
    if quant_method not in ("fp8", "compressed-tensors"):
        raise ValueError(
            f"Step5 Ascend fp8 experts support quant_method=fp8, got "
            f"{quant_method!r}."
        )
    block = quant_config.get("weight_block_size") or [128, 128]
    if list(block) != [128, 128]:
        raise ValueError(
            "Step5 Ascend fp8 experts require 128x128 weight blocks, got "
            f"{list(block)}."
        )
    bf16_layers: set[int] = set()
    for entry in quant_config.get("modules_to_not_convert") or []:
        match = re.search(r"layers\.(\d+)\.", entry)
        if match and ".moe." in entry:
            bf16_layers.add(int(match.group(1)))
    return bf16_layers, 128


class AscendStep5DSAAttention(nn.Module, AttentionLayerBase):
    """Step-4 DSA layer on Ascend: reference triton kernels, paged K/V pool."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float | list[float] | None = 10000,
        cache_config: Any | None = None,
        quant_config: Any | None = None,
        rope_scaling: dict[str, Any] | None = None,
        sliding_window: int | None = None,
        prefix: str = "",
        use_head_wise_attn_gate: bool = False,
        layer_types: list | None = None,
        use_rope_layers: list | None = None,
        yarn_only_types: list | None = None,
        swa_num_attention_heads: int | None = None,
        partial_rotary_factor: float = 1.0,
        zero_centered: bool = True,
        vllm_config: VllmConfig | None = None,
        sparse_config: Step5SparseConfig | None = None,
        norm_dtype: torch.dtype | None = None,
    ):
        super().__init__()
        assert sparse_config is not None and vllm_config is not None
        # Both are sliding-layer options the shared kwargs carry; DSA layers
        # are full-attention so they are accepted and ignored.
        del sliding_window, swa_num_attention_heads
        if qkv_bias:
            raise ValueError(
                "Step5 DSA layers fuse qkv with the sparse indexer into one "
                "bias-free GEMM, but attention_bias is enabled."
            )
        self.hidden_size = hidden_size
        self.total_num_heads = num_heads
        self.prefix = prefix
        self.layer_idx = extract_layer_index(prefix)
        tp_size = get_tensor_model_parallel_world_size()
        self.tp_size = tp_size
        self.rank = get_tensor_model_parallel_rank()

        layer_type = _per_layer_value(
            layer_types,
            self.layer_idx,
            name="layer_types",
            default="full_attention",
        )
        if yarn_only_types and layer_type not in yarn_only_types:
            rope_scaling = None
        if isinstance(rope_theta, list):
            rope_theta = _per_layer_value(
                rope_theta, self.layer_idx, name="rope_theta", default=None
            )

        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.partial_rotary_factor = float(partial_rotary_factor)
        self.rotary_dim = int(self.head_dim * self.partial_rotary_factor)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.zero_centered = zero_centered
        self.use_head_wise_attn_gate = use_head_wise_attn_gate
        self.use_rope = bool(
            _per_layer_value(
                use_rope_layers, self.layer_idx, name="use_rope_layers", default=True
            )
        )
        if norm_dtype is None:
            norm_dtype = get_step5_norm_dtype(vllm_config.model_config.hf_config)
        linear_quant_config = (
            quant_config
            if quant_config is None or quant_config.get_name() != "fp8"
            else None
        )

        # --- upstream DSA geometry validation -------------------------------
        if self.num_kv_heads != 1:
            raise ValueError(
                f"Step5 DSA requires exactly one local KV head, got {self.num_kv_heads}"
            )
        if self.head_dim not in (128, 192):
            raise ValueError(
                f"Step5 DSA head_dim must be 128/192, got {self.head_dim}"
            )
        _validate_step5_dsa_parallel_geometry(
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            indexer_num_heads=int(sparse_config.sparse_indexer_num_heads),
            index_tp_size=int(sparse_config.index_tp_size),
            tp_size=tp_size,
        )

        # --- projections -----------------------------------------------------
        proxy_dim = int(sparse_config.proxy_dim)
        self.sparse_config = sparse_config
        self.qkv_indexer_proj = Step5FusedQKVIndexerLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            indexer_q_output_size=(
                int(sparse_config.sparse_indexer_num_heads) * proxy_dim
            ),
            indexer_kv_output_size=(
                int(sparse_config.sparse_indexer_num_k_heads) * proxy_dim
            ),
            gate_output_size=(
                self.total_num_heads if use_head_wise_attn_gate else None
            ),
            proxy_dim=proxy_dim,
            index_tp_size=int(sparse_config.index_tp_size),
            quant_config=linear_quant_config,
            prefix=f"{prefix}.qkv_indexer_proj",
        )
        self.params_dtype = self.qkv_indexer_proj.params_dtype
        self.sparse_indexer_w = Step5SparseIndexerIndexTPLinear(
            hidden_size,
            int(sparse_config.sparse_indexer_num_heads),
            params_dtype=self.params_dtype,
            index_tp_size=int(sparse_config.index_tp_size),
        )
        # Dormant in the deployed scorer; registered for explicit loading.
        self.ssmax_s = Parameter(
            torch.zeros(self.total_num_heads, dtype=torch.float32),
            requires_grad=False,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=linear_quant_config,
            prefix=f"{prefix}.o_proj",
        )

        # --- main attention norms + rope -------------------------------------
        rope_parameters: dict[str, Any] = (
            dict(rope_scaling) if rope_scaling is not None else {}
        )
        rope_parameters.setdefault("rope_type", "default")
        if self.rope_theta is not None:
            rope_parameters["rope_theta"] = self.rope_theta
        rope_parameters["partial_rotary_factor"] = partial_rotary_factor

        self._sparse_indexer_base_rope_parameters = dict(rope_parameters)
        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            max_position=max_position,
            rope_parameters=rope_parameters,
            dtype=self.params_dtype,
        )
        self.q_norm = make_step5_rms_norm(
            self.head_dim,
            eps=rms_norm_eps,
            zero_centered=zero_centered,
            dtype=norm_dtype,
        )
        self.k_norm = make_step5_rms_norm(
            self.head_dim,
            eps=rms_norm_eps,
            zero_centered=zero_centered,
            dtype=norm_dtype,
        )

        # --- indexer norms + rope (upstream eager path) -----------------------
        sparse_params = dict(self._sparse_indexer_base_rope_parameters)
        if str(sparse_params.get("rope_type", "default")).strip().lower() == "none":
            sparse_params["rope_type"] = "default"
        sparse_params["partial_rotary_factor"] = float(
            int(sparse_config.sparse_indexer_rope_dim) / proxy_dim
        )
        self.sparse_indexer_rotary_emb = get_rope(
            head_size=proxy_dim,
            max_position=max_position,
            rope_parameters=sparse_params,
            dtype=self.params_dtype,
        )
        self.sparse_indexer_q_norm = make_step5_rms_norm(
            proxy_dim,
            eps=rms_norm_eps,
            zero_centered=zero_centered,
        )
        self.sparse_indexer_k_norm = Step5IndexerLayerNorm(
            proxy_dim,
            eps=rms_norm_eps,
        )

        # --- indexer fused norm+rope switch --------------------------------
        indexer_rope_dim = int(sparse_config.sparse_indexer_rope_dim)
        self.use_fused_indexer_norm_rope = (
            os.environ.get("VLLM_STEP4_FUSE_INDEXER_NORM_ROPE", "1") == "1"
            and indexer_rope_dim > 0
            and indexer_rope_dim < proxy_dim
        )
        if self.use_fused_indexer_norm_rope:
            self._indexer_rotary_dim = indexer_rope_dim
            self._indexer_num_q_heads = int(
                sparse_config.sparse_indexer_num_heads
            ) // int(sparse_config.index_tp_size)
            self._indexer_num_k_heads = int(
                sparse_config.sparse_indexer_num_k_heads
            )
            self._indexer_zero_centered = zero_centered
            indexer_rotary_cache = self.sparse_indexer_rotary_emb.cos_sin_cache
            self._indexer_rope_cos, self._indexer_rope_sin = (
                indexer_rotary_cache.chunk(2, dim=-1)
            )

        # --- vLLM attention-layer plumbing (minimax_m3 pattern) ---------------
        self.layer_name = f"{prefix}.attn"
        self.kv_cache_dtype = (
            cache_config.cache_dtype if cache_config is not None else "auto"
        )
        self.kv_cache_torch_dtype = _kv_cache_torch_dtype(cache_config, vllm_config)
        self.attn_backend = AscendStep5DSABackend
        # Mirror the upstream CUDA wiring (vllm.models.step5.model): the MTP
        # speculative token count is handed to the DSA impl so verify batches
        # (K+1 rows per request) take the staged/deferred-flush summary path.
        num_speculative_tokens = 0
        spec_config = vllm_config.speculative_config
        if spec_config is not None and spec_config.method == "mtp":
            num_speculative_tokens = int(
                getattr(spec_config, "num_speculative_tokens", 0) or 0
            )
        self.impl = AscendStep5DSAImpl(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
            kv_cache_dtype=self.kv_cache_dtype,
            topk_regions=int(sparse_config.topk),
            region_size=int(sparse_config.region_block_size),
            proxy_dim=proxy_dim,
            max_num_seqs=int(vllm_config.scheduler_config.max_num_seqs),
            max_model_len=int(vllm_config.model_config.max_model_len),
            block_size=int(vllm_config.cache_config.block_size),
            max_num_batched_tokens=int(
                vllm_config.scheduler_config.max_num_batched_tokens
            ),
            decode_out_dtype=vllm_config.model_config.dtype,
            num_speculative_tokens=num_speculative_tokens,
        )
        compilation_config = vllm_config.compilation_config
        if self.layer_name in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {self.layer_name}")
        compilation_config.static_forward_context[self.layer_name] = self
        self.kv_cache = torch.tensor([])
        self.max_position_embeddings = max_position

        self.use_fused_qknorm_rope = (
            os.environ.get("VLLM_STEP4_FUSE_QK_NORM_ROPE", "1") == "1"
            and self.use_rope
            and self.head_dim == 192
        )
        if self.use_fused_qknorm_rope:
            self._fused_rotary_pairs = self.rotary_dim // 2
            rotary_cache = self.rotary_emb.cos_sin_cache
            self.rope_cos, self.rope_sin = rotary_cache.chunk(2, dim=-1)
        else:
            self._fused_rotary_pairs = 0
            self.rope_cos = None
            self.rope_sin = None

        # Page-backed sparse summary sidecar, aligned with the upstream
        # KV-cache extra-budget contract: the sidecar's per-page bytes shrink
        # the KV block count, and its lifetime/zeroing/COW follow the owner
        # KV blocks through the generic side-storage hooks.
        self.kv_cache_requires_zeroing = True
        self.sparse_summary_cache = AscendStep5SparseSummaryCacheLayer(
            prefix=f"{prefix}.attn.sparse_summary_cache",
            target_impl=self.impl,
            sparse_config=sparse_config,
            main_layer_name=self.layer_name,
            static_forward_context=compilation_config.static_forward_context,
        )
        self.kv_cache_extra_budget_page_size_bytes = (
            self.sparse_summary_cache._set_summary_cache_budget
        )
        self.kv_cache_extra_budget_fixed_size_bytes = (
            self.sparse_summary_cache._fixed_runtime_state_budget
        )
        if cache_config is not None:
            # A prefix-cache hit replays at least one page of the prompt so
            # the recomputed tail rebuilds the summary state that the hit
            # pages alone cannot carry (the in-flight region's active slot).
            self.kv_cache_prefix_recompute_tokens = int(cache_config.block_size)
        else:
            self.kv_cache_prefix_recompute_tokens = 0


    def get_attn_backend(self) -> type[AscendStep5DSABackend]:
        return self.attn_backend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        block_size = vllm_config.cache_config.block_size
        page_budget = self.kv_cache_extra_budget_page_size_bytes
        if callable(page_budget):
            page_budget = page_budget(block_size)
        fixed_budget = self.kv_cache_extra_budget_fixed_size_bytes
        if callable(fixed_budget):
            fixed_budget = fixed_budget(block_size)
        return FullAttentionSpec(
            block_size=block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_dim,
            head_size_v=self.head_dim,
            dtype=self.kv_cache_torch_dtype,
        ).with_side_storage(
            requires_zeroing=self.kv_cache_requires_zeroing,
            extra_budget_page_size_bytes=int(page_budget or 0),
            extra_budget_fixed_size_bytes=int(fixed_budget or 0),
            prefix_cache_recompute_tokens=int(
                self.kv_cache_prefix_recompute_tokens or 0
            ),
        )

    def _apply_sparse_indexer_rope(
        self,
        positions: torch.Tensor,
        index_q: torch.Tensor,
        index_k: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        proxy_dim = int(index_q.shape[-1])
        positions = positions.reshape(-1).to(device=index_q.device, dtype=torch.long)
        q_shape = tuple(index_q.shape)
        k_shape = tuple(index_k.shape)
        q = index_q.reshape(q_shape[0], -1, proxy_dim)
        k = index_k.reshape(k_shape[0], -1, proxy_dim)
        q, k = self.sparse_indexer_rotary_emb(positions, q, k)
        return q.reshape(q_shape), k.reshape(k_shape)

    def _finish_indexer(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        index_q: torch.Tensor,
        index_k: torch.Tensor,
        index_z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Indexer norm+rope and head scores, all local shapes.

        Returns index_q [t, groups, heads_per_group, proxy], index_k [t, 1, proxy],
        index_z [t, 1, proxy], weights [t, groups, heads_per_group] fp32.
        """
        num_tokens = int(hidden_states.shape[0])
        proxy_dim = int(self.sparse_config.proxy_dim)
        weights, _ = self.sparse_indexer_w(hidden_states)

        num_index_q_heads = int(index_q.shape[-1]) // proxy_dim
        num_index_k_heads = int(index_k.shape[-1]) // proxy_dim
        heads_per_group = num_index_q_heads // self.num_kv_heads

        if self.use_fused_indexer_norm_rope:
            x_cat = torch.cat([index_q, index_k, index_z], dim=-1)
            index_q, index_k, index_z = fused_indexer_qkz_kv_norm_rope(
                x_cat,
                self.sparse_indexer_q_norm.weight,
                self.sparse_indexer_k_norm.weight,
                self._indexer_rope_cos,
                self._indexer_rope_sin,
                positions,
                q_norm_bias=None,
                k_norm_bias=self.sparse_indexer_k_norm.bias,
                head_dim=proxy_dim,
                num_q_heads=num_index_q_heads,
                num_k_heads=num_index_k_heads,
                rotary_dim=self._indexer_rotary_dim,
                eps=self.sparse_indexer_q_norm.variance_epsilon,
                q_norm_weight_bias=1.0 if self._indexer_zero_centered else 0.0,
                k_norm_weight_bias=0.0,
                subtract_mean_q=False,
                subtract_mean_k=True,
            )
            index_q = index_q.reshape(
                num_tokens, self.num_kv_heads, heads_per_group, proxy_dim
            )
            index_k = index_k.reshape(
                num_tokens, num_index_k_heads, proxy_dim
            )
            index_z = index_z.reshape(
                num_tokens, num_index_k_heads, proxy_dim
            )
        else:
            index_k = self.sparse_indexer_k_norm(index_k.contiguous())
            index_q = index_q.view(
                num_tokens, self.num_kv_heads, heads_per_group, proxy_dim
            ).contiguous()
            index_q = self.sparse_indexer_q_norm(index_q)
            index_k = index_k.view(
                num_tokens, num_index_k_heads, proxy_dim
            ).contiguous()
            index_q, index_k = self._apply_sparse_indexer_rope(
                positions,
                index_q.reshape(num_tokens, -1, proxy_dim),
                index_k,
            )
            index_q = index_q.reshape(
                num_tokens, self.num_kv_heads, heads_per_group, proxy_dim
            )
            index_z = index_z.view(
                num_tokens, num_index_k_heads, proxy_dim
            ).contiguous()

        weights = weights.to(dtype=torch.float32)
        weights = weights.view(num_tokens, self.num_kv_heads, heads_per_group)
        weights = weights * (float(heads_per_group) ** -0.5)
        return index_q, index_k, index_z, weights.contiguous()

    def _write_kv(self, key: torch.Tensor, value: torch.Tensor) -> None:
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            return
        md = attn_metadata[self.layer_name]
        num_tokens = md.num_actual_tokens
        k = key[:num_tokens].view(-1, self.num_kv_heads, self.head_dim)
        v = value[:num_tokens].view(-1, self.num_kv_heads, self.head_dim)
        if isinstance(self.kv_cache, (tuple, list)):
            key_cache, value_cache = self.kv_cache[0], self.kv_cache[1]
        else:
            key_cache, value_cache = self.kv_cache[0], self.kv_cache[1]
        DeviceOperator.reshape_and_cache(
            k, v, key_cache, value_cache, md.slot_mapping[:num_tokens]
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        reduce_scatter_output: bool = False,
    ) -> torch.Tensor:
        qkv, qkzg = self.qkv_indexer_proj(hidden_states)

        if self.use_fused_qknorm_rope:
            q, k, v = fused_qknorm_rope(
                qkv,
                self.q_norm.weight,
                self.k_norm.weight,
                self.rope_cos,
                self.rope_sin,
                positions,
                head_dim=self.head_dim,
                num_q_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                rotary_pairs=self._fused_rotary_pairs,
                eps=self.q_norm.variance_epsilon,
                norm_weight_bias=1.0 if self.zero_centered else 0.0,
            )
        else:
            q, k, v = qkv.split(
                [self.q_size, self.kv_size, self.kv_size], dim=-1
            )
            q_by_head = q.view(
                *q.shape[:-1], self.num_heads, self.head_dim
            )
            q = self.q_norm(q_by_head.contiguous()).view(q.shape)
            k_by_head = k.view(
                *k.shape[:-1], self.num_kv_heads, self.head_dim
            )
            k = self.k_norm(k_by_head.contiguous()).view(k.shape)
            if self.use_rope:
                q, k = self.rotary_emb(positions, q, k)

        index_q_raw, index_k_raw, index_z_raw, gate = (
            self.qkv_indexer_proj.split_indexer(qkzg)
        )
        index_q, index_k, index_z, weights = self._finish_indexer(
            positions, hidden_states, index_q_raw, index_k_raw, index_z_raw
        )

        num_tokens = int(hidden_states.shape[0])
        query = q.view(num_tokens, self.num_heads, self.head_dim)
        output = torch.empty(
            (num_tokens, self.num_heads, self.head_dim),
            dtype=q.dtype,
            device=q.device,
        )
        self._write_kv(k, v)
        self.impl.forward(
            self,
            query,
            self.kv_cache,
            index_q,
            index_k,
            index_z,
            weights,
            output,
            positions,
        )
        attn_output = output.view(num_tokens, self.num_heads * self.head_dim)
        if gate is not None:
            attn_output = (
                attn_output.view(num_tokens, self.num_heads, self.head_dim)
                * gate.unsqueeze(-1).sigmoid()
            ).view(num_tokens, -1)
        if reduce_scatter_output:
            output_tensor, _ = self.o_proj(
                attn_output, reduce_scatter_results=True, reduce_scatter_dim=0
            )
        else:
            output_tensor, _ = self.o_proj(attn_output)
        return output_tensor


class AscendStep5FusedMoEBlock(Step5MoE):
    """Step5 MoE block wired to the vllm-ascend fused MoE path.

    Two changes vs the upstream block (mirroring the MiniMax-M3 integration):

    - Router: no ``custom_routing_function``. A custom function forces
      vllm-ascend's router factory back onto the upstream
      ``CustomRoutingRouter`` (pure-torch greedy loop). The same semantics
      (sigmoid scoring, bias used for expert selection only, unbiased
      weights, renormalization, routed scaling) are expressed with the
      standard factory arguments so ``AscendFusedTopKRouter`` runs and the
      fused ``npu_moe_gating_top_k`` gating op is used.
    - Shared expert: built with ``reduce_results=False`` and handed to
      ``FusedMoEFactory(shared_experts=...)`` so ``AscendMoERunner`` owns
      its execution: the shared/routed sum and the TP reductions happen
      inside the runner, and the shared expert can overlap the routed
      experts on a separate stream via ``multistream_overlap_shared_expert``.
      The block's forward returns a single complete tensor instead of a
      (shared, routed) tuple.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        # Bypass the upstream __init__ entirely: the factory arguments below
        # are the whole point of this fork. Module and parameter names
        # (gate, router_bias, share_expert, experts) stay checkpoint
        # compatible with the upstream block.
        nn.Module.__init__(self)
        from vllm.model_executor.layers.fused_moe import FusedMoEFactory

        self.tp_size = get_tensor_model_parallel_world_size()
        self.layer_idx = extract_layer_index(prefix)

        self.ep_size = get_ep_group().device_group.size()
        self.ep_rank = get_ep_group().device_group.rank()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config

        self.hidden_size = config.hidden_size
        self.enable_eplb = parallel_config.enable_eplb
        self.n_routed_experts = config.moe_num_experts
        self.n_logical_experts = self.n_routed_experts
        self.n_redundant_experts = parallel_config.eplb_config.num_redundant_experts
        self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts
        self.n_local_physical_experts = self.n_physical_experts // self.ep_size

        self.physical_expert_start = self.ep_rank * self.n_local_physical_experts
        self.physical_expert_end = (
            self.physical_expert_start + self.n_local_physical_experts
        )

        if self.tp_size > config.moe_num_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.moe_num_experts}."
            )

        self.gate = Step5FP32RouterLinear(
            config.hidden_size,
            config.moe_num_experts,
            bias=False,
            quant_config=None,
            params_dtype=torch.float32,
            prefix=f"{prefix}.gate",
        )
        self.use_moe_router_bias = config.use_moe_router_bias
        if not self.use_moe_router_bias:
            raise ValueError("Step5 MoE currently requires use_moe_router_bias=true.")
        self.routed_scaling_factor = config.moe_router_scaling_factor
        self.router_bias = nn.Parameter(
            torch.zeros(config.moe_num_experts, dtype=torch.float32),
            requires_grad=False,
        )
        self.need_fp32_gate = config.need_fp32_gate
        if not self.need_fp32_gate:
            raise ValueError(
                "Step5 MoE requires need_fp32_gate=true for stable router logits."
            )

        activation = "silu"
        swiglu_limits = config.swiglu_limits or []
        swiglu_limit = (
            swiglu_limits[self.layer_idx]
            if self.layer_idx < len(swiglu_limits)
            else None
        )
        if swiglu_limit not in (None, 0):
            swiglu_limit = float(swiglu_limit)
            if swiglu_limit != 7.0:
                raise ValueError(
                    "Step5 fused MoE supports only swiglu_limit=7.0, got "
                    f"{swiglu_limit}."
                )
            activation = "swiglustep"

        # Optional Step5-native fp8 expert execution, gated by the
        # ``VLLM_ASCEND_ENABLE_STEP4_FP8`` env var (default off): the routed
        # experts keep the checkpoint's fp8 e4m3 weights and 128x128 block
        # scales and run them with the kernels vendored from the step5-hf
        # reference (Step5Fp8BlockFusedMoEMethod), instead of the default
        # AscendFp8BlockFusedMoEMethod path that resolves to bf16 or
        # requantizes to MXFP8. Layers the checkpoint left in bf16
        # (modules_to_not_convert) stay unquantized either way.
        moe_quant_config = quant_config
        enable_step5_fp8 = bool(envs.VLLM_ASCEND_ENABLE_STEP4_FP8)
        if enable_step5_fp8 and isinstance(quant_config, AscendFp8Config):
            bf16_expert_layers, _ = _step5_fp8_expert_layout(config)
            if self.layer_idx in bf16_expert_layers:
                moe_quant_config = None
            else:
                # Shallow copy re-bound to the Step5 config so the
                # framework's fp8-block parsing is preserved while the MoE
                # route changes.
                step5_config = copy.copy(quant_config)
                step5_config.__class__ = Step5Fp8BlockConfig
                moe_quant_config = step5_config
                logger.info(
                    "Layer %s: enable_step5_fp8 is set; routed experts run "
                    "the native fp8 block kernels (Step5Fp8BlockFusedMoEMethod).",
                    self.layer_idx,
                )

        share_expert_dim = config.share_expert_dim
        moe_intermediate_size = config.moe_intermediate_size
        self.fuse_all_reduce, reduce_results = _step5_moe_reduce_policy(
            self.tp_size,
            get_dp_group().world_size,
        )
        effective_sequence_parallel = (
            vllm_config.compilation_config.pass_config.enable_sp and self.tp_size > 1
        )

        # reduce_results=False: AscendMoERunner owns the shared-expert TP
        # reduction (fused into the token combine for MC2/All2All, or a
        # separate shared-only all-reduce otherwise).
        self.share_expert = Step5MLP(
            config=config,
            hidden_size=self.hidden_size,
            intermediate_size=share_expert_dim,
            hidden_act="silu",
            reduce_results=False,
            is_sequence_parallel=effective_sequence_parallel,
            quant_config=quant_config
            if quant_config and quant_config.get_name() != "fp8"
            else None,
            prefix=f"{prefix}.share_expert",
        )
        # No custom_routing_function here: it would disable the Ascend fused
        # router. scoring_func="sigmoid" + e_score_correction_bias +
        # renormalize + routed_scaling_factor reproduce router_bias_func
        # exactly (bias affects expert selection only; weights are the
        # unbiased sigmoid scores, renormalized and scaled).
        self.experts = FusedMoEFactory(
            shared_experts=self.share_expert,
            num_experts=config.moe_num_experts,
            top_k=config.moe_top_k,
            hidden_size=config.hidden_size,
            intermediate_size=moe_intermediate_size,
            reduce_results=reduce_results,
            scoring_func="sigmoid",
            renormalize=config.norm_expert_weight,
            quant_config=moe_quant_config,
            activation=activation,
            prefix=f"{prefix}.experts",
            e_score_correction_bias=self.router_bias,
            routed_scaling_factor=config.moe_router_scaling_factor,
            enable_eplb=self.enable_eplb,
            num_redundant_experts=self.n_redundant_experts,
            is_sequence_parallel=effective_sequence_parallel,
            router_logits_dtype=torch.float32,
        )
        # Ascend dispatch sizes the global physical expert space from this
        # metadata (mirrors MiniMaxM3MoE).
        self.experts.global_redundant_expert_num = self.n_redundant_experts
        self.experts.moe_config.global_redundant_expert_num = self.n_redundant_experts

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_is_sequence_parallel: bool = False,
    ) -> torch.Tensor:
        hidden_dim = hidden_states.shape[-1]
        hidden_states = hidden_states.view(-1, hidden_dim)
        # Sequence-parallel state is carried by the runner's FusedMoEConfig.
        if (
            self.experts.moe_config.is_sequence_parallel
            and not input_is_sequence_parallel
        ):
            hidden_states = sequence_parallel_chunk(hidden_states)

        if self.experts.is_internal_router:
            # Internal-router runners compute the gate themselves and the
            # runner sums the shared expert into the routed output.
            return self.experts(
                hidden_states=hidden_states, router_logits=hidden_states
            )
        router_logits, _ = self.gate(hidden_states)
        # Single complete tensor: AscendMoERunner sums the shared expert
        # output internally and applies the required TP reductions.
        return self.experts(hidden_states=hidden_states, router_logits=router_logits)


class AscendStep5Attention(Step5Attention):
    """Upstream Step5Attention with fused QKNorm+RoPE for Ascend.

    Sliding-window layers land here. The upstream forward decomposes RMSNorm
    and RoPE into ~34 eager aclnn ops because the Optimus/CuTeDSL fused
    kernels are CUDA-only. This override injects the Ascend SIMD Triton
    kernel for head_dim=192, falling back to the original eager path
    otherwise.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.use_fused_qknorm_rope = (
            os.environ.get("VLLM_STEP4_FUSE_QK_NORM_ROPE", "1") == "1"
            and getattr(self, "use_rope", False)
            and self.head_dim == 192
            and not getattr(self, "use_dsa_backend", False)
            and not getattr(self, "use_step5_fused_qknorm_cache", False)
            and not getattr(self, "use_step5_fused_qknorm", False)
        )
        if self.use_fused_qknorm_rope:
            self._fused_rotary_pairs = self.rotary_dim // 2
            rotary_cache = self.rotary_emb.cos_sin_cache
            self.rope_cos, self.rope_sin = rotary_cache.chunk(2, dim=-1)
        else:
            self._fused_rotary_pairs = 0
            self.rope_cos = None
            self.rope_sin = None

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        reduce_scatter_output: bool = False,
    ) -> torch.Tensor:
        if self.use_dsa_backend:
            qkv, qkzg = self.qkv_indexer_proj(hidden_states)
            extra_dims = None
        elif self.fuse_qkv_gate:
            qkvg, _ = self.qkvg_proj(hidden_states)
            qkv, extra_dims = qkvg.split(
                [self.q_size + 2 * self.kv_size, self.num_heads], dim=-1
            )
            qkv = qkv.contiguous()
            extra_dims = extra_dims.contiguous()
        else:
            qkv, _ = self.qkv_proj(hidden_states)
            qkzg = None
            extra_dims = None

        kv_cache_dummy_dep = None
        if self.use_fused_qknorm_rope:
            q, k, v = fused_qknorm_rope(
                qkv,
                self.q_norm.weight,
                self.k_norm.weight,
                self.rope_cos,
                self.rope_sin,
                positions,
                head_dim=self.head_dim,
                num_q_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                rotary_pairs=self._fused_rotary_pairs,
                eps=self.q_norm.variance_epsilon,
                norm_weight_bias=1.0 if self.zero_centered else 0.0,
            )
        else:
            q, k, v = qkv.split(
                [self.q_size, self.kv_size, self.kv_size], dim=-1
            )
            q_by_head = q.view(
                *q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim
            )
            q_by_head = self.q_norm(q_by_head.contiguous())
            q = q_by_head.view(q.shape)
            k_by_head = k.view(
                *k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim
            )
            k_by_head = self.k_norm(k_by_head.contiguous())
            k = k_by_head.view(k.shape)
            if self.use_rope:
                q, k = self.rotary_emb(positions, q, k)

        if self.sparse_attn is not None:
            attn_output, extra_dims = self.sparse_attn.forward(
                positions, hidden_states, q, k, v, qkzg=qkzg
            )
        else:
            attn_output = self.attn(
                q, k, v, kv_cache_dummy_dep=kv_cache_dummy_dep
            )
        if extra_dims is None and self.use_head_wise_attn_gate:
            extra_dims, _ = self.g_proj(hidden_states)

        if extra_dims is not None:
            attn_output = torch.ops.vllm.step5_materialize_gate_input(
                attn_output
            )
            extra_dims = torch.ops.vllm.step5_materialize_gate_input(
                extra_dims
            )

        if self.use_head_wise_attn_gate:
            output = (
                attn_output.view(
                    *attn_output.shape[:-1], self.num_heads, self.head_dim
                )
                * extra_dims.unsqueeze(-1).sigmoid()
            )
            attn_output = output.view(*attn_output.shape)
        if reduce_scatter_output:
            output, _ = self.o_proj(
                attn_output,
                reduce_scatter_results=True,
                reduce_scatter_dim=0,
            )
        else:
            output, _ = self.o_proj(attn_output)
        return output


class AscendStep5DecoderLayer(Step5DecoderLayer):
    """Upstream decoder layer with the attention swapped per layer type.

    DSA (full-attention) layers get the Ascend DSA module; sliding layers get
    the upstream dense Step5Attention on the platform-default backend (the
    construction the proven VLLM_STEP4_SPARSE=0 path used).
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        self.hidden_size = config.hidden_size
        self.fp32_residual_connection = config.fp32_residual_connection
        layer_idx = extract_layer_index(prefix)
        self.layer_idx = layer_idx
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        sliding_window = getattr(config, "sliding_window", None)
        if sliding_window is None:
            sliding_window = getattr(
                vllm_config.model_config.hf_text_config, "sliding_window", None
            )
        if sliding_window is None and cache_config is not None:
            sliding_window = cache_config.sliding_window
        if cache_config is not None:
            cache_config = copy.copy(cache_config)
            cache_config.sliding_window = None

        sparse_config = get_step5_sparse_config(config)
        checkpoint_has_dsa_layers = (
            checkpoint_has_step5_sparse_config(config) and sparse_config is not None
        )
        use_dsa_for_layer = (
            sparse_config is not None
            and _is_step5_full_attention_layer(config, layer_idx, sparse_config)
        )
        del checkpoint_has_dsa_layers  # dense layers must not opt into DSA wiring
        # Dense-fallback switch for A5 bring-up debugging: VLLM_STEP4_SPARSE=0
        # routes DSA layers through the platform dense backend as well.
        use_dsa_for_layer = use_dsa_for_layer and (
            os.environ.get("VLLM_STEP4_SPARSE", "1") != "0"
        )

        norm_dtype = get_step5_norm_dtype(config)
        layer_types = _get_step5_layer_types(config)
        partial_rotary_factors = getattr(config, "partial_rotary_factors", [])
        partial_rotary_factor = float(
            _per_layer_value(
                partial_rotary_factors,
                layer_idx,
                name="partial_rotary_factors",
                default=1.0,
            )
        )
        max_position = getattr(config, "max_position_embeddings", None)
        if max_position is None:
            max_position = vllm_config.model_config.max_model_len

        # Both layer families share 64 heads / 4 kv groups / head_dim 192 in
        # this checkpoint (num_attention_heads == num_sliding_attention_heads
        # == attention_other_setting), so config defaults resolve correctly.
        attn_kwargs = dict(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=max_position,
            num_kv_heads=config.num_attention_groups,
            rope_theta=config.rope_theta,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", False),
            head_dim=getattr(config, "head_dim", None),
            cache_config=cache_config,
            quant_config=quant_config,
            rope_scaling=getattr(config, "rope_scaling", None),
            sliding_window=sliding_window,
            use_head_wise_attn_gate=getattr(config, "use_head_wise_attn_gate", False),
            layer_types=layer_types,
            use_rope_layers=getattr(config, "use_rope_layers", []),
            yarn_only_types=getattr(config, "yarn_only_types", []),
            swa_num_attention_heads=getattr(
                config, "swa_num_attention_heads", None
            ),
            partial_rotary_factor=partial_rotary_factor,
            prefix=f"{prefix}.self_attn",
            zero_centered=config.zero_centered,
            vllm_config=vllm_config,
            norm_dtype=norm_dtype,
        )
        if use_dsa_for_layer:
            self.self_attn = AscendStep5DSAAttention(
                sparse_config=sparse_config,
                **attn_kwargs,
            )
        else:
            self.self_attn = AscendStep5Attention(
                sparse_config=None,
                model_has_dsa_layers=False,
                **attn_kwargs,
            )

        self.use_moe = False
        self.tp_group = get_tp_group()
        self.use_fused_all_reduce = (
            get_tensor_model_parallel_world_size() > 1
            and get_dp_group().world_size == 1
        )

        moe_layers_idx = _get_step5_moe_layer_indices(config)
        if layer_idx in moe_layers_idx:
            self.moe = AscendStep5FusedMoEBlock(
                vllm_config,
                prefix=f"{prefix}.moe",
            )
            self.use_moe = True
        else:
            self.mlp = Step5MLP(
                config=config,
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act="silu",
                quant_config=quant_config
                if quant_config and quant_config.get_name() != "fp8"
                else None,
                reduce_results=True,
                prefix=f"{prefix}.mlp",
            )
        self.input_layernorm = make_step5_rms_norm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            zero_centered=config.zero_centered,
            dtype=norm_dtype,
        )
        self.post_attention_layernorm = make_step5_rms_norm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            zero_centered=config.zero_centered,
            dtype=norm_dtype,
        )
        self.prefix = prefix
        self.use_attention_o_proj_reduce_scatter = False


    def _forward_ffn(
        self,
        hidden_states: torch.Tensor,
        input_is_sequence_parallel: bool = False,
        residual: torch.Tensor | None = None,
        orig_num_tokens: int | None = None,
    ) -> torch.Tensor:
        if not self.use_moe or not isinstance(self.moe, AscendStep5FusedMoEBlock):
            return super()._forward_ffn(
                hidden_states,
                input_is_sequence_parallel=input_is_sequence_parallel,
                residual=residual,
                orig_num_tokens=orig_num_tokens,
            )
        # AscendStep5FusedMoEBlock returns a single complete tensor: the
        # Ascend MoE combine kernel reduces the routed output across TP and
        # the runner reduces the shared-expert output to match, so no extra
        # all-reduce may be applied here (reducing the sum would count the
        # routed contribution once per TP rank, measured 8x).
        ffn_output = self.moe(
            hidden_states, input_is_sequence_parallel=input_is_sequence_parallel
        )
        # On Ascend the MoE comm impls (MC2 / AllGather / All2All) fuse the
        # EP/TP reduction into the token combine, so the routed output is
        # already complete even though FusedMoEBlock requested
        # reduce_results=False (upstream expects a partial here and reduces
        # shared + routed together). Only the shared expert (RowParallelLinear
        # with reduce_results=False) is partial w.r.t. TP: reduce it alone,
        # then add the complete routed output. Reducing the sum instead would
        # count the routed contribution once per TP rank (measured 8x).
        if self.moe.experts.moe_config.is_sequence_parallel:
            # Sequence-parallel outputs are token shards; all-gather like
            # the upstream layer does for the (shared, routed) tuple.
            if input_is_sequence_parallel:
                assert residual is not None
                assert orig_num_tokens is not None
                gathered = tensor_model_parallel_all_gather(
                    self._cast_for_residual(ffn_output) + residual, dim=0
                )
                return gathered[:orig_num_tokens]
            gathered = tensor_model_parallel_all_gather(
                self._cast_for_residual(ffn_output), dim=0
            )
            return gathered[: hidden_states.shape[0]]
        return self._cast_for_residual(ffn_output)

    def forward_attention(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """self_attn + fused post-attention residual-add/RMSNorm/cast.

        Takes the normed hidden states (param-op dtype) and the running
        residual (residual dtype), returns the post-attention normed hidden
        states and the updated residual.
        """
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )
        return fused_add_rms_norm_cast(
            hidden_states,
            residual,
            self.post_attention_layernorm.weight,
            self.post_attention_layernorm.variance_epsilon,
            zero_centered=getattr(
                self.post_attention_layernorm, "zero_centered", False
            ),
            fp32_residual=self.fp32_residual_connection,
        )

    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        # The reduce-scatter path needs the sequence-parallel-chunked residual
        # before the norm, which the fused kernel does not cover; keep the
        # upstream sequence there. (This layer never enables it on Ascend.)
        if self.use_attention_o_proj_reduce_scatter and hidden_states.dim() == 2:
            return super().forward(positions, hidden_states)
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self._cast_for_param_op(hidden_states)

        hidden_states, residual = self.forward_attention(
            positions, hidden_states, residual
        )

        ffn_output = self._forward_ffn(hidden_states)
        ffn_output = self._cast_for_residual(ffn_output)
        return ffn_output + residual



class AscendStep5Model(Step5Model):
    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        # The @support_torch_compile wrapper on the upstream __init__ sets
        # this; bypassing it here means we set it ourselves. The DSA modules
        # hold python-side per-request state, so this fork stays eager.
        self.do_not_compile = True
        self.vllm_config = vllm_config
        config = vllm_config.model_config.hf_config
        self.vocab_size = config.vocab_size
        self.config = config
        self.fp32_residual_connection = config.fp32_residual_connection
        self.moe_num_experts = config.moe_num_experts
        self.parallel_config = vllm_config.parallel_config

        if get_pp_group().is_first_rank or (
            config.tie_word_embeddings and get_pp_group().is_last_rank
        ):
            self.embed_tokens = VocabParallelEmbedding(
                self.vocab_size,
                config.hidden_size,
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: AscendStep5DecoderLayer(
                vllm_config,
                prefix=prefix,
            ),
            prefix=f"{prefix}.layers",
        )
        if get_pp_group().is_last_rank:
            norm_dtype = get_step5_norm_dtype(config)
            self.norm = make_step5_rms_norm(
                config.hidden_size,
                eps=config.rms_norm_eps,
                zero_centered=config.zero_centered,
                dtype=norm_dtype,
            )
        else:
            self.norm = PPMissingLayer()

        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states"], config.hidden_size
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
        residual = self._cast_for_residual(hidden_states)

        # The per-layer tail (cast the FFN output to the residual dtype and
        # add the residual) is deferred and fused with the *next* layer's
        # input RMSNorm + param-op cast in one kernel, so each layer
        # boundary costs a single fused launch instead of five.
        ffn_output: torch.Tensor | None = None
        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            if ffn_output is None:
                # First layer only: the input norm is not covered by the
                # fused boundary kernel. For zero-centered norms use the
                # fused CANN op (scale = weight + 1) instead of the eager
                # fp32 pow/mean/rsqrt/mul sequence. Input and weight share
                # the residual/norm dtype here, satisfying the op's
                # same-dtype constraint.
                if getattr(layer.input_layernorm, "zero_centered", False):
                    hidden_states = DeviceOperator.npu_gemma_rms_norm(
                        residual,
                        layer.input_layernorm.weight,
                        layer.input_layernorm.variance_epsilon,
                    )
                else:
                    hidden_states = layer.input_layernorm(residual)
                hidden_states = layer._cast_for_param_op(hidden_states)
            else:
                hidden_states, residual = fused_add_rms_norm_cast(
                    ffn_output,
                    residual,
                    layer.input_layernorm.weight,
                    layer.input_layernorm.variance_epsilon,
                    zero_centered=getattr(
                        layer.input_layernorm, "zero_centered", False
                    ),
                    fp32_residual=self.fp32_residual_connection,
                )
            hidden_states, residual = layer.forward_attention(
                positions, hidden_states, residual
            )
            ffn_output = layer._forward_ffn(hidden_states)
        hidden_states = self._cast_for_residual(ffn_output) + residual

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {
                    "hidden_states": hidden_states,
                }
            )

        return hidden_states


class AscendStep5ForCausalLM(Step5ForCausalLM):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        nn.Module.__init__(self)
        self.vllm_config = vllm_config
        model_config = vllm_config.model_config
        valid_vocab_size = _require_resolved_valid_vocab_size(model_config)
        config = model_config.hf_config
        self.config = config
        self.fp32_residual_connection = config.fp32_residual_connection
        self.model = AscendStep5Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=vllm_config.quant_config
                if vllm_config.quant_config
                and vllm_config.quant_config.get_name() != "fp8"
                else None,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            if config.tie_word_embeddings:
                self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
            self.logits_processor = LogitsProcessor(
                config.vocab_size,
                valid_vocab_size=valid_vocab_size,
            )
        else:
            self.lm_head = PPMissingLayer()

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

        # Set MoE hyperparameters
        self.moe_layers: list[Any] = []
        example_layer: Step5MoE | None = None
        for layer in self.model.layers:
            if isinstance(layer, PPMissingLayer):
                continue
            if hasattr(layer, "moe") and isinstance(layer.moe, Step5MoE):
                example_layer = layer.moe
                self.moe_layers.append(layer.moe.experts)
        _set_step5_moe_protocol_metadata(self, example_layer)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.fp32_residual_connection:
            hidden_states = hidden_states.to(torch.bfloat16)
        norm = self.model.norm
        if getattr(norm, "zero_centered", False):
            # Same-dtype constraint of the fused CANN op: the bf16-rounded
            # hidden states upcast to fp32 exactly, keeping the fp32 weight
            # math of the unfused path (verified bitwise-equal to it after
            # the bf16 output cast).
            hidden_states = DeviceOperator.npu_gemma_rms_norm(
                hidden_states.float(), norm.weight, norm.variance_epsilon
            ).to(torch.bfloat16)
        else:
            hidden_states = norm(hidden_states)
        return self.logits_processor(self.lm_head, hidden_states)
