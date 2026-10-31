# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Step4 device-tiling FA3, adapted to the Model Runner V2 lifecycle.

Based on Step4 commits 6d35bde5 and be2cce4e. Pool capacities are bound to
each metadata builder after allocation, never stored in a process global.
"""

import inspect
from dataclasses import dataclass, field, replace

import torch
from flash_attn_npu_3 import flash_attn_with_kvcache, get_scheduler_metadata
from vllm.config import VllmConfig
from vllm.v1.attention.backend import AttentionCGSupport, AttentionMetadataBuilder, AttentionType
from vllm.v1.attention.backends.registry import register_backend
from vllm.v1.kv_cache_interface import AttentionSpec, FullAttentionSpec

from vllm_ascend.attention.attention_v1 import (
    AscendAttentionBackend,
    AscendAttentionBackendImpl,
    AscendMetadata,
)
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.core.step5_kv_cache import _validate_step5_blnhc_spec, uses_step5_blnhc
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.attention_fence import record_attention_compute_start


def validate_fa3_kv_transfer_config(vllm_config: VllmConfig) -> None:
    """Limit PD to Step5's byte-compatible BLNHC main K/V planes."""
    transfer = vllm_config.kv_transfer_config
    if transfer is None or not getattr(transfer, "is_kv_transfer_instance", False) or not uses_step5_blnhc(vllm_config):
        raise ValueError("Device-tiling FA3 KV transfer requires the Step5 V2 Mooncake pull profile.")
    model = vllm_config.model_config
    cache = vllm_config.cache_config
    if model.dtype != torch.bfloat16 or cache.cache_dtype not in ("auto", "bfloat16"):
        raise ValueError("Device-tiling FA3 Step5 KV transfer requires BF16 model and main KV cache.")
    # Model construction precedes upstream layout resolution. BLNHC-only backend
    # capability and descriptor validation enforce the eventual physical layout.
    if getattr(cache, "kv_cache_layout", None) not in (None, "BLNHC"):
        raise ValueError("Device-tiling FA3 Step5 KV transfer requires BLNHC.")
    head_size = model.get_head_size()
    _validate_step5_blnhc_spec(
        FullAttentionSpec(
            block_size=cache.block_size,
            num_kv_heads=model.get_num_kv_heads(vllm_config.parallel_config),
            head_size=head_size,
            head_size_v=head_size,
            dtype=model.dtype,
        )
    )


def validate_fa3_scheduler_api() -> None:
    """Refuse the older scheduler ABI instead of silently losing accuracy."""
    required = {"num_blocks", "max_num_blocks_per_seq", "softmax_scale"}
    missing = required - inspect.signature(get_scheduler_metadata).parameters.keys()
    if missing:
        raise RuntimeError(
            "flash_attn_npu_3 lacks the FA3 accuracy-fix scheduler API: "
            f"{sorted(missing)}. Rebuild the operator with explicit KV pool capacity support."
        )
    if getattr(flash_attn_with_kvcache, "block_stride_abi_version", 0) != 1:
        raise RuntimeError("FA3 requires the compiled paged block-stride ABI; rebuild the operator")


@register_backend("ASCEND_DEVICE_FA3")
class AscendFlashAttentionBackend(AscendAttentionBackend):
    uses_device_fa3 = True
    supported_dtypes = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes = ["auto", "float16", "bfloat16"]

    @classmethod
    def supports_sliding_window(cls) -> bool:
        return True

    @classmethod
    def supported_kv_cache_layouts(cls):
        from vllm.config import get_current_vllm_config_or_none
        from vllm.v1.kv_cache_layout import KVCacheLayout

        from vllm_ascend.core.step5_kv_cache import uses_step5_blnhc

        config = get_current_vllm_config_or_none()
        if config is not None and uses_step5_blnhc(config):
            return (KVCacheLayout.BLNHC,)
        return (KVCacheLayout.LBNHC, KVCacheLayout.BLNHC)

    @classmethod
    def customize_spec(cls, spec):
        from vllm.config import get_current_vllm_config_or_none

        from vllm_ascend.core.step5_kv_cache import uses_step5_blnhc

        config = get_current_vllm_config_or_none()
        if config is not None and uses_step5_blnhc(config):
            # The native address calculation divides the global tile stride
            # by one dense BNHD row. Publish that constraint to main's planner.
            alignment = spec.num_kv_heads * spec.head_size * spec.dtype.itemsize
            return replace(spec, block_stride_alignment=alignment)
        return spec

    @staticmethod
    def get_name() -> str:
        return "ASCEND_DEVICE_FA3"

    @staticmethod
    def get_impl_cls() -> type["AscendFlashAttentionImpl"]:
        return AscendFlashAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["AscendFlashAttentionMetadataBuilder"]:
        return AscendFlashAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "",
    ) -> tuple[int, ...]:
        # FA3's paged ABI is always BNHD, independent of FIA's HND preference.
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec=None) -> list[int]:
        return [128]

    @classmethod
    def supports_kv_connector(cls) -> bool:
        from vllm.config import get_current_vllm_config_or_none

        config = get_current_vllm_config_or_none()
        if config is None or config.kv_transfer_config is None:
            return False
        try:
            validate_fa3_kv_transfer_config(config)
        except ValueError:
            return False
        return True


@dataclass
class AscendFlashAttentionMetadata(AscendMetadata):
    scheduler_metadata: dict[tuple, torch.Tensor] = field(default_factory=dict)
    uses_device_fa3: bool = True


class AscendFlashAttentionMetadataBuilder(AttentionMetadataBuilder[AscendFlashAttentionMetadata]):
    reorder_batch_threshold = None
    supports_padded_decode_graph = True

    def __init__(
        self, kv_cache_spec: AttentionSpec, layer_names: list[str], vllm_config: VllmConfig, device: torch.device
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        validate_fa3_scheduler_api()
        if vllm_config.kv_transfer_config is not None:
            validate_fa3_kv_transfer_config(vllm_config)
            _validate_step5_blnhc_spec(kv_cache_spec)
            if kv_cache_spec.head_size_v == 0:
                raise ValueError("Device-tiling FA3 Step5 KV transfer requires a main K/V attention spec.")
        self.max_num_reqs = vllm_config.scheduler_config.max_num_seqs + 1
        self.block_size = kv_cache_spec.block_size
        self.scheduler_specs: set[tuple] = set()
        self.scheduler_buffers: dict[tuple, torch.Tensor] = {}
        self.graph_buffers: dict[tuple, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self.graph_slot_buffers: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
        self.capture_keys = set()
        self.capture_scheduler_keys = set()
        self.preparing_capture = False

    def bind_fa3_kv_cache(self, forward_context: dict) -> None:
        """Bind actual kernel-block counts for target and draft pools.

        MRV2 can split a manager block into several kernel blocks. The logical
        KVCacheConfig count and block-table width are NOT physical pool sizes.
        """
        specs = set()
        for name in self.layer_names:
            layer = (
                forward_context[name]
                if name in forward_context
                else self.vllm_config.compilation_config.static_forward_context[name]
            )
            impl = layer.impl
            key_cache, value_cache = layer.kv_cache
            for cache in (key_cache, value_cache):
                if cache.dim() != 4 or cache.shape[0] <= 0:
                    raise ValueError("FA3 requires nonempty paged BNHD cache views")
                row = cache.shape[2] * cache.shape[3]
                if (
                    cache.stride(3) != 1
                    or cache.stride(2) != cache.shape[3]
                    or cache.stride(1) != row
                    or cache.stride(0) < cache.shape[1] * row
                    or cache.stride(0) % row
                ):
                    raise ValueError("FA3 supports dense BNHD rows with bounded block gaps only")
            impl.key_cache, impl.value_cache = key_cache, value_cache
            specs.add(impl.scheduler_key(key_cache.dtype))
        self.scheduler_specs = specs
        # Profiling and real allocation own different cache lifetimes.
        self.graph_buffers.clear()
        self.graph_slot_buffers.clear()
        self.scheduler_buffers.clear()
        self.capture_keys.clear()
        self.capture_scheduler_keys.clear()

    @classmethod
    def get_cudagraph_support(cls, vllm_config: VllmConfig, kv_cache_spec: AttentionSpec) -> AttentionCGSupport:
        return AttentionCGSupport.ALWAYS

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
        fast_build: bool = False,
        *,
        num_actual_reqs: int | None = None,
        full_graph_mode: bool = False,
    ) -> AscendFlashAttentionMetadata:
        if not self.scheduler_specs:
            raise RuntimeError("FA3 metadata requires KV pools to be bound before warmup or graph capture.")
        common = common_attn_metadata
        num_reqs = common.num_reqs if num_actual_reqs is None else num_actual_reqs
        if not 0 <= num_reqs <= common.num_reqs <= self.max_num_reqs:
            raise ValueError("FA3 request count exceeds the configured batch capacity.")
        if common.query_start_loc.shape[0] < num_reqs + 1:
            raise ValueError("FA3 query boundaries must include every real request.")
        if min(common.seq_lens.shape[0], common.block_table_tensor.shape[0]) < num_reqs:
            raise ValueError("FA3 sequence lengths and block tables must include every real request.")
        num_input_tokens = common.num_input_tokens or common.num_actual_tokens
        # MRV2 resolves capture sizes after constructing metadata builders.
        graph_shape = full_graph_mode
        # Capture uses dummy block tables; replay gathers real ones. Pointer
        # identity cannot select the buffers a graph has already captured.
        # Each target/draft builder owns its own slots. Multi-module MTP
        # keeps Q/KV lengths fixed across its independent module forwards.
        source_key = (
            ("graph", num_input_tokens, common.block_table_tensor.shape[1])
            if graph_shape
            else (
                common.query_start_loc.data_ptr(),
                common.seq_lens.data_ptr(),
                common.block_table_tensor.data_ptr(),
                common.block_table_tensor.shape[1],
            )
        )
        if graph_shape and not self.preparing_capture and source_key not in self.capture_keys:
            raise RuntimeError("FA3 FULL replay requires a registered capture slot")
        if source_key not in self.graph_buffers:
            self.graph_buffers[source_key] = (
                torch.empty(self.max_num_reqs + 1, dtype=torch.int32, device=self.device),
                torch.empty(self.max_num_reqs, dtype=torch.int32, device=self.device),
                torch.empty(
                    (self.max_num_reqs, common.block_table_tensor.shape[1]), dtype=torch.int32, device=self.device
                ),
            )
        query_start_loc, seq_lens, block_tables = self.graph_buffers[source_key]
        query_start_loc[: num_reqs + 1].copy_(common.query_start_loc[: num_reqs + 1])
        # Use the device boundary, not a possibly padded Python token count.
        # This keeps padded requests empty during adaptive/MTP graph replay.
        query_start_loc[num_reqs + 1 :].copy_(
            query_start_loc[num_reqs : num_reqs + 1].expand(self.max_num_reqs - num_reqs)
        )
        seq_lens.zero_()
        seq_lens[:num_reqs].copy_(common.seq_lens[:num_reqs])
        block_tables.zero_()
        block_tables[:num_reqs].copy_(common.block_table_tensor[:num_reqs])
        slot_mapping = common.slot_mapping
        if graph_shape:
            if (
                slot_mapping.ndim != 1
                or slot_mapping.numel() < num_input_tokens
                or slot_mapping.dtype not in (torch.int32, torch.int64)
                or slot_mapping.device != self.device
            ):
                raise ValueError("FA3 FULL writer slots must cover the captured token extent on the builder device")
            if source_key not in self.graph_slot_buffers:
                self.graph_slot_buffers[source_key] = (
                    torch.empty(num_input_tokens, dtype=slot_mapping.dtype, device=self.device),
                    torch.arange(num_input_tokens, device=self.device),
                )
            writer_slots, token_indices = self.graph_slot_buffers[source_key]
            if writer_slots.dtype != slot_mapping.dtype:
                raise ValueError("FA3 FULL writer slot dtype changed after capture")
            # The graph captures the inherited scatter writer as well as the
            # FA3 reader. Refresh both from arbitrary sources without replacing
            # their captured addresses; stale PAD slots must never write pages.
            writer_slots.copy_(slot_mapping[:num_input_tokens])
            writer_slots.masked_fill_(token_indices >= query_start_loc[num_reqs], -1)
            slot_mapping = writer_slots
        max_query_len = num_input_tokens if graph_shape else common.max_query_len
        scheduler_metadata = {}
        for spec in self.scheduler_specs:
            (
                num_heads,
                num_kv_heads,
                head_size,
                head_size_v,
                dtype,
                scale,
                softcap,
                window,
                num_blocks,
                _k_stride,
                _v_stride,
            ) = spec
            tiling = get_scheduler_metadata(
                batch_size=self.max_num_reqs,
                max_seqlen_q=max_query_len,
                max_seqlen_k=block_tables.shape[1] * self.block_size,
                num_heads_q=num_heads,
                num_heads_kv=num_kv_heads,
                headdim=head_size,
                headdim_v=head_size_v,
                cache_seqlens=seq_lens,
                qkv_dtype=dtype,
                cu_seqlens_q=query_start_loc,
                page_size=self.block_size,
                num_blocks=num_blocks,
                max_num_blocks_per_seq=block_tables.shape[1],
                causal=common.causal,
                softmax_scale=scale,
                has_softcap=softcap != 0.0,
                num_splits=0,
                window_size=window,
            )
            if graph_shape:
                buffer_key = (source_key, spec, max_query_len, common.causal)
                if not self.preparing_capture and buffer_key not in self.capture_scheduler_keys:
                    raise RuntimeError("FA3 FULL replay scheduler identity was not captured")
                if buffer_key not in self.scheduler_buffers:
                    self.scheduler_buffers[buffer_key] = tiling
                else:
                    self.scheduler_buffers[buffer_key].copy_(tiling)
                tiling = self.scheduler_buffers[buffer_key]
                if self.preparing_capture:
                    self.capture_scheduler_keys.add(buffer_key)
            scheduler_metadata[spec] = tiling
        return AscendFlashAttentionMetadata(
            num_actual_tokens=common.num_actual_tokens,
            query_start_loc=query_start_loc,
            query_start_loc_gpu=query_start_loc,
            seq_lens=seq_lens,
            seq_lens_gpu=seq_lens,
            block_tables=block_tables,
            slot_mapping=slot_mapping,
            max_query_len=max_query_len,
            attn_state=common.attn_state,
            causal=common.causal,
            model_runner_type=self.vllm_config.model_config.runner_type,
            scheduler_metadata=scheduler_metadata,
        )

    def build_for_cudagraph_capture(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
        **kwargs,
    ) -> AscendFlashAttentionMetadata:
        kwargs["full_graph_mode"] = True
        self.preparing_capture = True
        try:
            result = self.build(0, common_attn_metadata, **kwargs)
            num_tokens = common_attn_metadata.num_input_tokens or common_attn_metadata.num_actual_tokens
            self.capture_keys.add(("graph", num_tokens, common_attn_metadata.block_table_tensor.shape[1]))
            return result
        finally:
            self.preparing_capture = False


class AscendFlashAttentionImpl(AscendAttentionBackendImpl):
    def __init__(self, *args, **kwargs):
        params = inspect.signature(AscendAttentionBackendImpl.__init__).bind(self, *args, **kwargs)
        super().__init__(*args, **kwargs)
        if not self.vllm_config.use_v2_model_runner:
            raise ValueError("Device-tiling FA3 is adapted to Model Runner V2; enable the V2 runner.")
        if self.enable_c8_quant or self.kv_cache_dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("Device-tiling FA3 requires FP16/BF16 KV cache.")
        if self.attn_type != AttentionType.DECODER or self.alibi_slopes is not None or self.sinks is not None:
            raise ValueError("Device-tiling FA3 supports decoder attention without ALiBi or sinks.")
        parallel = self.vllm_config.parallel_config
        if parallel.prefill_context_parallel_size != 1 or parallel.decode_context_parallel_size != 1:
            raise ValueError("Device-tiling FA3 does not support context parallelism.")
        if self.vllm_config.kv_transfer_config is not None:
            validate_fa3_kv_transfer_config(self.vllm_config)
            main_spec = FullAttentionSpec(
                block_size=self.vllm_config.cache_config.block_size,
                num_kv_heads=self.num_kv_heads,
                head_size=self.head_size,
                head_size_v=self.head_size,
                dtype=self.kv_cache_dtype,
            )
            _validate_step5_blnhc_spec(main_spec)
        if self.sliding_window is not None and self.sliding_window <= 0:
            raise ValueError("FA3 sliding_window must be positive or None.")
        self.window_size = (self.sliding_window - 1, 0) if self.sliding_window is not None else (-1, -1)
        self.logits_soft_cap = params.arguments.get("logits_soft_cap") or 0.0
        self.use_bnsd_kv_cache = False

    def scheduler_key(self, dtype: torch.dtype) -> tuple:
        return (
            self.num_heads,
            self.num_kv_heads,
            self.head_size,
            self.value_cache.shape[-1],
            dtype,
            self.scale,
            self.logits_soft_cap,
            self.window_size,
            self.key_cache.shape[0],
            self.key_cache.stride(0),
            self.value_cache.stride(0),
        )

    @staticmethod
    def update_graph_params(*args, **kwargs):
        # Device lengths/tiling are refreshed by the builder before replay.
        pass

    def forward_impl(self, query, key, value, kv_cache, attn_metadata, output):
        record_attention_compute_start()
        num_tokens = attn_metadata.num_actual_tokens
        result = flash_attn_with_kvcache(
            query,
            self.key_cache,
            self.value_cache,
            cache_seqlens=attn_metadata.seq_lens,
            page_table=attn_metadata.block_tables,
            cu_seqlens_q=attn_metadata.query_start_loc,
            max_seqlen_q=attn_metadata.max_query_len,
            softmax_scale=self.scale,
            causal=attn_metadata.causal,
            softcap=self.logits_soft_cap,
            scheduler_metadata=attn_metadata.scheduler_metadata[self.scheduler_key(query.dtype)],
            window_size=self.window_size,
            num_splits=0,
        )
        output[:num_tokens].copy_(result[:num_tokens])
        return output
