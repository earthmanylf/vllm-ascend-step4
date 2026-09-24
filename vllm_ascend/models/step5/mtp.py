# SPDX-License-Identifier: Apache-2.0
"""Ascend Step-4 MTP draft model.

The upstream ``Step5MTP`` is CUDA-only (its ``__init__`` raises on
non-CUDA platforms and its predictor blocks build the CUDA
``Step5DecoderLayer``).  This fork keeps the upstream wire format and
interfaces (``load_weights`` is inherited untouched; ``compute_logits``
and ``get_top_tokens`` keep the upstream behavior with a fused fast
path) and re-creates the ``__init__`` paths so
the draft transformer block runs through ``AscendStep5DecoderLayer`` --
i.e. the platform Ascend attention backends plus the NPU MoE/rope fixes
already proven on the target model.  The MTP layer is a dense sliding
layer, so it lands on the platform sliding-window backend, not DSA.

Weight loading, module names and parameter names mirror the upstream
implementation exactly, so the checkpoint's ``model.layers.92.*`` MTP
weights load unchanged.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)
from vllm.models.step5.layernorm import get_step5_norm_dtype
from vllm.models.step5.mtp import (
    Step5MTPSharedHead,
    Step5MTP,
    Step5MultiTokenPredictor,
    Step5MultiTokenPredictorLayer,
    _get_mtp_config,
    _get_mtp_vllm_config,
    _get_step5_mtp_moe_blocks,
    _require_resolved_valid_vocab_size,
    _set_step5_moe_protocol_metadata,
)
from vllm.model_executor.models.utils import maybe_prefix

from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.models.step5.fused_add_rms_norm import fused_add_rms_norm_cast
from vllm_ascend.models.step5.model import AscendStep5DecoderLayer
from vllm_ascend.ops.layernorm import AscendGemmaRMSNorm, AscendRMSNorm


class AscendStep5MTPGemmaNorm(AscendGemmaRMSNorm):
    """Zero-centered MTP enorm/hnorm on the fused ``aclnnGemmaRmsNorm`` op.

    Computes x * rsqrt(mean(x^2) + eps) * (1 + w), matching the upstream
    ``Step5RMSNorm`` fallback (fp32 normalize, fp32 ``(w + 1)`` scale,
    output cast back to the input dtype) in a single fused kernel. The
    fused CANN op requires x and weight to share a dtype, so the input is
    upcast to the checkpoint's norm dtype (fp32; the bf16 -> fp32 upcast
    is exact) and the result is cast back -- byte-for-byte the eager
    fallback's compute/cast sequence, the same pattern the main model's
    ``compute_logits`` uses for its zero-centered norms.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float,
        norm_dtype: torch.dtype,
    ) -> None:
        super().__init__(hidden_size, eps=eps)
        self.zero_centered = True
        # GemmaRMSNorm builds `weight` in the ambient default dtype; pin the
        # checkpoint's norm dtype so the fused op keeps fp32 weight math.
        self.weight = nn.Parameter(self.weight.detach().to(norm_dtype))

    def forward_oot(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is not None:
            return super().forward_oot(x, residual)
        orig_dtype = x.dtype
        if x.dtype != self.weight.dtype:
            x = x.to(self.weight.dtype)
        out = DeviceOperator.npu_gemma_rms_norm(
            x, self.weight, self.variance_epsilon
        )
        return out.to(orig_dtype) if out.dtype != orig_dtype else out


class AscendStep5MTPRMSNorm(AscendRMSNorm):
    """Non-zero-centered MTP norm on the fused ``npu_rms_norm`` op.

    Computes x * rsqrt(mean(x^2) + eps) * w with the same dtype
    upcast/downcast contract as ``AscendStep5MTPGemmaNorm`` so both norm
    flavors keep the eager fallback's fp32 math.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float,
        norm_dtype: torch.dtype,
    ) -> None:
        super().__init__(hidden_size, eps=eps, dtype=norm_dtype)
        self.zero_centered = False

    def forward_oot(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        import torch_npu

        if residual is not None:
            return super().forward_oot(x, residual)
        orig_dtype = x.dtype
        if x.dtype != self.weight.dtype:
            x = x.to(self.weight.dtype)
        out, _ = torch_npu.npu_rms_norm(
            x, self.weight, self.variance_epsilon
        )
        return out.to(orig_dtype) if out.dtype != orig_dtype else out


def _build_ascend_mtp_norm(config: Any) -> nn.Module:
    """MTP enorm/hnorm on the fused NPU norm ops.

    Drop-in replacement for the upstream ``_build_mtp_norm``: selects the
    zero-centered (Gemma, ``w + 1``) or plain variant to preserve the
    reference semantics while running one fused kernel per call. The
    parameter keeps the name ``weight`` so the checkpoint's
    ``enorm.weight`` / ``hnorm.weight`` load unchanged, and the
    ``variance_epsilon`` / ``zero_centered`` attributes stay accessible.
    """
    norm_dtype = get_step5_norm_dtype(config)
    if bool(getattr(config, "zero_centered", True)):
        return AscendStep5MTPGemmaNorm(
            config.hidden_size, config.rms_norm_eps, norm_dtype
        )
    return AscendStep5MTPRMSNorm(
        config.hidden_size, config.rms_norm_eps, norm_dtype
    )


class AscendStep5MTPDecoderLayer(AscendStep5DecoderLayer):
    """MTP draft block that defers the FFN-tail residual add.

    Identical to ``AscendStep5DecoderLayer.forward`` except the closing
    ``_cast_for_residual(ffn_output) + residual`` is skipped: the
    un-summed ``(ffn_output, residual)`` pair is returned so the caller
    can fold the add into the shared-head final RMSNorm with
    ``fused_add_rms_norm_cast`` (the kernel upcasts to fp32 internally,
    matching what ``_cast_for_residual`` would produce).  This mirrors
    the main model, which passes the raw FFN output across the layer
    boundary for the same fusion.
    """

    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # The reduce-scatter path needs the sequence-parallel-chunked
        # residual before the norm, which the fused kernel does not cover;
        # fall back to the summed upstream sequence there (never enabled on
        # Ascend today, see AscendStep5DecoderLayer.forward).
        if self.use_attention_o_proj_reduce_scatter and hidden_states.dim() == 2:
            return super().forward(positions, hidden_states)
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self._cast_for_param_op(hidden_states)

        hidden_states, residual = self.forward_attention(
            positions, hidden_states, residual
        )

        ffn_output = self._forward_ffn(hidden_states)
        return ffn_output, residual


class AscendStep5MultiTokenPredictorLayer(Step5MultiTokenPredictorLayer):
    """Upstream predictor layer with the draft block swapped to Ascend."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
    ) -> None:
        # Bypass the parent __init__: it hardcodes the CUDA
        # Step5DecoderLayer.  Everything else mirrors it verbatim so
        # module/parameter names stay checkpoint-compatible.
        nn.Module.__init__(self)
        # The parent's @support_torch_compile wrapper reads this attribute
        # in __call__; the attribute itself is normally set by the
        # decorator's __init__ patch, which the bypass above skips.
        self.do_not_compile = True
        mtp_vllm_config = _get_mtp_vllm_config(vllm_config)
        config = mtp_vllm_config.model_config.hf_text_config
        self.fp32_residual_connection = config.fp32_residual_connection
        quant_config = vllm_config.quant_config
        self.enorm = _build_ascend_mtp_norm(config)
        self.hnorm = _build_ascend_mtp_norm(config)
        self.eh_proj = nn.Linear(
            config.hidden_size * 2, config.hidden_size, bias=False
        )
        self.shared_head = Step5MTPSharedHead(
            config=config, quant_config=quant_config, prefix=f"{prefix}.shared_head"
        )
        self.mtp_block = AscendStep5MTPDecoderLayer(
            mtp_vllm_config,
            prefix=f"{prefix}.mtp_block",
        )
        # Fused add+norm hand-off to compute_logits (see forward): the
        # normed hidden produced by fused_add_rms_norm_cast and the summed
        # hidden returned to the caller, so the shared-head norm is
        # applied exactly once.  Plain attributes: they hold graph-pool
        # tensors and must stay out of state_dict.
        self._fused_normed: torch.Tensor | None = None
        self._fused_summed: torch.Tensor | None = None

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_index: int = 0,
    ) -> torch.Tensor:
        # Upstream forward, except the draft block's tail residual add and
        # the shared-head final RMSNorm (deferred by the caller upstream to
        # Step5MTPSharedHead) are folded into one fused_add_rms_norm_cast
        # kernel here.
        assert inputs_embeds is not None
        if self.fp32_residual_connection:
            inputs_embeds = inputs_embeds.to(torch.bfloat16)
            previous_hidden_states = previous_hidden_states.to(torch.bfloat16)
        inputs_embeds = self.enorm(inputs_embeds)
        previous_hidden_states = self.hnorm(previous_hidden_states)

        hidden_states = self.eh_proj(
            torch.cat([inputs_embeds, previous_hidden_states], dim=-1)
        )
        if self.fp32_residual_connection:
            hidden_states = hidden_states.to(torch.float32)

        block_out = self.mtp_block(positions=positions, hidden_states=hidden_states)
        if not isinstance(block_out, tuple):
            # Reduce-scatter fallback ran the summed upstream sequence;
            # keep the upstream wire format (shared_head norms later).
            self._fused_normed = None
            self._fused_summed = None
            return block_out
        ffn_output, residual = block_out
        # One kernel for: residual add + final RMSNorm (+1 scale, fp32
        # internal) + bf16 cast for the LM head.  ``summed`` reproduces
        # the upstream return value (the residual-dtype sum the proposer
        # recycles as the next step's previous_hidden_states);
        # ``normed`` is what Step5MTPSharedHead.forward would produce and
        # is stashed for compute_logits.
        normed, summed = fused_add_rms_norm_cast(
            ffn_output,
            residual,
            self.shared_head.norm.weight,
            self.shared_head.norm.variance_epsilon,
            zero_centered=getattr(
                self.shared_head.norm, "zero_centered", False
            ),
            fp32_residual=self.fp32_residual_connection,
        )
        self._fused_normed = normed
        self._fused_summed = summed
        return summed


class AscendStep5MultiTokenPredictor(Step5MultiTokenPredictor):
    """Upstream predictor whose layers use the Ascend draft blocks."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        config = _get_mtp_config(vllm_config)
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )
        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = config.num_nextn_predict_layers
        if self.num_mtp_layers <= 0:
            raise ValueError(
                "Step5 MTP requires num_nextn_predict_layers > 0, got "
                f"{self.num_mtp_layers}."
            )
        # to map the exact layer index from weights
        self.layers = torch.nn.ModuleDict(
            {
                str(idx): AscendStep5MultiTokenPredictorLayer(
                    vllm_config=vllm_config,
                    prefix=f"{prefix}.layers.{idx}",
                )
                for idx in range(
                    self.mtp_start_layer_idx,
                    self.mtp_start_layer_idx + self.num_mtp_layers,
                )
            }
        )

        self.logits_processor = LogitsProcessor(
            config.vocab_size,
            valid_vocab_size=_require_resolved_valid_vocab_size(
                _get_mtp_vllm_config(vllm_config).model_config
            ),
        )

    def _shared_head_hidden(
        self, mtp_layer: AscendStep5MultiTokenPredictorLayer,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Normed hidden for the LM head, without double-norming.

        When ``hidden_states`` is the summed tensor this layer's forward
        just returned, the shared-head norm has already been computed
        inside fused_add_rms_norm_cast; reuse it.  Anything else (dummy
        runs, fallback paths, other steps' tensors) goes through the
        upstream shared_head unchanged.
        """
        fused_summed = mtp_layer._fused_summed
        if (
            mtp_layer._fused_normed is not None
            and fused_summed is not None
            and hidden_states is fused_summed
        ):
            return mtp_layer._fused_normed
        return mtp_layer.shared_head(hidden_states)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        current_step_idx = spec_step_idx % self.num_mtp_layers
        mtp_layer = self.layers[str(self.mtp_start_layer_idx + current_step_idx)]
        logits = self.logits_processor(
            mtp_layer.shared_head.head,
            self._shared_head_hidden(mtp_layer, hidden_states),
        )
        return logits

    def get_top_tokens(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        current_step_idx = spec_step_idx % self.num_mtp_layers
        mtp_layer = self.layers[str(self.mtp_start_layer_idx + current_step_idx)]
        return self.logits_processor.get_top_tokens(
            mtp_layer.shared_head.head,
            self._shared_head_hidden(mtp_layer, hidden_states),
        )


class AscendStep5MTP(Step5MTP):
    """Step5MTP for Ascend: same wire format, Ascend draft decoder layer."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # Bypass Step5MTP.__init__ (raises on non-CUDA platforms); the body
        # below replicates it with the Ascend predictor.
        nn.Module.__init__(self)
        self.config = _get_mtp_config(vllm_config)
        self.vllm_config = vllm_config
        _require_resolved_valid_vocab_size(
            _get_mtp_vllm_config(vllm_config).model_config
        )
        self.model = AscendStep5MultiTokenPredictor(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        moe_blocks = _get_step5_mtp_moe_blocks(self.model)
        # Expose the actual draft topology so dense MTP forwards do not
        # inherit the target model's DP collectives.
        self.is_dense_mtp = not moe_blocks
        self.moe_layers = [moe.experts for moe in moe_blocks]
        _set_step5_moe_protocol_metadata(
            self,
            moe_blocks[0] if moe_blocks else None,
        )
