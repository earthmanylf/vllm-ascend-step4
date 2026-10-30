# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fuse remaining direct Step5RMSNorm calls, not the already-fused model loop.

Step5RMSNorm is an nn.Module without an Ascend CustomOp extension point.
Remove this forward-only patch when upstream exposes one. FP32 kernel inputs
preserve Step5's FP32 normalization/scale math, including zero-centered weights;
low-precision inputs therefore pay for casts. Residual addition is deliberately
not fused: CANN add-RMSNorm need not match torch promotion/rounding or ownership.
Hardware accuracy and speed must be measured before claiming an improvement.
"""

import torch
import torch_npu

try:
    from vllm.models.step5.nvidia.layernorm import Step5RMSNorm
except ModuleNotFoundError as exc:
    # Public vLLM builds may not include the private Step5 model.
    if exc.name not in {
        "vllm.models",
        "vllm.models.step5",
        "vllm.models.step5.nvidia",
        "vllm.models.step5.nvidia.layernorm",
    }:
        raise
    Step5RMSNorm = None


_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _step5_rms_norm_forward_ascend(
    self,
    x: torch.Tensor,
    residual: torch.Tensor | None = None,
    output: torch.Tensor | None = None,
    fp16_out: bool = False,
    *,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    if not getattr(x, "is_npu", False):
        return _ORIGINAL_FORWARD(self, x, residual, output, fp16_out, out_dtype=out_dtype)

    if residual is not None:
        if output is not None or fp16_out:
            raise ValueError("Residual Step5RMSNorm does not support output buffers or fp16_out.")
    elif fp16_out:
        raise ValueError("Step5RMSNorm does not support fp16_out.")

    result_dtype = out_dtype or x.dtype
    if output is not None and (output.shape != x.shape or output.dtype != result_dtype):
        raise ValueError("Step5 RMSNorm output buffer must match shape and dtype")

    if (
        x.dtype not in _SUPPORTED_DTYPES
        or self.weight.dtype not in _SUPPORTED_DTYPES
        or x.ndim == 0
        or self.weight.ndim != 1
        or x.shape[-1] != self.weight.shape[0]
        or (residual is not None and residual.dtype not in _SUPPORTED_DTYPES)
    ):
        return _ORIGINAL_FORWARD(self, x, residual, output, fp16_out, out_dtype=out_dtype)

    # Preserve torch's promotion, addition rounding and fresh residual allocation.
    residual_out = x + residual if residual is not None else x
    scale = self.weight.float()
    if self.zero_centered:
        scale = scale + 1.0
    result, _ = torch_npu.npu_rms_norm(residual_out.float(), scale, self.variance_epsilon)
    result = result.to(result_dtype)
    if residual is not None:
        return result, residual_out
    if output is not None:
        output.copy_(result)
        return output
    return result


if Step5RMSNorm is not None:
    _ORIGINAL_FORWARD = Step5RMSNorm.forward
    Step5RMSNorm.forward = _step5_rms_norm_forward_ascend
