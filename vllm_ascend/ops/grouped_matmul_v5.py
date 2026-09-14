# SPDX-License-Identifier: Apache-2.0
"""Block-fp8 grouped matmul through the CANN V5 G-B scenario.

torch_npu's ``npu_grouped_matmul`` wrapper routes fp8 inputs to the V4 entry
and rejects them; the V5 G-B scenario (e4m3 weights + FLOAT32 128x128 block
scales + FLOAT32 per-token-per-128-group activation scales) is reached via
the custom op ``_C_ascend.grouped_matmul_block_fp8`` (csrc/torch_binding.cpp,
built with COMPILE_CUSTOM_KERNELS=1).

Verified on Ascend 950DT / CANN 9.1.0: numerics match the vendored step4-hf
Triton ``fp8_gemm`` (same rel_l2 vs the fp32 dequant reference; identical
bf16 rounding), and the device kernel is 1.2-2.2x faster than the Triton one
at Step4 expert shapes (msprof task-time trace).
"""

import functools

import torch

# On Ascend950 the aclnn V5 entry expects K-major weights: [g, K, N] fp8 and
# [g, ceil(K/128), ceil(N/128)] fp32 block scales.

_op_loaded = False


@functools.lru_cache(maxsize=1)
def block_fp8_gmm_available() -> bool:
    """Whether the packaged custom-op lib registers the V5 block-fp8 GMM.

    Requires a build with COMPILE_CUSTOM_KERNELS=1; when the op is absent
    callers should fall back to the vendored step4-hf triton kernels.
    """
    try:
        _ensure_op_loaded()
        return hasattr(torch.ops._C_ascend, "grouped_matmul_block_fp8")
    except Exception:
        return False


def _ensure_op_loaded() -> None:
    """Import the custom-op extension once (it self-registers with torch)."""
    global _op_loaded
    if not _op_loaded:
        import vllm_ascend.vllm_ascend_C  # noqa: F401
        _op_loaded = True


def grouped_matmul_block_fp8(
    x: list[torch.Tensor],
    weight: list[torch.Tensor],
    scale: list[torch.Tensor],
    per_token_scale: list[torch.Tensor],
    group_list: torch.Tensor,
) -> list[torch.Tensor]:
    """Run one block-fp8 grouped matmul per (x, weight, scale) list entry.

    Returns one bf16 ``[M, N]`` output per entry. Requires the custom-op
    build (``COMPILE_CUSTOM_KERNELS=1``); the op allocates its own outputs.
    """
    _ensure_op_loaded()
    return torch.ops._C_ascend.grouped_matmul_block_fp8(
        x, weight, scale, per_token_scale, group_list, 2, 0, 0
    )
