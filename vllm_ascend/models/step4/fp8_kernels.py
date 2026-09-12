# SPDX-License-Identifier: Apache-2.0
"""Step4 FP8 block-GEMM kernels for Ascend, vendored from the step4-hf
multi-hardware release (``inference/kernel.py``, which is Apache-2.0 and
device-portable Triton). Only the operators the FP8 expert path needs are
copied, unchanged:

- ``act_quant``: dynamic per-128-group e4m3 activation quantization.
- ``fp8_gemm``: block-scaled fp8 matmul (fp8 x fp8 -> fp32 accumulator,
  scales folded after the dot).
- ``linear_fp8_or_bf16``: dispatcher -- fp8 path when a weight scale is
  present, plain bf16 ``F.linear`` otherwise (the not-converted layers).

These exact kernels were verified to compile and run on Ascend950DT with
triton-ascend 3.2.2 by the step4-hf reference run (2026-09-03). Tile sizes
are load-bearing for correctness: ``BLOCK_K`` must equal the quantization
group (128) and ``BLOCK_N`` must divide the weight N-block (128) so one
K-iteration consumes exactly one scale entry and one output tile sits inside
one weight N-block.
"""

import torch
import triton
import triton.language as tl

try:
    import torch_npu  # noqa: F401
except ImportError:
    pass

FP8_E4M3_MAX: float = 448.0
FP8_AMAX_FLOOR: float = 1e-10

# K-block size shared by the activation quantizer and the weight scale grid.
BLOCK_K: int = 128
# Weight N-block; the kernel requires BLOCK_N to divide this value.
BLOCK_N: int = 128


@triton.jit
def _act_quant_kernel(
    x_ptr,
    y_ptr,
    scale_ptr,
    M,
    K,
    stride_xm,
    stride_xk,
    stride_ym,
    stride_yk,
    stride_sm,
    stride_sk,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
) -> None:
    """Per-row 1-D block quantization to fp8_e4m3fn along K.

    One program handles a ``[BLOCK_M, BLOCK_K]`` tile -- exactly one
    quantization group wide. The amax is computed over the *live* elements
    of each row, so the kernel handles a final ragged K-block where the
    tile extends past K.
    """
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    row_mask = rows < M
    col_mask = cols < K
    tile_mask = row_mask[:, None] & col_mask[None, :]

    x = tl.load(
        x_ptr + rows[:, None] * stride_xm + cols[None, :] * stride_xk,
        mask=tile_mask,
        other=0.0,
    ).to(tl.float32)

    amax = tl.max(tl.abs(x), axis=1)
    amax = tl.maximum(amax, 1e-10)
    scale = amax * (1.0 / 448.0)
    inv_scale = 1.0 / scale

    # Clamp to [-448, 448] before the round-to-fp8: a value just above 448
    # would otherwise wrap to NaN instead of saturating.
    q = tl.clamp(x * inv_scale[:, None], -448.0, 448.0)

    tl.store(
        y_ptr + rows[:, None] * stride_ym + cols[None, :] * stride_yk,
        q.to(y_ptr.dtype.element_ty),
        mask=tile_mask,
    )
    tl.store(
        scale_ptr + rows * stride_sm + pid_k * stride_sk,
        scale,
        mask=row_mask,
    )


def act_quant(
    x: torch.Tensor, block_size: int = 128
) -> tuple[torch.Tensor, torch.Tensor]:
    """Block-wise FP8 quantization along the last dim of ``x``.

    Returns ``(y, scale)`` where ``y`` is ``torch.float8_e4m3fn`` with
    ``x``'s shape and ``scale`` is ``[..., ceil(K/block_size)]`` fp32, with
    ``real_x = y.float() * scale.unsqueeze(-1)``.
    """
    if block_size & (block_size - 1):
        raise ValueError(f"block_size must be a power of two, got {block_size}")
    leading = x.shape[:-1]
    K = x.shape[-1]
    x2 = x.contiguous()
    flat = x2.view(-1, K)
    M = flat.shape[0]
    y = torch.empty_like(flat, dtype=torch.float8_e4m3fn)
    scale = torch.empty(
        M, (K + block_size - 1) // block_size, device=x.device, dtype=torch.float32
    )
    BLOCK_M = 16 if M < 32 else 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(K, block_size))
    _act_quant_kernel[grid](
        flat,
        y,
        scale,
        M,
        K,
        flat.stride(0),
        flat.stride(1),
        y.stride(0),
        y.stride(1),
        scale.stride(0),
        scale.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_K=block_size,
    )
    return y.view(*leading, K), scale.view(*leading, -1)


@triton.jit
def _fp8_gemm_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    A_scale_ptr,
    W_scale_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    stride_asm,
    stride_ask,
    stride_wsn,
    stride_wsk,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_N: tl.constexpr,
) -> None:
    """fp8 GEMM with per-128 block scaling on both operands.

    One program computes a ``[BLOCK_M, BLOCK_N]`` output tile. ``BLOCK_N``
    must divide GROUP_N (128) so the tile sits inside exactly one weight
    N-block, and ``BLOCK_K`` must equal the activation block size (128) so
    one K-iter consumes exactly one act scale entry and one weight scale
    entry per row. Loads B transposed (as ``[BLOCK_K, BLOCK_N]``) so
    ``tl.dot`` sees ``A @ B`` in the natural orientation.

    The two scales are folded into the fp32 accumulator after the fp8 dot,
    not into the fp8 operands; the post-dot multiply is the whole reason
    the accumulator is fp32.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    m_mask = offs_m < M
    n_mask = offs_n < N

    a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B_ptr + offs_n[None, :] * stride_bn + offs_k[:, None] * stride_bk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    n_block = pid_n * BLOCK_N // GROUP_N

    k_iters = (K + BLOCK_K - 1) // BLOCK_K
    for k_idx in tl.range(0, k_iters):
        k_offset = k_idx * BLOCK_K
        k_mask = (k_offset + offs_k) < K

        a = tl.load(
            a_ptrs + k_offset * stride_ak,
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        b = tl.load(
            b_ptrs + k_offset * stride_bk,
            mask=n_mask[None, :] & k_mask[:, None],
            other=0.0,
        )
        partial = tl.dot(a, b, out_dtype=tl.float32)

        a_s = tl.load(
            A_scale_ptr + offs_m * stride_asm + k_idx * stride_ask,
            mask=m_mask,
            other=0.0,
        )
        w_s = tl.load(W_scale_ptr + n_block * stride_wsn + k_idx * stride_wsk)

        acc += partial * a_s[:, None] * w_s

    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(
        c_ptrs,
        acc.to(C_ptr.dtype.element_ty),
        mask=m_mask[:, None] & n_mask[None, :],
    )


def fp8_gemm(
    x_fp8: torch.Tensor,
    x_scale: torch.Tensor,
    w_fp8: torch.Tensor,
    w_scale_inv: torch.Tensor,
) -> torch.Tensor:
    """Block-scaled fp8 matmul ``y = x @ w.T`` returning bf16 ``[M, N]``.

    ``w_scale_inv`` follows the inverse convention
    ``real_weight = w_fp8 * w_scale_inv[block]``.
    """
    if x_fp8.dtype != torch.float8_e4m3fn or w_fp8.dtype != torch.float8_e4m3fn:
        raise TypeError("operands must be float8_e4m3fn")
    if x_scale.dtype != torch.float32 or w_scale_inv.dtype != torch.float32:
        raise TypeError("scales must be float32")
    if x_fp8.dim() != 2 or w_fp8.dim() != 2:
        raise ValueError("operands must be 2-D; group by expert at the call site")

    M, K = x_fp8.shape
    N, K_w = w_fp8.shape
    if K != K_w:
        raise ValueError(f"K mismatch: x has {K}, weight has {K_w}")
    k_blocks = (K + BLOCK_K - 1) // BLOCK_K
    if x_scale.shape != (M, k_blocks):
        raise ValueError(f"x_scale shape {tuple(x_scale.shape)} != ({M}, {k_blocks})")
    if w_scale_inv.shape != ((N + BLOCK_N - 1) // BLOCK_N, k_blocks):
        raise ValueError(
            f"w_scale_inv shape {tuple(w_scale_inv.shape)} != "
            f"({(N + BLOCK_N - 1) // BLOCK_N}, {k_blocks})"
        )

    x_fp8 = x_fp8.contiguous()
    w_fp8 = w_fp8.contiguous()
    x_scale = x_scale.contiguous()
    w_scale_inv = w_scale_inv.contiguous()

    y = torch.empty((M, N), device=x_fp8.device, dtype=torch.bfloat16)
    BLOCK_M = 64 if M >= 64 else 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _fp8_gemm_kernel[grid](
        x_fp8,
        w_fp8,
        y,
        x_scale,
        w_scale_inv,
        M,
        N,
        K,
        x_fp8.stride(0),
        x_fp8.stride(1),
        w_fp8.stride(0),
        w_fp8.stride(1),
        y.stride(0),
        y.stride(1),
        x_scale.stride(0),
        x_scale.stride(1),
        w_scale_inv.stride(0),
        w_scale_inv.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_N=BLOCK_N,
    )
    return y


def linear_fp8_or_bf16(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale_inv: torch.Tensor | None,
) -> torch.Tensor:
    """``y = x @ weight.T`` with the fp8 path when a scale is present.

    The criterion is *presence of the scale tensor*, never the layer index:
    the not-converted layers ship bf16 experts with no scale.
    """
    if weight_scale_inv is None:
        if weight.dtype != torch.bfloat16:
            weight = weight.to(torch.bfloat16)
        x_bf = x.to(torch.bfloat16) if x.dtype != torch.bfloat16 else x
        return torch.nn.functional.linear(x_bf, weight)

    leading = x.shape[:-1]
    K = x.shape[-1]
    x_bf = x.to(torch.bfloat16) if x.dtype != torch.bfloat16 else x
    x_flat = x_bf.reshape(-1, K).contiguous()
    x_fp8, x_s = act_quant(x_flat)
    y = fp8_gemm(x_fp8, x_s, weight, weight_scale_inv)
    return y.view(*leading, -1)
