# SPDX-License-Identifier: Apache-2.0
"""Fused QKNorm+RoPE Triton kernel for Ascend NPU.

Replaces the eager 34-op decomposition (9 ops RMSNorm × 2 + 8 ops RoPE × 2)
with a single SIMD kernel launch plus a V passthrough copy.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

try:
    from triton.language.extra.cann import libdevice as _libdevice
except ImportError:
    try:
        from triton.language.extra import libdevice as _libdevice
    except ImportError:
        _libdevice = tl.libdevice


COMPILE_MODE: str | None = "simd"
BLOCK_M: int = 16

HEAD_DIM: int = 192
EPS: float = 1e-5
NORM_WEIGHT_BIAS: float = 1.0


@triton.jit
def _qknorm_rope_192_kernel(
    qkv_ptr,
    q_out_ptr,
    k_out_ptr,
    q_weight_ptr,
    k_weight_ptr,
    inv_freq_ptr,
    positions_ptr,
    tokens,
    stride_qkv_token,
    stride_q_out_token,
    stride_k_out_token,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    ROTARY_PAIRS: tl.constexpr,
    EPS: tl.constexpr,
    NORM_WEIGHT_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    pid_m = tl.program_id(0)
    packed_head = tl.program_id(1)
    is_q = packed_head < NUM_Q_HEADS

    offs_t = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    t_mask = offs_t < tokens

    dims = tl.arange(0, BLOCK_D)
    dim_mask = dims < HEAD_DIM
    pairs = tl.arange(0, BLOCK_R)
    pair_mask = pairs < ROTARY_PAIRS

    head_base = offs_t[:, None] * stride_qkv_token + packed_head * HEAD_DIM
    values = tl.load(
        qkv_ptr + head_base + dims[None, :],
        mask=t_mask[:, None] & dim_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    sum_squares = tl.sum(values * values, axis=1)
    mean_square = sum_squares / HEAD_DIM
    inverse_rms = tl.rsqrt(mean_square + EPS)

    q_weight = tl.load(q_weight_ptr + dims, mask=dim_mask, other=0.0).to(tl.float32)
    k_weight = tl.load(k_weight_ptr + dims, mask=dim_mask, other=0.0).to(tl.float32)
    weight = tl.where(is_q, q_weight, k_weight) + NORM_WEIGHT_BIAS
    normalized = (values * inverse_rms[:, None] * weight[None, :]).to(tl.bfloat16)
    normalized_f32 = normalized.to(tl.float32)

    idx0 = tl.broadcast_to(pairs[None, :], (BLOCK_M, BLOCK_R))
    idx1 = idx0 + ROTARY_PAIRS
    value0 = tl.gather(normalized_f32, idx0, axis=1)
    value1 = tl.gather(normalized_f32, idx1, axis=1)

    position = tl.load(positions_ptr + offs_t, mask=t_mask, other=0).to(tl.float32)
    inv_freq = tl.load(
        inv_freq_ptr + pairs,
        mask=pair_mask,
        other=0.0,
    ).to(tl.float32)
    angle = position[:, None] * inv_freq[None, :]
    cos_value = _libdevice.cos(angle)
    sin_value = _libdevice.sin(angle)

    rotated0 = (value0 * cos_value - value1 * sin_value).to(tl.bfloat16)
    rotated1 = tl.fma(value0, sin_value, value1 * cos_value).to(tl.bfloat16)

    q_base = offs_t[:, None] * stride_q_out_token + packed_head * HEAD_DIM
    k_base = (
        offs_t[:, None] * stride_k_out_token
        + (packed_head - NUM_Q_HEADS) * HEAD_DIM
    )
    store_mask = t_mask[:, None] & pair_mask[None, :]
    tl.store(q_out_ptr + q_base + idx0, rotated0, mask=store_mask & is_q)
    tl.store(q_out_ptr + q_base + idx1, rotated1, mask=store_mask & is_q)
    tl.store(k_out_ptr + k_base + idx0, rotated0, mask=store_mask & ~is_q)
    tl.store(k_out_ptr + k_base + idx1, rotated1, mask=store_mask & ~is_q)


@triton.jit
def _copy_v_192_kernel(
    qkv_ptr,
    v_out_ptr,
    stride_qkv_token,
    stride_v_out_token,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    dims = tl.arange(0, BLOCK_D)
    mask = dims < HEAD_DIM
    input_base = (
        token * stride_qkv_token + (NUM_Q_HEADS + NUM_KV_HEADS + head) * HEAD_DIM
    )
    output_base = token * stride_v_out_token + head * HEAD_DIM
    value = tl.load(qkv_ptr + input_base + dims, mask=mask)
    tl.store(v_out_ptr + output_base + dims, value, mask=mask)


_inv_freq_cache: dict[tuple[str, int], torch.Tensor] = {}


def _build_inv_freq(device, rotary_pairs: int) -> torch.Tensor:
    key = (str(device), rotary_pairs)
    cached = _inv_freq_cache.get(key)
    if cached is None:
        span = 2 * rotary_pairs
        theta = 10000.0
        idx = torch.arange(0, span, 2, device=device, dtype=torch.float32)
        cached = (1.0 / (theta ** (idx / span))).contiguous()
        _inv_freq_cache[key] = cached
    return cached


_out_buffers: dict[tuple, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def _get_outputs(
    tokens: int,
    num_q_heads: int,
    num_kv_heads: int,
    device,
    dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    key = (str(device), tokens, num_q_heads, num_kv_heads, str(dtype))
    cached = _out_buffers.get(key)
    if cached is None:
        q_out = torch.empty(
            (tokens, num_q_heads * HEAD_DIM), device=device, dtype=dtype
        )
        k_out = torch.empty(
            (tokens, num_kv_heads * HEAD_DIM), device=device, dtype=dtype
        )
        v_out = torch.empty(
            (tokens, num_kv_heads * HEAD_DIM), device=device, dtype=dtype
        )
        cached = (q_out, k_out, v_out)
        _out_buffers[key] = cached
    return cached


def _fused_qknorm_rope_impl(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    *,
    head_dim: int,
    num_q_heads: int,
    num_kv_heads: int,
    rotary_pairs: int,
    eps: float = EPS,
    norm_weight_bias: float = NORM_WEIGHT_BIAS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if head_dim != HEAD_DIM:
        raise ValueError(f"head_dim must be {HEAD_DIM}, got {head_dim}")
    inv_freq = _build_inv_freq(qkv.device, rotary_pairs)

    tokens = qkv.shape[0]
    q_out, k_out, v_out = _get_outputs(
        tokens, num_q_heads, num_kv_heads, qkv.device, qkv.dtype
    )
    if tokens == 0:
        return q_out, k_out, v_out

    launch_opts: dict = {"num_warps": 4}
    if COMPILE_MODE is not None:
        launch_opts["compile_mode"] = COMPILE_MODE

    _qknorm_rope_192_kernel[
        (triton.cdiv(tokens, BLOCK_M), num_q_heads + num_kv_heads)
    ](
        qkv,
        q_out,
        k_out,
        q_weight,
        k_weight,
        inv_freq,
        positions,
        tokens,
        qkv.stride(0),
        q_out.stride(0),
        k_out.stride(0),
        NUM_Q_HEADS=num_q_heads,
        NUM_KV_HEADS=num_kv_heads,
        ROTARY_PAIRS=rotary_pairs,
        EPS=float(eps),
        NORM_WEIGHT_BIAS=float(norm_weight_bias),
        BLOCK_M=BLOCK_M,
        BLOCK_D=256,
        BLOCK_R=128,
        **launch_opts,
    )
    _copy_v_192_kernel[(tokens, num_kv_heads)](
        qkv,
        v_out,
        qkv.stride(0),
        v_out.stride(0),
        NUM_Q_HEADS=num_q_heads,
        NUM_KV_HEADS=num_kv_heads,
        BLOCK_D=256,
        num_warps=4,
    )
    return q_out, k_out, v_out


def _fused_qknorm_rope_fake(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    *,
    head_dim: int,
    num_q_heads: int,
    num_kv_heads: int,
    rotary_pairs: int,
    eps: float = EPS,
    norm_weight_bias: float = NORM_WEIGHT_BIAS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tokens = qkv.shape[0]
    q = torch.empty(
        (tokens, num_q_heads * head_dim), device=qkv.device, dtype=qkv.dtype
    )
    k = torch.empty(
        (tokens, num_kv_heads * head_dim), device=qkv.device, dtype=qkv.dtype
    )
    v = torch.empty(
        (tokens, num_kv_heads * head_dim), device=qkv.device, dtype=qkv.dtype
    )
    return q, k, v


from vllm.utils.torch_utils import direct_register_custom_op

direct_register_custom_op(
    op_name="ascend_step4_fused_qknorm_rope",
    op_func=_fused_qknorm_rope_impl,
    mutates_args=[],
    fake_impl=_fused_qknorm_rope_fake,
)


def fused_qknorm_rope(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    *,
    head_dim: int,
    num_q_heads: int,
    num_kv_heads: int,
    rotary_pairs: int,
    eps: float = EPS,
    norm_weight_bias: float = NORM_WEIGHT_BIAS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.ops.vllm.ascend_step4_fused_qknorm_rope(
        qkv,
        q_weight,
        k_weight,
        cos,
        sin,
        positions,
        head_dim=head_dim,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        rotary_pairs=rotary_pairs,
        eps=eps,
        norm_weight_bias=norm_weight_bias,
    )
