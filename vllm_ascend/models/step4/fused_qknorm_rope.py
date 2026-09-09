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


# ============================================================================
# Fused indexer norm+RoPE for Q/K/Z (DSA sparse indexer)
# ============================================================================

import triton.language.extra.cann.extension as extension


def _npu_vec_core_num() -> int:
    import triton.runtime.driver as driver
    properties = driver.active.utils.get_device_properties(
        torch.npu.current_device()
    )
    return properties["num_vectorcore"]


def _pick_head_chunk(num_heads: int, head_dim: int) -> int:
    budget_cols = 4096
    width = num_heads * head_dim
    if width <= budget_cols:
        return num_heads
    h = max(1, min(num_heads, budget_cols // head_dim))
    return h


@triton.jit
def _norm_rope_tile_npu_chunk(
    x_ptr,
    out_ptr,
    cos_t,
    sin_t,
    token_start,
    stride_x_token,
    stride_out_token,
    seg_base,
    head_off,
    w_hd,
    b_hd,
    w_real,
    w_imag,
    b_real,
    b_imag,
    eps,
    weight_bias,
    tokens,
    CH: tl.constexpr,
    TOTAL_HEADS: tl.constexpr,
    head_dim: tl.constexpr,
    rotary_dim: tl.constexpr,
    SUBTRACT_MEAN: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_R: tl.constexpr,
    ROWS_PER_PROGRAM: tl.constexpr,
) -> None:
    half = tl.arange(0, BLOCK_R)
    half_ok = half < rotary_dim
    gr = tl.arange(0, ROWS_PER_PROGRAM * CH)
    tok = gr // CH
    head = gr % CH
    tok_abs = token_start + tok
    row_ok = tok_abs < tokens
    out_base = tok_abs * stride_out_token + head_off + head * head_dim

    x_bp = tl.make_block_ptr(
        base=x_ptr + seg_base,
        shape=(tokens, TOTAL_HEADS * head_dim),
        strides=(stride_x_token, 1),
        offsets=(token_start, head_off),
        block_shape=(ROWS_PER_PROGRAM, CH * head_dim),
        order=(1, 0),
    )
    values = tl.reshape(
        tl.load(x_bp, boundary_check=(0,), padding_option="zero").to(tl.float32),
        (ROWS_PER_PROGRAM * CH, head_dim),
    )

    cos = tl.reshape(
        tl.broadcast_to(cos_t[:, None, :], (ROWS_PER_PROGRAM, CH, BLOCK_R)),
        (ROWS_PER_PROGRAM * CH, BLOCK_R),
    )
    sin = tl.reshape(
        tl.broadcast_to(sin_t[:, None, :], (ROWS_PER_PROGRAM, CH, BLOCK_R)),
        (ROWS_PER_PROGRAM * CH, BLOCK_R),
    )

    sum_sq = tl.sum(values * values, axis=1)
    if SUBTRACT_MEAN:
        mean = tl.sum(values, axis=1) / head_dim
        variance = sum_sq / head_dim - mean * mean
    else:
        mean = tl.zeros((ROWS_PER_PROGRAM * CH,), dtype=tl.float32)
        variance = sum_sq / head_dim
    rstd = tl.rsqrt(variance + eps)
    result_tail = (values - mean[:, None]) * rstd[:, None] * (w_hd + weight_bias)
    if HAS_BIAS:
        result_tail = result_tail + b_hd

    if CH == 1:
        xrow = tok_abs * stride_x_token + head_off + head * head_dim + seg_base
        real_raw = tl.load(
            x_ptr + xrow[:, None] + half[None, :],
            mask=row_ok[:, None] & half_ok[None, :], other=0.0).to(tl.float32)
        imag_raw = tl.load(
            x_ptr + xrow[:, None] + rotary_dim + half[None, :],
            mask=row_ok[:, None] & half_ok[None, :], other=0.0).to(tl.float32)
    else:
        real_raw = extension.extract_slice(values, [0, 0], [ROWS_PER_PROGRAM * CH, BLOCK_R], [1, 1])
        imag_raw = extension.extract_slice(values, [0, rotary_dim], [ROWS_PER_PROGRAM * CH, BLOCK_R], [1, 1])
    real = (real_raw - mean[:, None]) * rstd[:, None] * (w_real + weight_bias)
    imaginary = (imag_raw - mean[:, None]) * rstd[:, None] * (w_imag + weight_bias)
    if HAS_BIAS:
        real = real + b_real
        imaginary = imaginary + b_imag
    real = real.to(out_ptr.dtype.element_ty).to(tl.float32)
    imaginary = imaginary.to(out_ptr.dtype.element_ty).to(tl.float32)
    rotated_first = real * cos - imaginary * sin
    rotated_second = real * sin + imaginary * cos

    out_bp = tl.make_block_ptr(
        base=out_ptr,
        shape=(tokens, TOTAL_HEADS * head_dim),
        strides=(stride_out_token, 1),
        offsets=(token_start, head_off),
        block_shape=(ROWS_PER_PROGRAM, CH * head_dim),
        order=(1, 0),
    )
    tl.store(
        out_bp,
        tl.reshape(result_tail.to(out_ptr.dtype.element_ty), (ROWS_PER_PROGRAM, CH * head_dim)),
        boundary_check=(0,),
    )
    mask_half = row_ok[:, None] & half_ok[None, :]
    tl.store(
        out_ptr + out_base[:, None] + half[None, :],
        rotated_first.to(out_ptr.dtype.element_ty),
        mask=mask_half,
    )
    tl.store(
        out_ptr + out_base[:, None] + rotary_dim + half[None, :],
        rotated_second.to(out_ptr.dtype.element_ty),
        mask=mask_half,
    )


@triton.jit
def _norm_rope_tile_npu(
    x_ptr,
    out_ptr,
    cos_ptr,
    sin_ptr,
    positions_ptr,
    token_idx,
    token_start,
    token_ok,
    stride_x_token,
    stride_out_token,
    stride_cos,
    seg_base,
    w_all,
    w_real,
    w_imag,
    b_all,
    b_real,
    b_imag,
    eps,
    weight_bias,
    tokens,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    rotary_dim: tl.constexpr,
    SUBTRACT_MEAN: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_R: tl.constexpr,
    ROWS_PER_PROGRAM: tl.constexpr,
    H_CHUNK: tl.constexpr,
    N_FULL: tl.constexpr,
    REM_HEADS: tl.constexpr,
) -> None:
    half = tl.arange(0, BLOCK_R)
    half_ok = half < rotary_dim
    w_hd = extension.extract_slice(tl.reshape(w_all, (1, BLOCK_D)), [0, 0], [1, head_dim], [1, 1])
    if HAS_BIAS:
        b_hd = extension.extract_slice(tl.reshape(b_all, (1, BLOCK_D)), [0, 0], [1, head_dim], [1, 1])
    else:
        b_hd = tl.zeros((1, head_dim), dtype=tl.float32)

    position = tl.load(positions_ptr + token_idx, mask=token_ok, other=0).to(tl.int32)
    cos_t = tl.load(
        cos_ptr + position[:, None] * stride_cos + half[None, :],
        mask=token_ok[:, None] & half_ok[None, :],
        other=0.0,
    ).to(tl.float32)
    extension.compile_hint(cos_t, "mayDiscretememaccess")
    sin_t = tl.load(
        sin_ptr + position[:, None] * stride_cos + half[None, :],
        mask=token_ok[:, None] & half_ok[None, :],
        other=0.0,
    ).to(tl.float32)
    extension.compile_hint(sin_t, "mayDiscretememaccess")

    for c in tl.static_range(N_FULL):
        _norm_rope_tile_npu_chunk(
            x_ptr, out_ptr, cos_t, sin_t, token_start,
            stride_x_token, stride_out_token, seg_base, c * H_CHUNK * head_dim,
            w_hd, b_hd, w_real, w_imag, b_real, b_imag, eps, weight_bias, tokens,
            CH=H_CHUNK, TOTAL_HEADS=num_heads, head_dim=head_dim, rotary_dim=rotary_dim,
            SUBTRACT_MEAN=SUBTRACT_MEAN, HAS_BIAS=HAS_BIAS,
            BLOCK_R=BLOCK_R, ROWS_PER_PROGRAM=ROWS_PER_PROGRAM,
        )
    if REM_HEADS > 0:
        _norm_rope_tile_npu_chunk(
            x_ptr, out_ptr, cos_t, sin_t, token_start,
            stride_x_token, stride_out_token, seg_base, N_FULL * H_CHUNK * head_dim,
            w_hd, b_hd, w_real, w_imag, b_real, b_imag, eps, weight_bias, tokens,
            CH=REM_HEADS, TOTAL_HEADS=num_heads, head_dim=head_dim, rotary_dim=rotary_dim,
            SUBTRACT_MEAN=SUBTRACT_MEAN, HAS_BIAS=HAS_BIAS,
            BLOCK_R=BLOCK_R, ROWS_PER_PROGRAM=ROWS_PER_PROGRAM,
        )


@triton.autotune(
    configs=[
        triton.Config({"ROWS_PER_PROGRAM_Q": q, "ROWS_PER_PROGRAM_KZ": kv})
        for q, kv in ((1, 2), (1, 4), (1, 8), (1, 16), (2, 2), (2, 4), (2, 8), (2, 16), (2, 32), (4, 16), (4, 32))
    ],
    key=["M_q", "M_k", "head_dim", "rotary_dim"],
)
@triton.jit
def _indexer_qkz_kv_norm_rope_kernel(
    x_cat_ptr,
    out_q_ptr,
    out_k_ptr,
    out_z_ptr,
    q_weight_ptr,
    q_bias_ptr,
    k_weight_ptr,
    k_bias_ptr,
    cos_ptr,
    sin_ptr,
    positions_ptr,
    M_q,
    M_k,
    stride_x_token,
    stride_out_q,
    stride_out_k,
    stride_cos,
    q_seg_len,
    z_seg_len,
    eps,
    weight_bias_q,
    weight_bias_k,
    num_q_heads: tl.constexpr,
    num_k_heads: tl.constexpr,
    head_dim: tl.constexpr,
    rotary_dim: tl.constexpr,
    SUBTRACT_MEAN_Q: tl.constexpr,
    HAS_BIAS_Q: tl.constexpr,
    SUBTRACT_MEAN_K: tl.constexpr,
    HAS_BIAS_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_R: tl.constexpr,
    ROWS_PER_PROGRAM_Q: tl.constexpr,
    ROWS_PER_PROGRAM_KZ: tl.constexpr,
    H_CHUNK_Q: tl.constexpr,
    H_CHUNK_K: tl.constexpr,
    N_FULL_Q: tl.constexpr,
    REM_Q: tl.constexpr,
    N_FULL_K: tl.constexpr,
    REM_K: tl.constexpr,
) -> None:
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)

    dims = tl.arange(0, BLOCK_D)
    dim_ok = dims < head_dim
    half = tl.arange(0, BLOCK_R)
    half_ok = half < rotary_dim

    w_all_q = tl.load(q_weight_ptr + dims, mask=dim_ok, other=0.0).to(tl.float32)
    w_real_q = tl.load(q_weight_ptr + half, mask=half_ok, other=0.0).to(tl.float32)
    w_imag_q = tl.load(q_weight_ptr + rotary_dim + half, mask=half_ok, other=0.0).to(tl.float32)
    if HAS_BIAS_Q:
        b_all_q = tl.load(q_bias_ptr + dims, mask=dim_ok, other=0.0).to(tl.float32)
        b_real_q = tl.load(q_bias_ptr + half, mask=half_ok, other=0.0).to(tl.float32)
        b_imag_q = tl.load(q_bias_ptr + rotary_dim + half, mask=half_ok, other=0.0).to(tl.float32)
    else:
        b_all_q = tl.zeros((BLOCK_D,), dtype=tl.float32)
        b_real_q = tl.zeros((BLOCK_R,), dtype=tl.float32)
        b_imag_q = tl.zeros((BLOCK_R,), dtype=tl.float32)
    num_q_tokens = M_q // num_q_heads
    num_q_tiles = (num_q_tokens + ROWS_PER_PROGRAM_Q - 1) // ROWS_PER_PROGRAM_Q
    for t in range(pid, num_q_tiles, nprog):
        token_start = t * ROWS_PER_PROGRAM_Q
        token_idx = token_start + tl.arange(0, ROWS_PER_PROGRAM_Q)
        token_ok = token_idx < num_q_tokens
        _norm_rope_tile_npu(
            x_cat_ptr, out_q_ptr, cos_ptr, sin_ptr, positions_ptr,
            token_idx, token_start, token_ok,
            stride_x_token, stride_out_q, stride_cos, 0,
            w_all_q, w_real_q, w_imag_q, b_all_q, b_real_q, b_imag_q, eps, weight_bias_q,
            num_q_tokens,
            num_heads=num_q_heads, head_dim=head_dim, rotary_dim=rotary_dim,
            SUBTRACT_MEAN=SUBTRACT_MEAN_Q, HAS_BIAS=HAS_BIAS_Q,
            BLOCK_D=BLOCK_D, BLOCK_R=BLOCK_R, ROWS_PER_PROGRAM=ROWS_PER_PROGRAM_Q,
            H_CHUNK=H_CHUNK_Q, N_FULL=N_FULL_Q, REM_HEADS=REM_Q,
        )

    w_all_k = tl.load(k_weight_ptr + dims, mask=dim_ok, other=0.0).to(tl.float32)
    w_real_k = tl.load(k_weight_ptr + half, mask=half_ok, other=0.0).to(tl.float32)
    w_imag_k = tl.load(k_weight_ptr + rotary_dim + half, mask=half_ok, other=0.0).to(tl.float32)
    if HAS_BIAS_K:
        b_all_k = tl.load(k_bias_ptr + dims, mask=dim_ok, other=0.0).to(tl.float32)
        b_real_k = tl.load(k_bias_ptr + half, mask=half_ok, other=0.0).to(tl.float32)
        b_imag_k = tl.load(k_bias_ptr + rotary_dim + half, mask=half_ok, other=0.0).to(tl.float32)
    else:
        b_all_k = tl.zeros((BLOCK_D,), dtype=tl.float32)
        b_real_k = tl.zeros((BLOCK_R,), dtype=tl.float32)
        b_imag_k = tl.zeros((BLOCK_R,), dtype=tl.float32)
    num_k_tokens = M_k // num_k_heads
    num_k_tiles = (num_k_tokens + ROWS_PER_PROGRAM_KZ - 1) // ROWS_PER_PROGRAM_KZ
    for t in range(pid, num_k_tiles, nprog):
        token_start = t * ROWS_PER_PROGRAM_KZ
        token_idx = token_start + tl.arange(0, ROWS_PER_PROGRAM_KZ)
        token_ok = token_idx < num_k_tokens
        _norm_rope_tile_npu(
            x_cat_ptr, out_k_ptr, cos_ptr, sin_ptr, positions_ptr,
            token_idx, token_start, token_ok,
            stride_x_token, stride_out_k, stride_cos, q_seg_len,
            w_all_k, w_real_k, w_imag_k, b_all_k, b_real_k, b_imag_k, eps, weight_bias_k,
            num_k_tokens,
            num_heads=num_k_heads, head_dim=head_dim, rotary_dim=rotary_dim,
            SUBTRACT_MEAN=SUBTRACT_MEAN_K, HAS_BIAS=HAS_BIAS_K,
            BLOCK_D=BLOCK_D, BLOCK_R=BLOCK_R, ROWS_PER_PROGRAM=ROWS_PER_PROGRAM_KZ,
            H_CHUNK=H_CHUNK_K, N_FULL=N_FULL_K, REM_HEADS=REM_K,
        )
        z_bp = tl.make_block_ptr(
            base=x_cat_ptr + z_seg_len,
            shape=(num_k_tokens, num_k_heads * head_dim),
            strides=(stride_x_token, 1),
            offsets=(token_start, 0),
            block_shape=(ROWS_PER_PROGRAM_KZ, num_k_heads * head_dim),
            order=(1, 0),
        )
        z_vals = tl.load(z_bp, boundary_check=(0,), padding_option="zero")
        z_out_bp = tl.make_block_ptr(
            base=out_z_ptr,
            shape=(num_k_tokens, num_k_heads * head_dim),
            strides=(num_k_heads * head_dim, 1),
            offsets=(token_start, 0),
            block_shape=(ROWS_PER_PROGRAM_KZ, num_k_heads * head_dim),
            order=(1, 0),
        )
        tl.store(z_out_bp, z_vals.to(out_z_ptr.dtype.element_ty), boundary_check=(0,))


def _fused_indexer_qkz_kv_norm_rope_impl(
    x_cat: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    q_norm_bias: torch.Tensor | None = None,
    k_norm_bias: torch.Tensor | None = None,
    *,
    head_dim: int,
    num_q_heads: int,
    num_k_heads: int,
    rotary_dim: int,
    eps: float = 1e-6,
    q_norm_weight_bias: float = 1.0,
    k_norm_weight_bias: float = 0.0,
    subtract_mean_q: bool = False,
    subtract_mean_k: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tokens = x_cat.shape[0]
    out_q = torch.empty((tokens, num_q_heads * head_dim), device=x_cat.device, dtype=x_cat.dtype)
    out_k = torch.empty((tokens, num_k_heads * head_dim), device=x_cat.device, dtype=x_cat.dtype)
    out_z = torch.empty((tokens, num_k_heads * head_dim), device=x_cat.device, dtype=x_cat.dtype)

    m_q = tokens * num_q_heads
    m_k = tokens * num_k_heads
    q_seg_len = num_q_heads * head_dim
    z_seg_len = (num_q_heads + num_k_heads) * head_dim
    token_stride = (num_q_heads + 2 * num_k_heads) * head_dim

    grid = lambda meta: (
        min(
            max(
                (tokens + meta["ROWS_PER_PROGRAM_Q"] - 1) // meta["ROWS_PER_PROGRAM_Q"],
                (tokens + meta["ROWS_PER_PROGRAM_KZ"] - 1) // meta["ROWS_PER_PROGRAM_KZ"],
            ),
            _npu_vec_core_num(),
        ),
    )
    _indexer_qkz_kv_norm_rope_kernel[grid](
        x_cat,
        out_q,
        out_k,
        out_z,
        q_norm_weight,
        q_norm_bias if q_norm_bias is not None else q_norm_weight,
        k_norm_weight,
        k_norm_bias if k_norm_bias is not None else k_norm_weight,
        cos,
        sin,
        positions,
        m_q,
        m_k,
        token_stride,
        out_q.stride(0),
        out_k.stride(0),
        cos.stride(0),
        q_seg_len,
        z_seg_len,
        eps,
        q_norm_weight_bias,
        k_norm_weight_bias,
        num_q_heads=num_q_heads,
        num_k_heads=num_k_heads,
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        SUBTRACT_MEAN_Q=subtract_mean_q,
        HAS_BIAS_Q=q_norm_bias is not None,
        SUBTRACT_MEAN_K=subtract_mean_k,
        HAS_BIAS_K=k_norm_bias is not None,
        BLOCK_D=triton.next_power_of_2(head_dim),
        BLOCK_R=triton.next_power_of_2(rotary_dim),
        H_CHUNK_Q=(hc_q := _pick_head_chunk(num_q_heads, head_dim)),
        H_CHUNK_K=(hc_k := _pick_head_chunk(num_k_heads, head_dim)),
        N_FULL_Q=num_q_heads // hc_q,
        REM_Q=num_q_heads % hc_q,
        N_FULL_K=num_k_heads // hc_k,
        REM_K=num_k_heads % hc_k,
    )
    return out_q, out_k, out_z


def _fused_indexer_qkz_kv_norm_rope_fake(
    x_cat: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    q_norm_bias: torch.Tensor | None = None,
    k_norm_bias: torch.Tensor | None = None,
    *,
    head_dim: int,
    num_q_heads: int,
    num_k_heads: int,
    rotary_dim: int,
    eps: float = 1e-6,
    q_norm_weight_bias: float = 1.0,
    k_norm_weight_bias: float = 0.0,
    subtract_mean_q: bool = False,
    subtract_mean_k: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tokens = x_cat.shape[0]
    q = torch.empty((tokens, num_q_heads * head_dim), device=x_cat.device, dtype=x_cat.dtype)
    k = torch.empty((tokens, num_k_heads * head_dim), device=x_cat.device, dtype=x_cat.dtype)
    z = torch.empty((tokens, num_k_heads * head_dim), device=x_cat.device, dtype=x_cat.dtype)
    return q, k, z


direct_register_custom_op(
    op_name="ascend_step4_fused_indexer_norm_rope",
    op_func=_fused_indexer_qkz_kv_norm_rope_impl,
    mutates_args=[],
    fake_impl=_fused_indexer_qkz_kv_norm_rope_fake,
)


def fused_indexer_qkz_kv_norm_rope(
    x_cat: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    q_norm_bias: torch.Tensor | None = None,
    k_norm_bias: torch.Tensor | None = None,
    *,
    head_dim: int,
    num_q_heads: int,
    num_k_heads: int,
    rotary_dim: int,
    eps: float = 1e-6,
    q_norm_weight_bias: float = 1.0,
    k_norm_weight_bias: float = 0.0,
    subtract_mean_q: bool = False,
    subtract_mean_k: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.ops.vllm.ascend_step4_fused_indexer_norm_rope(
        x_cat,
        q_norm_weight,
        k_norm_weight,
        cos,
        sin,
        positions,
        q_norm_bias,
        k_norm_bias,
        head_dim=head_dim,
        num_q_heads=num_q_heads,
        num_k_heads=num_k_heads,
        rotary_dim=rotary_dim,
        eps=eps,
        q_norm_weight_bias=q_norm_weight_bias,
        k_norm_weight_bias=k_norm_weight_bias,
        subtract_mean_q=subtract_mean_q,
        subtract_mean_k=subtract_mean_k,
    )
