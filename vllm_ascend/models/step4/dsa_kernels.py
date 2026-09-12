"""Generated Step-4 minimal inference kernel.py; do not edit by hand.

Regenerate with ``python3 -m dsa_parity.build_step4_minimal_release``.
Numerical bodies come from the source manifest below.  Only merged-module
imports/metadata and the experimental Triton sliding-attention branch are
rewritten; sliding attention is the validated PyTorch SDPA implementation.
"""

from __future__ import annotations

# Source manifest (step4-minimal-inference-release-v1):
#   inference/dsa_kernels.py  sha256=7fa17fc7886bc20eaa9a2cb34c623b11ff152389442f43314199e3c8bcccd449
#   inference/dsa_meta.py  sha256=484ff7c6de71966cdc80d32ad58b091b49d730d3752f9b90b42f2e19d1df9e9f
#   inference/sparse_attn.py  sha256=22f733eb5fa0069dc11b688a2d3d3acd42125a4c379a660fef9454f783b49c2f
#   inference/dsa_attention.py  sha256=b3404cc32a8715b99bba14a0372e53e0ea5b35e582810536002dafa710494460
#   inference/fp8_gemm.py  sha256=808bad1273267b6587b582614e7745b80517d33f4a8cc79050fd2e93414ef943
#   inference/moe_kernels.py  sha256=c1c2c83d13758d9a16e6cec5795ecbd7a1081724f5d5098cda13ca8bf48a5f43
#   inference/qknorm_rope.py  sha256=f60bccb1bd62b0b6db268202ee526c501877bf66b5c6c9db5975dc6b9f5123ea



# ============================================================================
# merged source: inference/dsa_kernels.py
# ============================================================================
"""Triton kernels for Step-4's DeepSeek-style sparse attention (DSA).

Pure Triton on purpose: the reference implementations of this path depend either on
NVIDIA's CuTe-DSL package or on vLLM internals, neither of which can ship in an
open-source model release. Semantics here are pinned against the CuTe kernels at
Step-4 geometry; see ``dsa_parity/`` for the parity harness that enforces that.

Where this deviates from the CuTe kernels it is deliberate and noted at the point of
deviation. The one structural deviation: summaries are stored in a plain logical layout
rather than the CuTe cache's WGMMA XOR-swizzled byte order, because that swizzle exists
to feed Hopper WGMMA operands from shared memory and Triton chooses its own operand
layouts. Producer and consumer are both in this file, so the layout is ours to pick.
"""


import torch
import triton
import triton.language as tl

# Step-4 compresses every 8 consecutive tokens into one summary vector. The CuTe kernels
# hard-specialize on this; we keep it a parameter but never exercise anything else.
REGION_BLOCK_SIZE = 8
PROXY_DIM = 256


def shared_table_stride(block_table: torch.Tensor) -> int:
    """Row stride to pass for a block table, zero when the table is shared.

    A single-row table means "every request uses this mapping". Expressing that as a zero
    row stride rather than a branch inside the kernel keeps the two cases on one code path,
    and -- more importantly -- makes the shared case safe: the alternative, indexing a
    one-row table by a nonzero request id, reads past the tensor and returns plausible
    garbage instead of failing.
    """
    if block_table.ndim != 2:
        raise ValueError(
            f"block_table must be [num_reqs, pages], got {tuple(block_table.shape)}"
        )
    return 0 if block_table.shape[0] == 1 else block_table.stride(0)


@triton.jit
def _affine_normalize(
    raw,
    mean,
    rstd,
    weight_ptr,
    bias_ptr,
    offsets,
    mask,
    weight_bias,
    HAS_BIAS: tl.constexpr,
):
    """Apply ``(x - mean) * rstd * (w + weight_bias) [+ b]`` to one dim slice.

    A device function rather than a closure because Triton cannot compile nested
    ``def`` inside a kernel, and the normalization has to be applied to three different
    dim slices of the same head: the tail, and each half of the rotated span.
    """
    scale = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    result = (raw - mean) * rstd * (scale + weight_bias)
    if HAS_BIAS:
        result += tl.load(bias_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    return result


@triton.jit
def _indexer_norm_rope_kernel(
    x_ptr,
    out_ptr,
    weight_ptr,
    bias_ptr,
    cos_ptr,
    sin_ptr,
    positions_ptr,
    stride_x_token,
    stride_out_token,
    stride_cos,
    eps,
    weight_bias,
    head_dim: tl.constexpr,
    rotary_dim: tl.constexpr,
    SUBTRACT_MEAN: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_R: tl.constexpr,
) -> None:
    """Normalize one head of one token, then rotate its leading ``2 * rotary_dim`` dims.

    ``SUBTRACT_MEAN`` selects LayerNorm over RMSNorm. The indexer needs both: q is
    RMSNorm with a zero-centered weight, k is LayerNorm with a weight and a bias. They
    differ by one reduction and are trivial to confuse, which would shift every key
    without changing any shape.

    The rotated span is written from a separate load rather than by patching the
    normalized vector in place, because Triton cannot index a value tensor by another
    tensor -- and overlapping stores from one program would depend on store ordering the
    compiler does not promise.
    """
    token = tl.program_id(0)
    head = tl.program_id(1)
    base = token * stride_x_token + head * head_dim
    out_base = token * stride_out_token + head * head_dim

    dims = tl.arange(0, BLOCK_D)
    dim_ok = dims < head_dim
    values = tl.load(x_ptr + base + dims, mask=dim_ok, other=0.0).to(tl.float32)

    sum_sq = tl.sum(values * values)
    if SUBTRACT_MEAN:
        # Match the deployed LayerNorm reduction and fast-rsqrt path.  Computing
        # ``mean((x - mean) ** 2)`` is algebraically equivalent, but changes the
        # FP32 reduction/rounding before every sparse-indexer key.
        mean = tl.sum(values) / head_dim
        variance = sum_sq / head_dim - mean * mean
    else:
        mean = 0.0
        variance = sum_sq / head_dim
    rstd = tl.rsqrt(variance + eps)

    # Dims past the rotated span keep the plain normalized value.
    tl.store(
        out_ptr + out_base + dims,
        _affine_normalize(
            values,
            mean,
            rstd,
            weight_ptr,
            bias_ptr,
            dims,
            dim_ok,
            weight_bias,
            HAS_BIAS=HAS_BIAS,
        ).to(out_ptr.dtype.element_ty),
        mask=dim_ok & (dims >= 2 * rotary_dim),
    )

    half = tl.arange(0, BLOCK_R)
    half_ok = half < rotary_dim
    position = tl.load(positions_ptr + token).to(tl.int32)
    cos = tl.load(cos_ptr + position * stride_cos + half, mask=half_ok, other=0.0).to(tl.float32)
    sin = tl.load(sin_ptr + position * stride_cos + half, mask=half_ok, other=0.0).to(tl.float32)

    real = _affine_normalize(
        tl.load(x_ptr + base + half, mask=half_ok, other=0.0).to(tl.float32),
        mean,
        rstd,
        weight_ptr,
        bias_ptr,
        half,
        half_ok,
        weight_bias,
        HAS_BIAS=HAS_BIAS,
    ).to(out_ptr.dtype.element_ty).to(tl.float32)
    imaginary = _affine_normalize(
        tl.load(x_ptr + base + rotary_dim + half, mask=half_ok, other=0.0).to(tl.float32),
        mean,
        rstd,
        weight_ptr,
        bias_ptr,
        rotary_dim + half,
        half_ok,
        weight_bias,
        HAS_BIAS=HAS_BIAS,
    ).to(out_ptr.dtype.element_ty).to(tl.float32)
    tl.store(
        out_ptr + out_base + half,
        (real * cos - imaginary * sin).to(out_ptr.dtype.element_ty),
        mask=half_ok,
    )
    tl.store(
        out_ptr + out_base + rotary_dim + half,
        (real * sin + imaginary * cos).to(out_ptr.dtype.element_ty),
        mask=half_ok,
    )


def _norm_rope(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    *,
    num_heads: int,
    head_dim: int,
    rotary_dim: int,
    eps: float,
    weight_bias: float,
    subtract_mean: bool,
) -> torch.Tensor:
    out = torch.empty_like(x)
    _indexer_norm_rope_kernel[(x.shape[0], num_heads)](
        x,
        out,
        weight,
        bias if bias is not None else weight,
        cos,
        sin,
        positions,
        x.stride(0),
        out.stride(0),
        cos.stride(0),
        eps,
        weight_bias,
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        SUBTRACT_MEAN=subtract_mean,
        HAS_BIAS=bias is not None,
        BLOCK_D=triton.next_power_of_2(head_dim),
        BLOCK_R=triton.next_power_of_2(rotary_dim),
    )
    return out


def indexer_norm_rope(
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    index_z: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    k_norm_bias: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    *,
    head_dim: int,
    num_q_heads: int,
    num_k_heads: int,
    rotary_dim: int,
    eps: float = 1e-6,
    q_norm_weight_bias: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare the indexer's q/k/z from their fused projection.

    Args:
        index_q: ``[tokens, num_q_heads * head_dim]``.
        index_k: ``[tokens, num_k_heads * head_dim]``.
        index_z: ``[tokens, num_k_heads * head_dim]`` -- returned untouched.
        cos, sin: ``[seq, rotary_dim]``, half the width of the rotated span.
        q_norm_weight_bias: ``1.0`` for a zero-centered RMSNorm weight (``x * (1 + w)``),
            ``0.0`` otherwise. This follows the checkpoint's norm class, so it must be
            threaded through from config rather than hardcoded.
    """
    return (
        _norm_rope(
            index_q,
            q_norm_weight,
            None,
            cos,
            sin,
            positions,
            num_heads=num_q_heads,
            head_dim=head_dim,
            rotary_dim=rotary_dim,
            eps=eps,
            weight_bias=q_norm_weight_bias,
            subtract_mean=False,
        ),
        _norm_rope(
            index_k,
            k_norm_weight,
            k_norm_bias,
            cos,
            sin,
            positions,
            num_heads=num_k_heads,
            head_dim=head_dim,
            rotary_dim=rotary_dim,
            eps=eps,
            weight_bias=0.0,
            subtract_mean=True,
        ),
        index_z.contiguous(),
    )


@triton.jit
def _csa_compress_regions_kernel(
    index_k_ptr,
    index_z_ptr,
    token_start_ptr,
    token_count_ptr,
    summary_ptr,
    mean_ptr,
    stride_token,
    stride_head,
    stride_summary_region,
    stride_summary_head,
    num_heads: tl.constexpr,
    proxy_dim: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    WRITE_FP32: tl.constexpr,
) -> None:
    """One program per (region, head): compress a region's tokens into one vector.

    The weight is a softmax over the token axis of ``index_z`` taken **per dimension**,
    so this is an element-wise weighted mean, not a convex combination of whole key
    vectors:

        s_d = sum_t exp(z_td - m) * k_td / sum_t exp(z_td - m)

    ``m`` is a scalar over both axes. It cancels algebraically and only exists to keep
    ``exp`` in range -- but it is the *same* scalar the CuTe kernel uses, which is why
    the two agree in the last places rather than merely within tolerance.
    """
    region = tl.program_id(0)
    head = tl.program_id(1)

    start = tl.load(token_start_ptr + region).to(tl.int64)
    count = tl.load(token_count_ptr + region).to(tl.int64)

    tokens = tl.arange(0, BLOCK_T)
    dims = tl.arange(0, BLOCK_D)
    token_ok = tokens < count
    dim_ok = dims < proxy_dim
    mask = token_ok[:, None] & dim_ok[None, :]

    offsets = (
        (start + tokens.to(tl.int64))[:, None] * stride_token
        + head * stride_head
        + dims[None, :]
    )
    logits = tl.load(index_z_ptr + offsets, mask=mask, other=float("-inf")).to(tl.float32)
    values = tl.load(index_k_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    shift = tl.max(tl.where(mask, logits, float("-inf")))
    weights = tl.where(mask, tl.exp(logits - shift), 0.0)
    denominator = tl.sum(weights, axis=0)
    numerator = tl.sum(weights * values, axis=0)
    summary = tl.where(
        denominator > 0.0, numerator / tl.maximum(denominator, 1e-20), 0.0
    )

    out = region * stride_summary_region + head * stride_summary_head + dims
    if WRITE_FP32:
        tl.store(summary_ptr + out, summary, mask=dim_ok)
    # Round through bfloat16 on the way to e4m3, matching the CuTe kernel. Going
    # straight from fp32 is observably different: bf16 keeps 8 mantissa bits and e4m3
    # keeps 3, so a value sitting on an e4m3 midpoint can round the other way once bf16
    # has already moved it. The parity gate compares bytes, so this is not cosmetic.
    tl.store(mean_ptr + out, summary.to(tl.bfloat16).to(tl.float8e4nv), mask=dim_ok)


def csa_region_layout(
    seq_lens: torch.Tensor, *, regions_per_seq: int, region_size: int = REGION_BLOCK_SIZE
) -> tuple[torch.Tensor, torch.Tensor]:
    """Token span of every region, for sequences packed back to back.

    ``regions_per_seq`` is the stride of the region id space, not the number of regions a
    sequence actually fills -- a sequence's regions have to land at a fixed offset so the
    summary store can be indexed without a second lookup. Regions past a sequence's end
    get a count of zero and are skipped rather than written, which keeps a stale summary
    from a previous request out of the selection.
    """
    device = seq_lens.device
    lengths = seq_lens.to(torch.int64)
    token_starts = torch.cumsum(lengths, dim=0) - lengths

    regions = torch.arange(regions_per_seq, device=device, dtype=torch.int64)
    offset = regions.view(1, -1) * region_size
    count = (lengths.view(-1, 1) - offset).clamp(0, region_size)
    start = token_starts.view(-1, 1) + offset
    return start.reshape(-1).to(torch.int32), count.reshape(-1).to(torch.int32)


def csa_compress_regions(
    index_k: torch.Tensor,
    index_z: torch.Tensor,
    token_start: torch.Tensor,
    token_count: torch.Tensor,
    *,
    region_size: int = REGION_BLOCK_SIZE,
    write_fp32: bool = True,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    """Compress packed tokens into per-region summaries.

    Args:
        index_k: ``[tokens, heads, proxy_dim]`` proxy keys.
        index_z: ``[tokens, heads, proxy_dim]`` per-dimension softmax logits.
        token_start: ``[regions]`` int32, first token of each region.
        token_count: ``[regions]`` int32, tokens present in each region; 0 = skip.
        write_fp32: also produce the fp32 copy. Every vLLM consumer only reads the
            fp8 tensor; the fp32 one exists for parity attribution, and skipping it
            saves a full-size zeros allocation per call.

    Returns:
        ``(summary_fp32, summary_fp8)``, both ``[regions, heads, proxy_dim]``. The fp8
        tensor is what selection and sparse attention consume; the fp32 one exists
        because it is the only place the unrounded value is observable, which is what
        lets a parity failure be attributed to the reduction rather than the rounding.
        ``summary_fp32`` is ``None`` when ``write_fp32=False``.
    """
    if index_k.shape != index_z.shape:
        raise ValueError(
            f"index_k and index_z must match, got {tuple(index_k.shape)} and "
            f"{tuple(index_z.shape)}"
        )
    if index_k.ndim != 3:
        raise ValueError(f"index_k must be [tokens, heads, proxy_dim], got {tuple(index_k.shape)}")
    if index_k.stride(2) != 1:
        raise ValueError("index_k must be contiguous in the proxy dimension")

    regions = int(token_start.numel())
    _, heads, proxy_dim = index_k.shape
    mean = torch.zeros((regions, heads, proxy_dim), device=index_k.device, dtype=torch.float8_e4m3fn)
    if write_fp32:
        summary = torch.zeros((regions, heads, proxy_dim), device=index_k.device, dtype=torch.float32)
    else:
        # The fp32 store is compiled out; aliasing the fp8 output keeps the
        # pointer argument valid and the strides correct for the live store.
        summary = mean

    _csa_compress_regions_kernel[(regions, heads)](
        index_k,
        index_z,
        token_start,
        token_count,
        summary,
        mean,
        index_k.stride(0),
        index_k.stride(1),
        mean.stride(0),
        mean.stride(1),
        num_heads=heads,
        proxy_dim=proxy_dim,
        BLOCK_T=triton.next_power_of_2(region_size),
        BLOCK_D=triton.next_power_of_2(proxy_dim),
        WRITE_FP32=write_fp32,
    )
    return (None if not write_fp32 else summary), mean


@triton.jit
def _append_regions_kernel(
    pending_k_ptr,
    pending_z_ptr,
    chunk_k_ptr,
    chunk_z_ptr,
    meta_ptr,
    summary_ptr,
    meta_stride,
    stride_pending_row,
    stride_pending_tok,
    stride_chunk_tok,
    stride_summary_row,
    stride_summary_region,
    REGION_SIZE: tl.constexpr,
    PROXY_DIM_C: tl.constexpr,
) -> None:
    """Batched prefill append: compress each request's newly completed regions.

    One program per (request, region index within this append). The region's
    tokens live in two places -- the first few in the request's pending tail
    buffer, the rest in the packed prefill chunk -- and both are read directly,
    so no per-request ``cat``/gather staging tensor is materialized. The
    softmax-weighted-mean body is byte-identical to
    ``_csa_compress_regions_kernel`` (including the fp32 -> bf16 -> e4m3 double
    rounding), and the result is stored straight into the fp8 summary cache.

    ``meta`` is an ``[R, >= 7]`` int32 device tensor, per request:
    ``[row, r0, delta, plen, ts, n_regions, q_len]`` where ``r0`` is the first
    newly completed region, ``delta`` the misalignment of the concat stream
    (0 in the steady state), ``plen`` the effective pending-tail length,
    ``ts`` the request's first token offset inside the chunk.
    """
    req = tl.program_id(0)
    j = tl.program_id(1)

    row = tl.load(meta_ptr + req * meta_stride + 0).to(tl.int64)
    n_regions = tl.load(meta_ptr + req * meta_stride + 5).to(tl.int64)
    if j >= n_regions:
        return
    r0 = tl.load(meta_ptr + req * meta_stride + 1).to(tl.int64)
    delta = tl.load(meta_ptr + req * meta_stride + 2).to(tl.int64)
    plen = tl.load(meta_ptr + req * meta_stride + 3).to(tl.int64)
    ts = tl.load(meta_ptr + req * meta_stride + 4).to(tl.int64)

    toks = tl.arange(0, REGION_SIZE)
    dims = tl.arange(0, PROXY_DIM_C)
    # Position within the virtual concat stream (pending tail ++ chunk).
    v = delta + j * REGION_SIZE + toks.to(tl.int64)
    from_pending = v < plen
    v_src = tl.where(from_pending, v, v - plen)
    off_pending = row * stride_pending_row + v_src * stride_pending_tok
    off_chunk = (ts + v_src) * stride_chunk_tok

    mask = from_pending[:, None] & (dims < PROXY_DIM_C)[None, :]
    zp = tl.load(pending_z_ptr + off_pending[:, None] + dims[None, :], mask=mask, other=0.0)
    kp = tl.load(pending_k_ptr + off_pending[:, None] + dims[None, :], mask=mask, other=0.0)
    mask_c = (~from_pending)[:, None] & (dims < PROXY_DIM_C)[None, :]
    zc = tl.load(chunk_z_ptr + off_chunk[:, None] + dims[None, :], mask=mask_c, other=0.0)
    kc = tl.load(chunk_k_ptr + off_chunk[:, None] + dims[None, :], mask=mask_c, other=0.0)

    logits = tl.where(from_pending[:, None], zp, zc).to(tl.float32)
    values = tl.where(from_pending[:, None], kp, kc).to(tl.float32)

    shift = tl.max(logits)
    weights = tl.exp(logits - shift)
    denominator = tl.sum(weights, axis=0)
    numerator = tl.sum(weights * values, axis=0)
    summary = tl.where(
        denominator > 0.0, numerator / tl.maximum(denominator, 1e-20), 0.0
    )

    out = (
        row * stride_summary_row
        + (r0 + j) * stride_summary_region
        + dims
    )
    tl.store(
        summary_ptr + out,
        summary.to(tl.bfloat16).to(tl.float8e4nv),
        mask=dims < PROXY_DIM_C,
    )


@triton.jit
def _write_pending_tail_kernel(
    pending_k_ptr,
    pending_z_ptr,
    chunk_k_ptr,
    chunk_z_ptr,
    meta_ptr,
    meta_stride,
    stride_pending_row,
    stride_pending_tok,
    stride_chunk_tok,
    REGION_SIZE: tl.constexpr,
    PROXY_DIM_C: tl.constexpr,
) -> None:
    """Store each request's new ragged tail into the pending buffer.

    The tail is the last ``(plen + q_len) % REGION_SIZE`` tokens of the concat
    stream. Slots past the tail are written with clamped garbage -- they are
    never read, because every read of the pending buffer is bounded by
    ``pending_len``.
    """
    req = tl.program_id(0)

    row = tl.load(meta_ptr + req * meta_stride + 0).to(tl.int64)
    plen = tl.load(meta_ptr + req * meta_stride + 3).to(tl.int64)
    q_len = tl.load(meta_ptr + req * meta_stride + 6).to(tl.int64)
    ts = tl.load(meta_ptr + req * meta_stride + 4).to(tl.int64)
    total = plen + q_len
    tail = total % REGION_SIZE

    toks = tl.arange(0, REGION_SIZE)
    dims = tl.arange(0, PROXY_DIM_C)
    v = total - tail + toks.to(tl.int64)
    # Clamp keeps the masked-out lanes' addresses in bounds.
    v = tl.maximum(tl.minimum(v, total - 1), 0)
    from_pending = v < plen
    v_src = tl.where(from_pending, v, v - plen)
    off_pending = row * stride_pending_row + v_src * stride_pending_tok
    off_chunk = (ts + v_src) * stride_chunk_tok

    mask = (toks < REGION_SIZE)[:, None] & (dims < PROXY_DIM_C)[None, :]
    mask_p = mask & from_pending[:, None]
    mask_c = mask & (~from_pending)[:, None]
    kp = tl.load(pending_k_ptr + off_pending[:, None] + dims[None, :], mask=mask_p, other=0.0)
    z_p = tl.load(pending_z_ptr + off_pending[:, None] + dims[None, :], mask=mask_p, other=0.0)
    kc = tl.load(chunk_k_ptr + off_chunk[:, None] + dims[None, :], mask=mask_c, other=0.0)
    zc = tl.load(chunk_z_ptr + off_chunk[:, None] + dims[None, :], mask=mask_c, other=0.0)

    dst = row * stride_pending_row + toks.to(tl.int64) * stride_pending_tok
    tl.store(pending_k_ptr + dst[:, None] + dims[None, :], tl.where(from_pending[:, None], kp, kc), mask=mask)
    tl.store(pending_z_ptr + dst[:, None] + dims[None, :], tl.where(from_pending[:, None], z_p, zc), mask=mask)


def append_regions(
    pending_key: torch.Tensor,
    pending_z: torch.Tensor,
    chunk_k: torch.Tensor,
    chunk_z: torch.Tensor,
    meta: torch.Tensor,
    summary: torch.Tensor,
    *,
    region_size: int = REGION_BLOCK_SIZE,
    max_new_regions: int,
) -> None:
    """Compress and store all requests' newly completed regions in one launch.

    See ``_append_regions_kernel`` for the ``meta`` layout. ``max_new_regions``
    is a host-side upper bound on column 5 of ``meta``; extra programs exit
    after reading their count. The tail write is a second launch over the same
    metadata (``write_pending_tail``); splitting them keeps each kernel's block
    shape trivial.
    """
    rows = int(meta.shape[0])
    proxy_dim = int(pending_key.shape[-1])
    _append_regions_kernel[(rows, max_new_regions)](
        pending_key,
        pending_z,
        chunk_k,
        chunk_z,
        meta,
        summary,
        meta.stride(0),
        pending_key.stride(0),
        pending_key.stride(1),
        chunk_k.stride(0),
        summary.stride(0),
        summary.stride(1),
        REGION_SIZE=region_size,
        PROXY_DIM_C=proxy_dim,
    )


def write_pending_tail(
    pending_key: torch.Tensor,
    pending_z: torch.Tensor,
    chunk_k: torch.Tensor,
    chunk_z: torch.Tensor,
    meta: torch.Tensor,
    *,
    region_size: int = REGION_BLOCK_SIZE,
) -> None:
    """Refresh every request's pending tail from the concat stream (one launch)."""
    rows = int(meta.shape[0])
    proxy_dim = int(pending_key.shape[-1])
    _write_pending_tail_kernel[(rows,)](
        pending_key,
        pending_z,
        chunk_k,
        chunk_z,
        meta,
        meta.stride(0),
        pending_key.stride(0),
        pending_key.stride(1),
        chunk_k.stride(0),
        REGION_SIZE=region_size,
        PROXY_DIM_C=proxy_dim,
    )


@triton.jit
def _indexer_logits_kernel(
    index_q_ptr,
    mean_cache_ptr,
    weights_ptr,
    out_ptr,
    block_table_ptr,
    token_req_ptr,
    seq_k,
    block_table_cols,
    stride_q_token,
    stride_q_group,
    stride_q_head,
    stride_w_token,
    stride_w_group,
    stride_keys_page,
    stride_keys_region,
    stride_bt_row,
    stride_out_row,
    out_q_stride,
    heads_per_group: tl.constexpr,
    proxy_dim: tl.constexpr,
    GROUPS: tl.constexpr,
    SUMMARIES_PER_PAGE: tl.constexpr,
    BLOCK_K: tl.constexpr,
) -> None:
    """Weighted-ReLU indexer scores against the paged fp8 summary cache.

        score[t, r] = sum_h relu(dot(q_th, k_r)) * w_th

    One program per (query token, key tile), one kernel launch for the whole
    batch -- packed prefill and decode share the same form. Query token ``t``
    scores against its own request's key stream: ``token_req_ptr[t]`` selects
    the block-table row, and logical region ``r`` of that row lives at
    physical region ``block_table[req][r // SUMMARIES_PER_PAGE] *
    SUMMARIES_PER_PAGE + r % SUMMARIES_PER_PAGE``. Unmapped pages (``-1``)
    contribute zeros; dead columns past a row's live history hold zeros or
    stale scores, which the top-k stage never reads because each row's
    candidate range is bounded by its own history length.

    The dot product is a multiply-and-reduce rather than ``tl.dot`` on
    purpose: one form serves prefill and decode alike, and the shape is
    memory-bound on the summary read either way. Both operands are e4m3
    and products and accumulation are fp32, mirroring the upstream SM90
    WGMMA (``Float8E4M3FN x Float8E4M3FN -> Float32``): widening an e4m3
    value to fp32 is exact, so in-register conversion preserves the
    quantized value set bit-for-bit.

    The ReLU is per head and *inside* the sum, so the heads cannot be folded into a
    single matmul -- hence the loop. The per-head weights are signed, so applying
    ReLU to each dot does **not** make the final weighted sum non-negative. Exact
    zeros are still common because a score is zero whenever every head dot is
    negative; for Step-4's 4 heads per group that event is about 6.25%. Those zeros
    are what the top-k tie-break then has to resolve, so they are load-bearing,
    not noise.

    The e4m3 activation quantization of the query stays on the host on
    purpose: triton-ascend's in-register fp8 convert does not match
    torch_npu's ``.to(float8_e4m3fn)`` rounding, and the deployed semantics
    are the torch ones (the mismatch rate on random bf16 values is ~93%,
    far beyond accumulation-order noise). Upstream pre-quantizes q on the
    host for its WGMMA entrypoint the same way; both sides then read the
    fp8 tensors directly.

    The group dimension is a loop inside the kernel rather than a grid axis: the
    key tile is loaded once per (token, k_block) program and every group reuses
    that UB-resident copy instead of re-reading it from global memory.

    ``out_q_stride`` lets the kernel write into a group-major
    ``[groups * total_q, width]`` buffer: token ``t`` of the batch lands at row
    ``group * out_q_stride + t``.
    """
    token = tl.program_id(0)
    k_block = tl.program_id(1)
    keys = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
    dims = tl.arange(0, proxy_dim)
    key_ok = keys < seq_k

    req = tl.load(token_req_ptr + token).to(tl.int64)
    page_cols = keys // SUMMARIES_PER_PAGE
    col_ok = page_cols < block_table_cols
    pages = tl.load(
        block_table_ptr + req * stride_bt_row + page_cols,
        mask=key_ok & col_ok,
        other=-1,
    ).to(tl.int64)
    load_ok = key_ok & col_ok & (pages >= 0)
    safe_pages = tl.where(load_ok, pages, 0)
    key_offsets = (
        safe_pages * stride_keys_page + (keys % SUMMARIES_PER_PAGE) * stride_keys_region
    )
    k_tile = tl.load(
        mean_cache_ptr + key_offsets[:, None] + dims[None, :],
        mask=load_ok[:, None],
        other=0.0,
    ).to(tl.float32)

    for group in tl.range(0, GROUPS):
        acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
        for head in range(heads_per_group):
            q_vec = tl.load(
                index_q_ptr
                + token.to(tl.int64) * stride_q_token
                + group * stride_q_group
                + head * stride_q_head
                + dims
            ).to(tl.float32)
            head_weight = tl.load(
                weights_ptr + token * stride_w_token + group * stride_w_group + head
            ).to(tl.float32)
            dots = tl.sum(k_tile * q_vec[None, :], axis=1)
            acc += tl.maximum(dots, 0.0) * head_weight
        out_row = group * out_q_stride + token.to(tl.int64)
        tl.store(
            out_ptr + out_row * stride_out_row + keys,
            acc,
            mask=key_ok,
        )


def quantize_activations_e4m3(tensor: torch.Tensor) -> torch.Tensor:
    """Quantize indexer activations to e4m3, aligned with the deployed kernels.

    The reference SM90 kernels feed e4m3 operands into a fp32-accumulating
    WGMMA, and the checkpoint stores these activations in bf16, so the
    quantization is part of the semantics: skipping it changes which regions
    get selected. The quantization itself stays on the host on purpose --
    triton-ascend's in-register fp8 convert does not match torch_npu's
    ``.to(float8_e4m3fn)`` rounding -- mirroring upstream, which also
    pre-quantizes q on the host before its WGMMA entrypoint.
    """
    return tensor.to(torch.float8_e4m3fn)


def indexer_logits_paged(
    index_q: torch.Tensor,
    weights: torch.Tensor,
    mean_cache: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    *,
    out: torch.Tensor,
    out_q_stride: int | None = None,
    summaries_per_page: int,
    block_k: int = 32,
) -> torch.Tensor:
    """Score a packed batch's query tokens against the paged fp8 summaries.

    One kernel launch for the whole batch, prefill or decode: query token
    ``t`` is scored against the key stream of request ``token_to_req[t]``,
    and the results land at logits rows ``group * out_q_stride + t`` of the
    group-major buffer. The token -> request map covers every batch shape:
    a plain decode batch passes ``torch.arange(n)`` (one token per request),
    a spec-decode verify batch repeats each request id ``k`` times, and a
    packed prefill batch maps each chunk token to its request. Dead columns
    past a row's live history hold zeros or stale scores, which the top-k
    stage never reads because each row's candidate range is bounded by its
    own history length.

    Args:
        index_q: ``[total_tokens, groups, heads_per_group, proxy_dim]`` bf16,
            packed over all requests; quantized to e4m3 on the host so the
            kernel scores e4m3 x e4m3 -> fp32 like the upstream WGMMA.
        weights: ``[total_tokens, groups, heads_per_group]`` fp32, prescaled.
        mean_cache: fp8 ``[num_pages, summaries_per_page, 1, proxy_dim]``
            (any row stride).
        block_table: int32 ``[num_reqs, pages]`` KV block table.
        token_to_req: int32 ``[total_tokens]``; the block-table row (request)
            each query token belongs to.
        out: ``[groups * total_tokens, seq_k]`` fp32 destination, where
            ``seq_k = out.shape[1]``.
        summaries_per_page: regions per physical page (``block_size //
            region_block_size``).
    """
    total_tokens, groups, heads_per_group, proxy_dim = index_q.shape
    seq_k = int(out.shape[1])
    out_q_stride = total_tokens if out_q_stride is None else out_q_stride
    index_q = quantize_activations_e4m3(index_q)

    _indexer_logits_kernel[
        (total_tokens, triton.cdiv(seq_k, block_k))
    ](
        index_q,
        mean_cache,
        weights,
        out,
        block_table,
        token_to_req,
        seq_k,
        int(block_table.shape[1]),
        index_q.stride(0),
        index_q.stride(1),
        index_q.stride(2),
        weights.stride(0),
        weights.stride(1),
        int(mean_cache.stride(0)),
        int(mean_cache.stride(1)),
        int(block_table.stride(0)),
        out.stride(0),
        out_q_stride,
        heads_per_group=heads_per_group,
        proxy_dim=proxy_dim,
        GROUPS=groups,
        SUMMARIES_PER_PAGE=int(summaries_per_page),
        BLOCK_K=block_k,
    )
    return out


# Prefill selection packs one entry as ``phys_region | (valid_tokens << 24)``.
# These are ``tl.constexpr`` because Triton kernels cannot read plain module globals.
PREFILL_REGION_VALID_SHIFT = tl.constexpr(24)
RADIX_BITS = tl.constexpr(8)
RADIX_BUCKETS = tl.constexpr(256)
RADIX_PASSES = tl.constexpr(4)


@triton.jit
def _ordered_float_key(scores):
    """Map fp32 values to monotonically ordered unsigned-32 keys held in int64.

    Negative IEEE-754 values reverse their bit ordering, while non-negative values
    become ordered after setting the sign bit. Keeping the resulting 32-bit pattern
    in a non-negative int64 lets ordinary Triton integer comparisons implement the
    required unsigned ordering. Branching on ``scores < 0`` deliberately maps
    ``-0.0`` and ``+0.0`` to the same key, matching numeric equality and CuTe.

    NaNs follow CuTe's raw ordered-float behavior and sort above ``+inf``. Production
    indexer scores are expected to be finite; this helper does not add a costly
    full-tensor finiteness scan to the hot path.
    """
    bits = scores.to(tl.int32, bitcast=True).to(tl.int64) & 0xFFFFFFFF
    return tl.where(
        scores < 0.0,
        (~bits) & 0xFFFFFFFF,
        bits | 0x80000000,
    )


@triton.jit
def _radix_topk_threshold(
    logits_ptr,
    row,
    stride_logits_row,
    visible,
    budget,
    seq_regions,
    BLOCK_R: tl.constexpr,
):
    """Find the exact score threshold separating the top ``budget`` entries of one row.

    Returns ``(threshold, remaining)``: the fp32 bit pattern of the ``budget``-th largest
    score, and how many entries *at* that exact key still belong in the selection. A
    threshold plus a tie count is what lets the caller take a **prefix** of a tied group
    rather than all or none of it, without ever materializing a sort.

    Scores may be signed. :func:`_ordered_float_key` converts their IEEE-754 bit
    patterns into unsigned-monotone keys before the radix passes.
    """
    threshold = budget.to(tl.int64) * 0
    remaining = budget
    bucket_ids = tl.arange(0, RADIX_BUCKETS)

    # Narrow the threshold one byte at a time, most significant first.
    for radix_pass in tl.static_range(RADIX_PASSES):
        shift = 32 - RADIX_BITS * (radix_pass + 1)
        histogram = tl.zeros((RADIX_BUCKETS,), dtype=tl.int32)
        for base in tl.range(0, seq_regions, BLOCK_R):
            regions = base + tl.arange(0, BLOCK_R)
            live = regions < visible
            scores = tl.load(
                logits_ptr + row * stride_logits_row + regions, mask=live, other=0.0
            )
            keys = _ordered_float_key(scores)
            if radix_pass == 0:
                # No high bits are fixed yet. Testing them would need a shift by 32,
                # which is undefined.
                matches = live
            else:
                high = shift + RADIX_BITS
                matches = live & ((keys >> high) == (threshold >> high))
            # Park non-matching lanes in bucket 0 and subtract them back out. Relying on
            # tl.histogram to drop an out-of-range sentinel would corrupt the bucket
            # totals if it clamps instead -- and the totals are what pick the cut, so the
            # failure would look like a wrong threshold rather than a bad histogram.
            buckets = tl.where(
                matches,
                ((keys >> shift) & (RADIX_BUCKETS - 1)).to(tl.int32),
                0,
            )
            counts = tl.histogram(buckets, RADIX_BUCKETS)
            parked = tl.sum((~matches).to(tl.int32))
            histogram += counts - tl.where(bucket_ids == 0, parked, 0)

        total = tl.sum(histogram)
        above = total - tl.cumsum(histogram, axis=0)
        # ``above`` is non-increasing, so the buckets satisfying above < remaining form a
        # suffix; the smallest of them is the one holding the cut.
        chosen = tl.min(tl.where(above < remaining, bucket_ids, RADIX_BUCKETS))
        remaining -= tl.sum(tl.where(bucket_ids > chosen, histogram, 0))
        # Cast before shifting the MSB byte. An int32 bucket >= 128 would otherwise
        # become negative first and sign-extend into the int64 threshold.
        threshold |= chosen.to(tl.int64) << shift

    return threshold, remaining


@triton.jit
def _region_topk_pack_kernel(
    logits_ptr,
    lengths_ptr,
    q_positions_ptr,
    block_table_ptr,
    request_ptr,
    out_ptr,
    seq_regions,
    stride_logits_row,
    stride_table_row,
    stride_out_row,
    topk: tl.constexpr,
    region_size: tl.constexpr,
    regions_per_page: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
) -> None:
    """Select the top-``topk`` visible regions of one row and pack them.

    Selection is ``torch.topk`` with ties broken by ascending region id: order the
    visible scores by ``(-score, id)`` and keep the first ``min(topk, visible)``. Ties at
    the cut take a *prefix* of the tied group, they do not drop it -- and that case is
    common rather than exotic, because post-ReLU indexer scores are exactly zero about
    6.25% of the time, so the group sitting at the cut is usually the zero group.
    """
    row = tl.program_id(0)
    visible = tl.load(lengths_ptr + row).to(tl.int32)
    visible = tl.minimum(tl.maximum(visible, 0), seq_regions)
    q_position = tl.load(q_positions_ptr + row).to(tl.int32)
    request = tl.load(request_ptr + row).to(tl.int32)

    slots = tl.arange(0, BLOCK_TOPK)
    tl.store(out_ptr + row * stride_out_row + slots, -1, mask=slots < topk)
    if visible <= 0:
        return

    budget = tl.minimum(topk, visible)
    threshold, remaining = _radix_topk_threshold(
        logits_ptr, row, stride_logits_row, visible, budget, seq_regions, BLOCK_R=BLOCK_R
    )

    # Emit in ascending region order. The CuTe selector emits in radix-bucket order, but
    # that order is not part of the contract -- consumers compare sets -- and ascending
    # is both cheaper here and what the standalone metadata converter produces.
    cursor = 0
    ties_taken = 0
    for base in tl.range(0, seq_regions, BLOCK_R):
        regions = base + tl.arange(0, BLOCK_R)
        live = regions < visible
        scores = tl.load(
            logits_ptr + row * stride_logits_row + regions, mask=live, other=0.0
        )
        keys = _ordered_float_key(scores)
        greater = live & (keys > threshold)
        tied = live & (keys == threshold)
        tied_rank = ties_taken + tl.cumsum(tied.to(tl.int32), axis=0) - 1
        selected = greater | (tied & (tied_rank < remaining))

        positions = cursor + tl.cumsum(selected.to(tl.int32), axis=0) - 1
        safe_regions = tl.where(selected, regions, 0)
        pages = tl.load(
            block_table_ptr
            + request * stride_table_row
            + safe_regions // regions_per_page,
            mask=selected,
            other=0,
        )
        physical = pages * regions_per_page + safe_regions % regions_per_page
        valid_tokens = tl.minimum(
            tl.maximum(q_position + 1 - safe_regions * region_size, 0), region_size
        )
        packed = physical | (valid_tokens << PREFILL_REGION_VALID_SHIFT)
        # An unmapped page is a legal block-table state, not an error. It has to become a
        # ``-1`` slot rather than an address computed from a negative page, which would
        # read from before the cache.
        packed = tl.where(pages >= 0, packed, -1)
        tl.store(
            out_ptr + row * stride_out_row + positions,
            packed,
            mask=selected & (positions < topk),
        )
        cursor += tl.sum(selected.to(tl.int32))
        ties_taken += tl.sum(tied.to(tl.int32))


@triton.jit
def _region_topk_ids_kernel(
    logits_ptr,
    lengths_ptr,
    out_ptr,
    seq_regions,
    stride_logits_row,
    stride_out_row,
    topk: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
) -> None:
    """Select the top-``topk`` visible regions and emit **logical ids**, ascending.

    The same selection as :func:`_region_topk_pack_kernel` with the packing left off.
    Prefill and decode pack the identical selection into two different words, and the
    decode word needs the logical region id, which the prefill word does not carry -- so
    decode metadata cannot be derived from prefill metadata, only from the raw ids.
    """
    row = tl.program_id(0)
    visible = tl.load(lengths_ptr + row).to(tl.int32)
    visible = tl.minimum(tl.maximum(visible, 0), seq_regions)

    slots = tl.arange(0, BLOCK_TOPK)
    tl.store(out_ptr + row * stride_out_row + slots, -1, mask=slots < topk)
    if visible <= 0:
        return

    budget = tl.minimum(topk, visible)
    threshold, remaining = _radix_topk_threshold(
        logits_ptr, row, stride_logits_row, visible, budget, seq_regions, BLOCK_R=BLOCK_R
    )

    cursor = 0
    ties_taken = 0
    for base in tl.range(0, seq_regions, BLOCK_R):
        regions = base + tl.arange(0, BLOCK_R)
        live = regions < visible
        scores = tl.load(
            logits_ptr + row * stride_logits_row + regions, mask=live, other=0.0
        )
        keys = _ordered_float_key(scores)
        greater = live & (keys > threshold)
        tied = live & (keys == threshold)
        tied_rank = ties_taken + tl.cumsum(tied.to(tl.int32), axis=0) - 1
        selected = greater | (tied & (tied_rank < remaining))

        positions = cursor + tl.cumsum(selected.to(tl.int32), axis=0) - 1
        tl.store(
            out_ptr + row * stride_out_row + positions,
            regions,
            mask=selected & (positions < topk),
        )
        cursor += tl.sum(selected.to(tl.int32))
        ties_taken += tl.sum(tied.to(tl.int32))


def region_topk_ids(
    logits: torch.Tensor,
    lengths: torch.Tensor,
    *,
    topk: int,
    block_r: int = 1024,
) -> torch.Tensor:
    """Top-k region selection, returning logical region ids in ascending order.

    Args:
        logits: ``[rows, seq_regions]`` float32 indexer scores, which may be signed.
        lengths: ``[rows]`` int32, candidate region count per row. For DSA this is the
            **history** length, ``q_position // region_size``, which excludes the region
            containing the query itself -- that one is appended by the metadata builder
            rather than made to compete.

    Returns:
        ``[rows, topk]`` int32 logical region ids, ``-1`` padding the tail.
    """
    if logits.dtype != torch.float32:
        raise ValueError(f"selector expects float32 scores, got {logits.dtype}")

    rows, seq_regions = logits.shape
    out = torch.empty((rows, int(topk)), device=logits.device, dtype=torch.int32)
    _region_topk_ids_kernel[(rows,)](
        logits,
        lengths,
        out,
        seq_regions,
        logits.stride(0),
        out.stride(0),
        topk=int(topk),
        BLOCK_R=min(block_r, 256),
        BLOCK_TOPK=triton.next_power_of_2(int(topk)),
    )
    return out


def region_topk_pack(
    logits: torch.Tensor,
    lengths: torch.Tensor,
    q_positions: torch.Tensor,
    block_table: torch.Tensor,
    request_indices: torch.Tensor | None = None,
    *,
    topk: int,
    region_size: int = REGION_BLOCK_SIZE,
    regions_per_page: int,
    block_r: int = 1024,
) -> torch.Tensor:
    """Top-k region selection fused with prefill metadata packing.

    Args:
        logits: ``[rows, seq_regions]`` float32 indexer scores, which may be signed, where
            ``rows = groups * seq_q`` in group-major order.
        lengths: ``[rows]`` int32, causally visible region count per row.
        q_positions: ``[rows]`` int32, absolute token position of each row's query.
        block_table: ``[num_reqs, pages]`` int32, logical page -> physical page, one row
            per request. ``-1`` marks an unmapped page. A table with a single row is
            treated as shared by every request, regardless of ``request_indices``.
        request_indices: ``[rows]`` int32, which block-table row each score row belongs
            to. Defaults to all-zero.

    Returns:
        ``[rows, topk]`` int32, each valid slot holding
        ``phys_region | (valid_tokens << 24)``, with ``-1`` padding the tail. A row fills
        exactly ``min(topk, visible)`` slots.
    """
    if logits.dtype != torch.float32:
        raise ValueError(f"selector expects float32 scores, got {logits.dtype}")
    if block_table.ndim != 2:
        raise ValueError(
            f"block_table must be [num_reqs, pages], got {tuple(block_table.shape)}"
        )

    rows, seq_regions = logits.shape
    if request_indices is None:
        request_indices = torch.zeros((rows,), device=logits.device, dtype=torch.int32)
    out = torch.empty((rows, int(topk)), device=logits.device, dtype=torch.int32)
    _region_topk_pack_kernel[(rows,)](
        logits,
        lengths,
        q_positions,
        block_table,
        request_indices,
        out,
        seq_regions,
        logits.stride(0),
        shared_table_stride(block_table),
        out.stride(0),
        topk=int(topk),
        region_size=int(region_size),
        regions_per_page=int(regions_per_page),
        BLOCK_R=min(block_r, 256),
        BLOCK_TOPK=triton.next_power_of_2(int(topk)),
    )
    return out

# Preserve the first definition so later merged modules cannot silently
# overwrite a different metadata encoding.
_MERGED_PREFILL_REGION_VALID_SHIFT = PREFILL_REGION_VALID_SHIFT


# ============================================================================
# merged source: inference/dsa_meta.py
# ============================================================================
"""Stage 4: turn a region selection into the sparse-attention metadata.

This is the layer between the selector and the attention kernels, and it carries the two
facts about DSA that are easiest to get wrong because nothing crashes when you do.

**The query's own region never competes.** The selector's candidate range is
``[0, q_position // region_size)`` -- strictly *past* regions. The region containing the
query itself is appended afterwards, unconditionally. So a query always attends to at
least itself, and ``region_counts`` is one more than the number of history slots. Letting
the current region into the top-k instead would work for most rows and silently drop the
query's own context on the rows where it lost, which is exactly the kind of degradation
that shows up as slightly worse output rather than as a bug.

**There is no sliding window here.** ``sliding_window: 512`` in the checkpoint config
belongs to the sliding-attention layers, which are a different set of layers. The upstream
metadata converter does take a ``window`` argument -- and every DSA call site passes
``window=0``. When it is non-zero it *replaces* the single current-region append with the
full run of regions covering ``[q_position - window + 1, q_position]``, unioned with the
history; that generalisation is deliberately not implemented here, because implementing an
unused mode would leave it ungated.

The prefill and decode words are different, and so are the conventions around them:

============  ==============================  ================================
\\             prefill                         decode
============  ==============================  ================================
dtype         int32                           int64
word          ``phys | valid_tokens << 24``    ``start_token | phys << 32``
valid_tokens  packed                          derived from ``kv_seqlen``
dropped slot  left in place as ``-1``         compacted out
tail padding  ``-1``                          **zero**
order         ascending logical region        ascending packed word (physical)
============  ==============================  ================================

The zero padding on the decode side is the sharp edge: a zero word is a structurally valid
slot pointing at physical region 0 with a full complement of tokens. Nothing but
``region_counts`` distinguishes it from a real selection.
"""


import torch
import triton
import triton.language as tl


PREFILL_REGION_VALID_SHIFT = tl.constexpr(24)
DECODE_PHYS_SHIFT = tl.constexpr(32)
# Larger than any real packed word, so invalid slots sort to the tail.
DECODE_SORT_SENTINEL = tl.constexpr(1 << 62)


@triton.jit
def _append_current_region_kernel(
    history_ptr,
    q_positions_ptr,
    block_table_ptr,
    request_ptr,
    out_ptr,
    counts_ptr,
    stride_history_row,
    stride_table_row,
    stride_out_row,
    topk: tl.constexpr,
    region_size: tl.constexpr,
    regions_per_page: tl.constexpr,
    BLOCK_K: tl.constexpr,
) -> None:
    """Widen prefill history metadata to ``topk + 1`` slots and append the current region.

    The history is copied verbatim -- including its ``-1`` slots, which stay *in place*
    rather than being compacted, because prefill's ``region_counts`` is positional
    (``min(current_region, topk) + 1``) and not a count of live entries. The appended slot
    lands at exactly the index the selector stopped filling.

    The two stores are kept **disjoint** (``slots != history_count``) rather than letting
    the second overwrite the first. Writing the whole row and then patching one slot is a
    cross-thread race: the row store is a vector store spread over lanes, the patch is a
    scalar store from one lane, and nothing orders them. It loses roughly one row in a
    thousand, so it survives a small gate and fails a large one. The upstream kernel
    carries a comment about this exact hazard, which suggests it was found the same way.
    """
    row = tl.program_id(0)
    q_position = tl.load(q_positions_ptr + row).to(tl.int32)
    request = tl.load(request_ptr + row).to(tl.int32)
    current_region = tl.maximum(q_position // region_size, 0)
    history_count = tl.minimum(current_region, topk)

    slots = tl.arange(0, BLOCK_K)
    history = tl.load(
        history_ptr + row * stride_history_row + slots, mask=slots < topk, other=-1
    )
    tl.store(
        out_ptr + row * stride_out_row + slots,
        history,
        mask=(slots < topk + 1) & (slots != history_count),
    )

    page = tl.load(
        block_table_ptr + request * stride_table_row + current_region // regions_per_page
    )
    physical = page * regions_per_page + current_region % regions_per_page
    valid_tokens = tl.minimum(
        tl.maximum(q_position + 1 - current_region * region_size, 0), region_size
    )
    packed = physical | (valid_tokens << PREFILL_REGION_VALID_SHIFT)
    tl.store(
        out_ptr + row * stride_out_row + history_count,
        tl.where(page >= 0, packed, -1),
    )
    tl.store(counts_ptr + row, history_count + 1)


@triton.jit
def _decode_pack_kernel(
    ids_ptr,
    seqlens_ptr,
    block_table_ptr,
    request_ptr,
    out_ptr,
    counts_ptr,
    stride_ids_row,
    stride_table_row,
    stride_out_row,
    topk: tl.constexpr,
    region_size: tl.constexpr,
    regions_per_page: tl.constexpr,
    BLOCK_K: tl.constexpr,
) -> None:
    """Pack selected logical regions into the int64 decode word.

    One row fits in one tile, so the compaction is a sort rather than a scatter: invalid
    slots are mapped to a sentinel above every real word, and sorting by the packed value
    both drops them to the tail and produces the physical-major order the upstream
    converter emits. The current region is then written at the surviving count -- after the
    sort, so it stays last, which is also where the reference puts it.
    """
    row = tl.program_id(0)
    request = tl.load(request_ptr + row).to(tl.int32)
    kv_seqlen = tl.load(seqlens_ptr + request).to(tl.int32)
    valid_regions = (kv_seqlen + region_size - 1) // region_size

    slots = tl.arange(0, BLOCK_K)
    ids = tl.load(ids_ptr + row * stride_ids_row + slots, mask=slots < topk, other=-1)

    live = (slots < topk) & (ids >= 0) & (ids < valid_regions)
    safe_ids = tl.where(live, ids, 0)
    pages = tl.load(
        block_table_ptr + request * stride_table_row + safe_ids // regions_per_page,
        mask=live,
        other=-1,
    )
    live = live & (pages >= 0)

    physical = pages * regions_per_page + safe_ids % regions_per_page
    packed = (safe_ids * region_size).to(tl.int64) | (
        physical.to(tl.int64) << DECODE_PHYS_SHIFT
    )
    # Compaction without tl.sort: sorting the int64 words needs ~6.4 Mbit of
    # unified buffer at BLOCK_K = 512, which the A5 does not have. The selector
    # already emits ids in ascending region order and the decode attention kernel
    # consumes slots order-independently, so a cumsum scatter is equivalent.
    count = tl.sum(live.to(tl.int32))
    tl.store(
        out_ptr + row * stride_out_row + slots,
        0,
        mask=slots < topk + 1,
    )
    positions = tl.cumsum(live.to(tl.int32), axis=0) - 1
    tl.store(
        out_ptr + row * stride_out_row + positions,
        packed,
        mask=live,
    )

    current_region = tl.maximum((kv_seqlen - 1) // region_size, 0)
    current_page = tl.load(
        block_table_ptr + request * stride_table_row + current_region // regions_per_page
    )
    current_live = (current_region < valid_regions) & (current_page >= 0)
    current_physical = (
        current_page * regions_per_page + current_region % regions_per_page
    )
    current_packed = (current_region * region_size).to(tl.int64) | (
        current_physical.to(tl.int64) << DECODE_PHYS_SHIFT
    )
    # Written unconditionally so the slot is defined even when the current region is
    # unmapped, in which case it becomes ordinary zero padding.
    tl.store(
        out_ptr + row * stride_out_row + count,
        tl.where(current_live, current_packed, 0),
    )
    tl.store(counts_ptr + row, count + current_live.to(tl.int32))


def prefill_sparse_meta(
    logits: torch.Tensor,
    q_positions: torch.Tensor,
    block_table: torch.Tensor,
    request_indices: torch.Tensor | None = None,
    *,
    topk: int,
    region_size: int = REGION_BLOCK_SIZE,
    regions_per_page: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select regions and build prefill sparse metadata.

    Args:
        logits: ``[rows, seq_regions]`` float32 indexer scores, which may be signed,
            group-major (``rows = num_kv_groups * total_q``). The scores are weighted
            sums of per-head ReLU terms and unconstrained signed indexer weights.
        q_positions: ``[rows]`` int32, absolute token position of each row's query.
        block_table: ``[num_reqs, pages]`` int32, logical page -> physical page, one row
            per request. ``-1`` marks an unmapped page, which yields a ``-1`` slot rather
            than an error.
        request_indices: ``[rows]`` int32, which block-table row each score row uses.
            Defaults to all-zero, i.e. every row shares one table.

    Returns:
        ``(packed [rows, topk + 1] int32, region_counts [rows] int32)``, ready for
        :func:`inference.sparse_attn.sparse_attention_prefill`.

    The history length is derived here as ``q_position // region_size`` rather than taken
    from the caller, because getting it wrong by one is the difference between the current
    region being force-included and it competing for a slot.
    """
    rows = int(logits.shape[0])
    if request_indices is None:
        request_indices = torch.zeros((rows,), device=logits.device, dtype=torch.int32)
    history_lengths = torch.div(
        q_positions.to(torch.int32), region_size, rounding_mode="floor"
    ).contiguous()

    history = region_topk_pack(
        logits,
        history_lengths,
        q_positions,
        block_table,
        request_indices,
        topk=topk,
        region_size=region_size,
        regions_per_page=regions_per_page,
    )

    packed = torch.empty((rows, int(topk) + 1), device=logits.device, dtype=torch.int32)
    counts = torch.empty((rows,), device=logits.device, dtype=torch.int32)
    _append_current_region_kernel[(rows,)](
        history,
        q_positions,
        block_table,
        request_indices,
        packed,
        counts,
        history.stride(0),
        shared_table_stride(block_table),
        packed.stride(0),
        topk=int(topk),
        region_size=int(region_size),
        regions_per_page=int(regions_per_page),
        BLOCK_K=triton.next_power_of_2(int(topk) + 1),
    )
    return packed, counts


def decode_sparse_meta(
    logits: torch.Tensor,
    kv_seqlens: torch.Tensor,
    block_table: torch.Tensor,
    request_indices: torch.Tensor | None = None,
    *,
    topk: int,
    region_size: int = REGION_BLOCK_SIZE,
    regions_per_page: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select regions and build decode sparse metadata.

    Args:
        logits: ``[num_kv_groups * num_reqs, seq_regions]`` float32 indexer scores,
            which may be signed, group-major. The selector orders them numerically rather
            than by the magnitude encoded in their IEEE-754 bit pattern.
        kv_seqlens: ``[num_reqs]`` int32, context length *including* the current token.
        block_table: ``[num_reqs, pages]`` int32, logical page -> physical page.
        request_indices: ``[rows]`` int32. Defaults to the group-major layout implied by
            the row count, ``arange(num_reqs)`` tiled once per KV group.

    Returns:
        ``(packed [rows, topk + 1] int64, region_counts [rows] int32)``, ready for
        :func:`inference.sparse_attn.sparse_attention_decode`.

    ``region_counts`` here is a count of *surviving* entries, so it varies with how much of
    the block table is mapped. The tail beyond it is zero, and a zero word is not
    self-identifying -- the count is the only bound.
    """
    rows, _ = logits.shape
    num_reqs = int(kv_seqlens.shape[0])
    if rows % num_reqs:
        raise ValueError(f"{rows} score rows do not divide into {num_reqs} requests")

    if request_indices is None:
        request_indices = (
            torch.arange(num_reqs, device=logits.device, dtype=torch.int32)
            .repeat(rows // num_reqs)
            .contiguous()
        )
    # Decode's query position is the last token, so the history is everything strictly
    # before the current region.
    history_lengths = torch.div(
        (kv_seqlens.to(torch.int32) - 1).clamp_min(0), region_size, rounding_mode="floor"
    )[request_indices.to(torch.long)].contiguous()

    ids = region_topk_ids(logits, history_lengths, topk=topk)

    packed = torch.empty((rows, int(topk) + 1), device=logits.device, dtype=torch.int64)
    counts = torch.empty((rows,), device=logits.device, dtype=torch.int32)
    _decode_pack_kernel[(rows,)](
        ids,
        kv_seqlens,
        block_table,
        request_indices,
        packed,
        counts,
        ids.stride(0),
        shared_table_stride(block_table),
        packed.stride(0),
        topk=int(topk),
        region_size=int(region_size),
        regions_per_page=int(regions_per_page),
        BLOCK_K=triton.next_power_of_2(int(topk) + 1),
    )
    return packed, counts

assert (
    PREFILL_REGION_VALID_SHIFT.value
    == _MERGED_PREFILL_REGION_VALID_SHIFT.value
    == 24
), "merged DSA modules disagree on PREFILL_REGION_VALID_SHIFT"
del _MERGED_PREFILL_REGION_VALID_SHIFT
_MERGED_DECODE_PHYS_SHIFT = DECODE_PHYS_SHIFT


# ============================================================================
# merged source: inference/sparse_attn.py
# ============================================================================
"""Triton sparse attention over the selected regions -- the DSA forward itself.

Separate from ``dsa_kernels`` because this is the only stage that touches the main KV
cache and the main 192-dim heads; everything there operates on the 256-dim indexer proxy.

Semantics are pinned against ``token_wise_flash_attn_prefill_sm90_gqa_func`` and its
reference in ``optimus_jit/tests/test_sparse_gqa_steptron_fwd_reference.py``. Three
details of that contract are worth stating up front because none is guessable:

* Each query row carries its **own** region list. Queries cannot share a K tile the way
  dense attention does, so the natural unit of work is one (query, KV group) pair rather
  than a query tile.
* There is **no causal mask** in the attention itself. Causality is already baked into the
  packed metadata: ``valid_tokens`` says how much of each region the query may see, and
  the selector only ever offered causally visible regions.
* Softmax runs over the gathered token set only, and an empty selection yields a zero
  output with ``lse = -inf`` rather than a NaN.
"""


import torch
import triton
import triton.language as tl

REGION_VALID_SHIFT = tl.constexpr(24)
REGION_ID_MASK = tl.constexpr((1 << 24) - 1)
DECODE_PHYS_SHIFT = tl.constexpr(32)
DECODE_LOW_MASK = tl.constexpr((1 << 32) - 1)
NEGATIVE_INFINITY = tl.constexpr(float("-inf"))


@triton.jit
def _sparse_attn_prefill_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    packed_ptr,
    counts_ptr,
    out_ptr,
    lse_ptr,
    stride_q_token,
    stride_q_head,
    stride_k_token,
    stride_k_group,
    stride_v_token,
    stride_v_group,
    stride_packed_row,
    stride_out_token,
    stride_out_head,
    stride_lse_token,
    softmax_scale,
    total_q,
    topk: tl.constexpr,
    heads_per_group: tl.constexpr,
    head_dim: tl.constexpr,
    region_size: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_R: tl.constexpr,
) -> None:
    """Attention for one query against one KV group's selected regions.

    Online softmax over region tiles. The accumulator is per (head, dim) and the running
    max is per head, exactly as in dense flash attention -- the only difference is that
    the K/V rows come from a gather driven by the packed region list instead of a
    contiguous range.
    """
    query = tl.program_id(0)
    group = tl.program_id(1)
    row = group * total_q + query
    count = tl.load(counts_ptr + row).to(tl.int32)

    heads = tl.arange(0, BLOCK_H)
    dims = tl.arange(0, BLOCK_D)
    head_ok = heads < heads_per_group
    # BLOCK_D == head_dim by construction, so a dim mask is trivially all-true. Leaving
    # it on the K/V loads only predicated a copy that is always full width.
    tile_ok = head_ok[:, None]

    queries = tl.load(
        q_ptr
        + query * stride_q_token
        + (group * heads_per_group + heads)[:, None] * stride_q_head
        + dims[None, :],
        mask=tile_ok,
        other=0.0,
    )

    running_max = tl.full((BLOCK_H,), NEGATIVE_INFINITY, dtype=tl.float32)
    running_sum = tl.zeros((BLOCK_H,), dtype=tl.float32)
    accumulator = tl.zeros((BLOCK_H, BLOCK_D), dtype=tl.float32)

    # Keep region and in-region offsets as separate axes. Flattening first and recovering
    # them with vector // and % scalarizes address generation on Ascend and also reloads
    # each packed metadata word region_size times. BLOCK_R remains small enough that the
    # expanded K/V tile (BLOCK_R * region_size tokens) fits in UB.
    offsets_in_region = tl.arange(0, region_size)

    # Bound the loop by the row's own count rather than topk. Padding-only tiles used to
    # run a full gather and both dots against a tile that is entirely -inf. Set this back
    # to `topk` to isolate the fused-gather change on its own.
    upper = tl.minimum(count, topk)

    for base in tl.range(0, upper, BLOCK_R):
        slots = base + tl.arange(0, BLOCK_R)
        slot_ok = slots < count
        meta = tl.load(
            packed_ptr + row * stride_packed_row + slots, mask=slot_ok, other=-1
        )
        # Signed int32 metadata: a negative sentinel has valid < 0; an
        # invisible region has valid == 0. The load's other=-1 handles slot_ok.
        # Keep only this small [BLOCK_R] vector live across the K/V aggregation.
        valid = meta >> REGION_VALID_SHIFT
        # Treat every selected region as a small KV page. Loading all regions through one
        # flattened gather gives Ascend many short, discontinuous GM accesses, and makes
        # the row axis data-dependent so the backend has to materialise a full
        # [BLOCK_R * region_size, BLOCK_D] address tensor. Instead, load one contiguous
        # region/page at a time and aggregate the slices in UB, as in
        # paged_prefill_page_aggregation_kernel.
        #
        # K and V share the same slot metadata, the same token indices and the same mask,
        # so both are gathered in a single pass: one scalar metadata read and one address
        # computation per page instead of two, and V's MTE2 traffic now sits alongside
        # K's where it can overlap the QK nd2nz/mmad/fixpipe chain instead of queueing
        # behind the softmax. The cost is holding both tiles in UB across the QK dot.
        keys = tl.zeros(
            (BLOCK_R * region_size, BLOCK_D), dtype=k_ptr.dtype.element_ty
        )
        values = tl.zeros(
            (BLOCK_R * region_size, BLOCK_D), dtype=v_ptr.dtype.element_ty
        )
        for page_iter in tl.extra.cann.extension.parallel(0, BLOCK_R):
            # Scalar pointer -> scalar load; decode the fields inside this loop.
            page_slot = base + page_iter
            page_slot_ok = (page_slot < count) & (page_slot < topk)
            page_meta = tl.load(
                packed_ptr + row * stride_packed_row + page_slot,
                mask=page_slot_ok,
                other=-1,
            )
            page_valid = page_meta >> REGION_VALID_SHIFT
            page_live = page_slot_ok & (page_meta >= 0) & (page_valid > 0)
            # Clamped on dead slots. Left unclamped this is (-1 & 0xFFFFFF) == 16777215,
            # which is masked off but still carries a ~1.3e8 token index through the
            # address arithmetic.
            page_physical = tl.where(page_live, page_meta & REGION_ID_MASK, 0)
            page_tokens = page_physical * region_size + offsets_in_region
            page_token_ok = offsets_in_region < tl.where(page_live, page_valid, 0)

            key_slice = tl.load(
                k_ptr
                + page_tokens[:, None] * stride_k_token
                + group * stride_k_group
                + dims[None, :],
                mask=page_token_ok[:, None],
                other=0.0,
            )
            keys = tl.extra.cann.extension.insert_slice(
                keys,
                key_slice,
                offsets=(page_iter * region_size, 0),
                sizes=(region_size, BLOCK_D),
                strides=(1, 1),
            )

            value_slice = tl.load(
                v_ptr
                + page_tokens[:, None] * stride_v_token
                + group * stride_v_group
                + dims[None, :],
                mask=page_token_ok[:, None],
                other=0.0,
            )
            values = tl.extra.cann.extension.insert_slice(
                values,
                value_slice,
                offsets=(page_iter * region_size, 0),
                sizes=(region_size, BLOCK_D),
                strides=(1, 1),
            )

        scores = tl.dot(queries, tl.trans(keys), out_dtype=tl.float32) * softmax_scale
        # offset < valid -> 0; otherwise -> -1, using signed arithmetic shift.
        invalid_sign = ((valid - 1)[:, None] - offsets_in_region[None, :]) >> 31
        # int32(-1) << 23 is 0xff800000: the FP32 -inf bit pattern.
        # bitcast=True is essential; a numeric cast would give -8388608.0.
        token_bias = tl.reshape(
            (invalid_sign << 23).to(tl.float32, bitcast=True),
            (BLOCK_R * region_size,),
        )
        # Requires finite Q and scale so masked K rows yield finite raw scores.
        # Heads are independent; padding heads are discarded by the store mask.
        scores = scores + token_bias[None, :]

        tile_max = tl.maximum(running_max, tl.max(scores, axis=1, propagate_nan=True), propagate_nan=tl.PropagateNan.ALL)
        # A tile can be entirely masked -- every slot exhausted, or a row with no
        # selection at all. Then both maxima are -inf and ``m - m`` is NaN, which would
        # poison the accumulator for the rest of the loop. Substituting a finite shift
        # keeps the arithmetic well defined and leaves the running state untouched.
        empty = tile_max == NEGATIVE_INFINITY
        shift = tl.where(empty, 0.0, tile_max)
        rescale = tl.where(empty, 1.0, tl.exp(running_max - shift))

        probabilities = tl.exp(scores - shift[:, None])
        running_sum = running_sum * rescale + tl.sum(probabilities, axis=1)
        accumulator = accumulator * rescale[:, None] + tl.dot(
            probabilities.to(values.dtype), values, out_dtype=tl.float32
        )
        running_max = tile_max

    selected = running_sum > 0
    output = tl.where(selected[:, None], accumulator / running_sum[:, None], 0.0)
    tl.store(
        out_ptr
        + query * stride_out_token
        + (group * heads_per_group + heads)[:, None] * stride_out_head
        + dims[None, :],
        output.to(out_ptr.dtype.element_ty),
        mask=tile_ok,
    )
    tl.store(
        lse_ptr + query * stride_lse_token + group * heads_per_group + heads,
        tl.where(selected, running_max + tl.log(running_sum), NEGATIVE_INFINITY),
        mask=head_ok,
    )


def sparse_attention_prefill(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    packed_regions: torch.Tensor,
    region_counts: torch.Tensor,
    *,
    num_kv_groups: int,
    region_size: int = 8,
    softmax_scale: float | None = None,
    block_regions: int | None = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sparse attention over per-query selected regions.

    Args:
        query: ``[total_q, num_heads, head_dim]``.
        key_cache, value_cache: ``[tokens, num_kv_groups, head_dim]``, i.e. the paged
            cache already flattened across pages. Flattening is safe because a region's
            tokens are contiguous in the physical token space by construction.
        packed_regions: ``[num_kv_groups * total_q, topk]`` int32 from the selector,
            each slot ``phys_region | (valid_tokens << 24)``, ``-1`` padding the tail.
        region_counts: ``[num_kv_groups * total_q]`` int32, how many slots to read. This
            truncates *positionally* -- it does not re-select by score.
        block_regions: Deprecated compatibility option. When provided, it is converted
            to ``block_regions * region_size`` tokens. Do not set it together with a
            non-default ``block_tokens`` value.

    Returns:
        ``(out, lse)`` shaped ``[total_q, num_heads, head_dim]`` and
        ``[total_q, num_heads]``. Rows with no selected region get a zero output and
        ``lse = -inf``.

    The caller must guarantee the selected regions are distinct. Our selector does (the
    stage 3 gate checks it), but a union-with-sliding-window path would not, and duplicate
    regions would silently double-weight those tokens in the softmax rather than fail.

    ``key_cache`` and ``value_cache`` are indexed only through ``stride_*_token`` and
    ``stride_*_group``, so a group-major physical layout can be passed as a transposed
    view with no kernel change. At head_dim 128 that turns each region from eight 256 B
    strided rows into a single contiguous 2 KB burst.
    """
    total_q, num_heads, head_dim = query.shape
    if num_heads % num_kv_groups:
        raise ValueError(f"{num_heads} heads do not divide into {num_kv_groups} groups")
    if packed_regions.shape[0] != num_kv_groups * total_q:
        raise ValueError(
            f"expected {num_kv_groups * total_q} metadata rows, got {packed_regions.shape[0]}"
        )
    if key_cache.shape != value_cache.shape:
        raise ValueError(
            f"key/value cache shape mismatch: {tuple(key_cache.shape)} vs {tuple(value_cache.shape)}"
        )

    heads_per_group = num_heads // num_kv_groups
    topk = int(packed_regions.shape[1])
    scale = softmax_scale if softmax_scale is not None else head_dim**-0.5

    block_regions = min(topk, 16)

    out = torch.empty_like(query)
    lse = torch.empty((total_q, num_heads), device=query.device, dtype=torch.float32)
    _sparse_attn_prefill_kernel[(total_q, num_kv_groups)](
        query,
        key_cache,
        value_cache,
        packed_regions,
        region_counts,
        out,
        lse,
        query.stride(0),
        query.stride(1),
        key_cache.stride(0),
        key_cache.stride(1),
        value_cache.stride(0),
        value_cache.stride(1),
        packed_regions.stride(0),
        out.stride(0),
        out.stride(1),
        lse.stride(0),
        scale,
        total_q,
        topk=topk,
        heads_per_group=heads_per_group,
        head_dim=head_dim,
        region_size=region_size,
        BLOCK_H=triton.next_power_of_2(heads_per_group),
        BLOCK_D=head_dim,
        BLOCK_R=block_regions,
        enable_dynamic_cv_pipeline=False,
        set_workspace_multibuffer=1,
        enable_preload=False,
    )
    return out, lse


@triton.jit
def _sparse_attn_decode_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    packed_ptr,
    counts_ptr,
    seqlens_ptr,
    out_ptr,
    lse_ptr,
    stride_q_token,
    stride_q_head,
    stride_k_token,
    stride_k_group,
    stride_v_token,
    stride_v_group,
    stride_packed_row,
    stride_out_split,
    stride_out_token,
    stride_out_head,
    stride_lse_split,
    stride_lse_token,
    softmax_scale,
    num_reqs,
    topk: tl.constexpr,
    heads_per_group: tl.constexpr,
    head_dim: tl.constexpr,
    region_size: tl.constexpr,
    slots_per_split: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_R: tl.constexpr,
) -> None:
    """One decode query against one KV group's selected regions, over one slot window.

    Structurally the same online softmax as prefill, but the metadata contract differs in
    the one way that matters: decode packs ``start_token | (phys_region << 32)`` in int64
    and does **not** carry ``valid_tokens``. The visible length is recomputed here as
    ``clamp(kv_seqlen - start_token, 0, region_size)``, which is why the logical start
    token has to be in the word at all -- the physical region alone cannot say how much of
    the last region exists yet.

    That also disposes of both padding sentinels without a special case. ``-1`` and
    ``0x7FFF_FFFF_FFFF_FFFF`` both leave ``0xFFFF_FFFF`` in the low word, so
    ``kv_seqlen - start_token`` is hugely negative and the slot clamps to zero tokens.
    """
    request = tl.program_id(0)
    group = tl.program_id(1)
    split = tl.program_id(2)
    row = group * num_reqs + request
    count = tl.load(counts_ptr + row).to(tl.int32)
    kv_seqlen = tl.load(seqlens_ptr + request).to(tl.int64)

    heads = tl.arange(0, BLOCK_H)
    dims = tl.arange(0, BLOCK_D)
    head_ok = heads < heads_per_group
    dim_ok = dims < head_dim
    tile_ok = head_ok[:, None] & dim_ok[None, :]

    queries = tl.load(
        q_ptr
        + request * stride_q_token
        + (group * heads_per_group + heads)[:, None] * stride_q_head
        + dims[None, :],
        mask=tile_ok,
        other=0.0,
    )

    running_max = tl.full((BLOCK_H,), NEGATIVE_INFINITY, dtype=tl.float32)
    running_sum = tl.zeros((BLOCK_H,), dtype=tl.float32)
    accumulator = tl.zeros((BLOCK_H, BLOCK_D), dtype=tl.float32)

    offsets_in_region = tl.arange(0, region_size)
    window_begin = split * slots_per_split
    for base in tl.range(window_begin, window_begin + slots_per_split, BLOCK_R):
        slots = base + tl.arange(0, BLOCK_R)
        slot_ok = (slots < count) & (slots < topk)
        meta = tl.load(
            packed_ptr + row * stride_packed_row + slots, mask=slot_ok, other=-1
        )
        physical = (meta >> DECODE_PHYS_SHIFT) & DECODE_LOW_MASK
        start_token = meta & DECODE_LOW_MASK
        valid = tl.minimum(tl.maximum(kv_seqlen - start_token, 0), region_size)
        live = slot_ok & (valid > 0)

        # ``physical`` is a sentinel-sized integer on dead slots, so it must be
        # neutralised before it becomes an address -- masking the load is not enough,
        # since the pointer arithmetic itself would overflow.
        safe_physical = tl.where(live, physical, 0)
        tokens = tl.reshape(
            safe_physical[:, None] * region_size + offsets_in_region[None, :],
            (BLOCK_R * region_size,),
        )
        token_ok = tl.reshape(
            live[:, None] & (offsets_in_region[None, :] < valid[:, None]),
            (BLOCK_R * region_size,),
        )

        keys = tl.load(
            k_ptr + tokens[:, None] * stride_k_token + group * stride_k_group + dims[None, :],
            mask=token_ok[:, None] & dim_ok[None, :],
            other=0.0,
        )
        values = tl.load(
            v_ptr + tokens[:, None] * stride_v_token + group * stride_v_group + dims[None, :],
            mask=token_ok[:, None] & dim_ok[None, :],
            other=0.0,
        )

        scores = tl.dot(queries, tl.trans(keys), out_dtype=tl.float32) * softmax_scale
        scores = tl.where(token_ok[None, :] & head_ok[:, None], scores, NEGATIVE_INFINITY)

        tile_max = tl.maximum(running_max, tl.max(scores, axis=1))
        empty = tile_max == NEGATIVE_INFINITY
        shift = tl.where(empty, 0.0, tile_max)
        rescale = tl.where(empty, 1.0, tl.exp(running_max - shift))

        probabilities = tl.exp(scores - shift[:, None])
        running_sum = running_sum * rescale + tl.sum(probabilities, axis=1)
        accumulator = accumulator * rescale[:, None] + tl.dot(
            probabilities.to(values.dtype), values, out_dtype=tl.float32
        )
        running_max = tile_max

    selected = running_sum > 0
    output = tl.where(selected[:, None], accumulator / running_sum[:, None], 0.0)
    tl.store(
        out_ptr
        + split * stride_out_split
        + request * stride_out_token
        + (group * heads_per_group + heads)[:, None] * stride_out_head
        + dims[None, :],
        output.to(out_ptr.dtype.element_ty),
        mask=tile_ok,
    )
    tl.store(
        lse_ptr
        + split * stride_lse_split
        + request * stride_lse_token
        + group * heads_per_group
        + heads,
        tl.where(selected, running_max + tl.log(running_sum), NEGATIVE_INFINITY),
        mask=head_ok,
    )


@triton.jit
def _merge_split_states_kernel(
    partial_out_ptr,
    partial_lse_ptr,
    out_ptr,
    lse_ptr,
    stride_po_split,
    stride_po_token,
    stride_po_head,
    stride_pl_split,
    stride_pl_token,
    stride_out_token,
    stride_out_head,
    stride_lse_token,
    num_splits,
    head_dim: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
) -> None:
    """Combine per-split attention states into one, weighting by the partial lse.

    Each split produced an output already divided by its own softmax denominator, so the
    combination is a softmax over the partial log-sum-exps rather than a plain sum:
    ``out = sum_s softmax(lse)_s * out_s`` and ``lse = logsumexp_s(lse_s)``. A split that
    covered no live slot reports ``lse = -inf`` and drops out with weight zero; if *every*
    split does, the row selected nothing and must come back as a zero output with
    ``lse = -inf`` rather than ``0/0``.
    """
    token = tl.program_id(0)
    head = tl.program_id(1)

    splits = tl.arange(0, BLOCK_S)
    dims = tl.arange(0, BLOCK_D)
    split_ok = splits < num_splits
    dim_ok = dims < head_dim

    partial_lse = tl.load(
        partial_lse_ptr + splits * stride_pl_split + token * stride_pl_token + head,
        mask=split_ok,
        other=NEGATIVE_INFINITY,
    )
    peak = tl.max(partial_lse)
    empty = peak == NEGATIVE_INFINITY
    shift = tl.where(empty, 0.0, peak)
    weights = tl.where(split_ok, tl.exp(partial_lse - shift), 0.0)
    total = tl.sum(weights)

    states = tl.load(
        partial_out_ptr
        + splits[:, None] * stride_po_split
        + token * stride_po_token
        + head * stride_po_head
        + dims[None, :],
        mask=split_ok[:, None] & dim_ok[None, :],
        other=0.0,
    ).to(tl.float32)
    merged = tl.sum(states * weights[:, None], axis=0) / tl.maximum(total, 1.0e-20)

    tl.store(
        out_ptr + token * stride_out_token + head * stride_out_head + dims,
        merged.to(out_ptr.dtype.element_ty),
        mask=dim_ok,
    )
    tl.store(
        lse_ptr + token * stride_lse_token + head,
        tl.where(empty, NEGATIVE_INFINITY, shift + tl.log(total)),
    )


def merge_split_states(
    partial_out: torch.Tensor, partial_lse: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce ``[splits, tokens, heads, dim]`` partial states to ``[tokens, heads, dim]``.

    Semantics follow ``merge_variable_split_nat_lse_states_sm90_gqa`` -- "nat" being the
    natural-log lse the sparse kernels emit. The upstream kernel additionally supports a
    *variable* split plan, where the first ``n_split4`` requests are split four ways, the
    next ``n_split2`` twice, and the rest not at all, packed contiguously. That layout
    exists to fill the SMs at small batch and carries no semantics, so the release uses a
    uniform split and pays a little occupancy for a much simpler contract.
    """
    num_splits, tokens, num_heads, head_dim = partial_out.shape
    out = torch.empty((tokens, num_heads, head_dim), device=partial_out.device, dtype=partial_out.dtype)
    lse = torch.empty((tokens, num_heads), device=partial_out.device, dtype=torch.float32)
    _merge_split_states_kernel[(tokens, num_heads)](
        partial_out,
        partial_lse,
        out,
        lse,
        partial_out.stride(0),
        partial_out.stride(1),
        partial_out.stride(2),
        partial_lse.stride(0),
        partial_lse.stride(1),
        out.stride(0),
        out.stride(1),
        lse.stride(0),
        num_splits,
        head_dim=head_dim,
        BLOCK_S=triton.next_power_of_2(num_splits),
        BLOCK_D=triton.next_power_of_2(head_dim),
    )
    return out, lse


def sparse_attention_decode(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    packed_regions: torch.Tensor,
    region_counts: torch.Tensor,
    kv_seqlens: torch.Tensor,
    *,
    num_kv_groups: int,
    region_size: int = 8,
    softmax_scale: float | None = None,
    num_splits: int = 1,
    block_regions: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sparse attention for one query token per request.

    Args:
        query: ``[num_reqs, num_heads, head_dim]``.
        key_cache, value_cache: ``[tokens, num_kv_groups, head_dim]``, pages flattened.
        packed_regions: ``[num_kv_groups * num_reqs, topk]`` **int64**, each slot
            ``start_token | (phys_region << 32)``. Note this is a different packing from
            prefill's int32 ``phys_region | (valid_tokens << 24)``: decode carries the
            logical start token and derives the visible length from ``kv_seqlens``.
        region_counts: ``[num_kv_groups * num_reqs]`` int32, slots to read per row.
        kv_seqlens: ``[num_reqs]`` int32, context length including the current token.
        num_splits: split the slot window this many ways and return partial states. With
            ``num_splits == 1`` the returned tensors are already the final result;
            otherwise pass them to :func:`merge_split_states`.

    Returns:
        ``num_splits == 1``: ``(out [num_reqs, num_heads, head_dim], lse [num_reqs, num_heads])``.
        Otherwise the same with a leading ``num_splits`` axis.

    As in prefill, the selected regions must be distinct. The upstream decode reference is
    stricter about this than the prefill one: it concatenates the gathered spans with no
    dedup at all, so a duplicated region would be counted twice in the softmax.
    """
    num_reqs, num_heads, head_dim = query.shape
    if num_heads % num_kv_groups:
        raise ValueError(f"{num_heads} heads do not divide into {num_kv_groups} groups")
    if packed_regions.shape[0] != num_kv_groups * num_reqs:
        raise ValueError(
            f"expected {num_kv_groups * num_reqs} metadata rows, got {packed_regions.shape[0]}"
        )
    if packed_regions.dtype != torch.int64:
        raise ValueError(f"decode metadata must be int64, got {packed_regions.dtype}")
    if kv_seqlens.shape[0] != num_reqs:
        raise ValueError(f"expected {num_reqs} kv_seqlens, got {kv_seqlens.shape[0]}")

    heads_per_group = num_heads // num_kv_groups
    topk = int(packed_regions.shape[1])
    scale = softmax_scale if softmax_scale is not None else head_dim**-0.5
    # Every split walks the same number of slots so the window bound is a constexpr; the
    # tail split simply finds every slot masked out.
    slots_per_split = -(-topk // num_splits)

    out = torch.empty((num_splits, num_reqs, num_heads, head_dim), device=query.device, dtype=query.dtype)
    lse = torch.empty((num_splits, num_reqs, num_heads), device=query.device, dtype=torch.float32)
    _sparse_attn_decode_kernel[(num_reqs, num_kv_groups, num_splits)](
        query,
        key_cache,
        value_cache,
        packed_regions,
        region_counts,
        kv_seqlens,
        out,
        lse,
        query.stride(0),
        query.stride(1),
        key_cache.stride(0),
        key_cache.stride(1),
        value_cache.stride(0),
        value_cache.stride(1),
        packed_regions.stride(0),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        lse.stride(0),
        lse.stride(1),
        scale,
        num_reqs,
        topk=topk,
        heads_per_group=heads_per_group,
        head_dim=head_dim,
        region_size=region_size,
        slots_per_split=slots_per_split,
        BLOCK_H=max(16, triton.next_power_of_2(heads_per_group)),
        BLOCK_D=triton.next_power_of_2(head_dim),
        BLOCK_R=min(block_regions, triton.next_power_of_2(slots_per_split)),
        num_stages=1,
    )
    if num_splits == 1:
        return out[0], lse[0]
    return out, lse

assert (
    DECODE_PHYS_SHIFT.value == _MERGED_DECODE_PHYS_SHIFT.value == 32
), "merged DSA modules disagree on DECODE_PHYS_SHIFT"
del _MERGED_DECODE_PHYS_SHIFT


# ============================================================================
# merged source: inference/dsa_attention.py
# ============================================================================
"""DSA plumbing: the cache that persists a selection's inputs, and the scoring that uses it.

The kernels in this package are stateless -- each one takes the whole context as a tensor.
Generation is not: a decode step sees one token and has to score it against a context it no
longer holds. This module is the state that closes that gap, and it holds exactly three
things per DSA layer.

**The main K/V, in physical token order.** Written where the block table says, read by the
attention kernels. Nothing here is logical.

**One compressed summary per *completed* region, in logical order.** This is what the
indexer scores against, and "completed" is the load-bearing word. A region's summary is a
softmax-weighted mean over its tokens, so a summary computed from a partial region is a
different vector than the one computed from the full region -- storing the partial value
would silently poison every later selection. It is safe to only ever store completed
summaries because the selector's candidate range stops at ``q_position // region_size``:
every region it can choose ended before the query's own region began, so every candidate is
complete by construction. The query's own region is force-appended by the metadata builder
and never scored.

**The tokens of the region currently being filled.** At most ``region_size - 1`` of them.
When the region fills, its summary is computed with the same kernel the prefill path uses,
from the same inputs -- so the streaming result is not merely close to the batch result, it
is the same call. The alternative, carrying a running fp32 accumulator, is also exact but
has to reproduce the reduction order to stay that way.

Batching is by block table: request ``b``'s logical page ``p`` is physical page
``b * pages_per_request + p``, which makes the flat token space a reshaped
``[batch, max_tokens, ...]`` and the region space a reshaped ``[batch, max_regions, ...]``.
Scoring is looped over requests because each one scores against a different key set; the
attention itself is a single batched launch.
"""


import math
from dataclasses import dataclass

import torch



def build_rope_cache(
    *,
    rotary_span: int,
    theta: float,
    max_position: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    scaling: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cos/sin tables for a NEOX-style partial rotation.

    Args:
        rotary_span: how many of the head's dimensions rotate. The returned tables are
            ``[max_position, rotary_span // 2]`` -- *half* the span, because each frequency
            drives the pair ``(d, d + rotary_span // 2)``. The kernels take that halved
            width as their ``rotary_dim``, which is easy to misread as the span itself.
        scaling: the ``rope_scaling`` block, or ``None``. Only ``llama3`` is implemented,
            because it is the only type step4's config asks for; an unrecognised type
            raises rather than silently falling back to unscaled, which would look like a
            long-context quality regression rather than a bug.
    """
    if rotary_span % 2:
        raise ValueError(f"rotary span must be even, got {rotary_span}")
    half = rotary_span // 2
    # Match vLLM's RotaryEmbedding cache construction exactly.  In particular this must
    # stay float32 throughout, even though computing the table in float64 looks more
    # accurate in isolation.  The deployed model rounds the float32 frequencies/trig
    # results to the requested cache dtype; using float64 changes BF16 table entries as
    # early as position 24 (and therefore changes Q/K before the first long-context
    # feature is involved).
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, rotary_span, 2, device=device, dtype=torch.float32) / rotary_span)
    )
    if scaling:
        inv_freq = _apply_llama3_scaling(inv_freq, scaling)

    positions = torch.arange(max_position, device=device, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    assert freqs.shape[1] == half
    return freqs.cos().to(dtype).contiguous(), freqs.sin().to(dtype).contiguous()


def _apply_llama3_scaling(inv_freq: torch.Tensor, scaling: dict) -> torch.Tensor:
    rope_type = str(scaling.get("rope_type", scaling.get("type", ""))).lower()
    if rope_type != "llama3":
        raise NotImplementedError(f"unsupported rope_scaling type {rope_type!r}")

    factor = float(scaling["factor"])
    low_freq_factor = float(scaling["low_freq_factor"])
    high_freq_factor = float(scaling["high_freq_factor"])
    original_max = float(scaling["original_max_position_embeddings"])

    low_wavelen = original_max / low_freq_factor
    high_wavelen = original_max / high_freq_factor
    wavelen = 2 * math.pi / inv_freq

    # Long wavelengths are stretched by the full factor, short ones are left alone, and the
    # band between is interpolated. Applying the stretch everywhere is the classic mistake:
    # it destroys the high-frequency positional detail the short-range heads rely on.
    scaled = torch.where(wavelen > low_wavelen, inv_freq / factor, inv_freq)
    smooth = (original_max / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    smoothed = (1 - smooth) * scaled / factor + smooth * scaled
    in_band = (wavelen >= high_wavelen) & (wavelen <= low_wavelen)
    return torch.where(in_band, smoothed, scaled)


@dataclass(frozen=True)
class DSAGeometry:
    """The region/page arithmetic, in one place.

    ``regions_per_page`` is derived rather than configured: it is the ratio the metadata
    kernels use to split a logical region id into a page lookup and an offset, so a
    configured value that disagreed with ``page_size // region_size`` would address the
    wrong tokens.
    """

    proxy_dim: int
    topk: int
    region_size: int = REGION_BLOCK_SIZE
    page_size: int = 16

    @property
    def regions_per_page(self) -> int:
        if self.page_size % self.region_size:
            raise ValueError(
                f"page size {self.page_size} must be a multiple of region size {self.region_size}"
            )
        return self.page_size // self.region_size


class DSALayerCache:
    """Per-layer DSA state for one batch of in-flight sequences.

    Allocated once for a whole generation, so ``max_tokens`` is a capacity rather than a
    length; ``lengths`` tracks how much of it is live. Every tensor is addressed by the
    conventions the module docstring lists, and mixing them up is the failure this class
    exists to prevent. ``num_kv_groups`` is the *local* count: it is one for the production
    TP=8 layout, with the corresponding global KV head selected while pre-sharding.
    """

    def __init__(
        self,
        *,
        batch: int,
        max_tokens: int,
        num_kv_groups: int,
        head_dim: int,
        geometry: DSAGeometry,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        pages = -(-max_tokens // geometry.page_size)
        self.geometry = geometry
        self.batch = batch
        # Rounded up to a whole page so a request's token span never straddles the boundary
        # between two requests' page ranges.
        self.max_tokens = pages * geometry.page_size
        self.max_regions = self.max_tokens // geometry.region_size

        self.key = torch.zeros(
            (batch * self.max_tokens, num_kv_groups, head_dim), device=device, dtype=dtype
        )
        self.value = torch.zeros_like(self.key)
        self.summary = torch.zeros(
            (batch * self.max_regions, 1, geometry.proxy_dim),
            device=device,
            dtype=torch.float8_e4m3fn,
        )
        self.pending_key = torch.zeros(
            (batch, geometry.region_size, 1, geometry.proxy_dim), device=device, dtype=torch.bfloat16
        )
        self.pending_z = torch.zeros_like(self.pending_key)
        self.lengths = torch.zeros((batch,), device=device, dtype=torch.int32)

        # Request b owns pages [b * pages, (b + 1) * pages), so the mapping is an offset
        # identity. Materialising it as a real table rather than special-casing the
        # contiguous layout keeps the one code path the metadata gates actually cover.
        self.block_table = (
            torch.arange(pages, device=device, dtype=torch.int32).view(1, -1)
            + torch.arange(batch, device=device, dtype=torch.int32).view(-1, 1) * pages
        ).contiguous()

    def token_slice(self, request: int, start: int, count: int) -> slice:
        base = request * self.max_tokens + start
        return slice(base, base + count)

    def write_kv(self, request: int, start: int, key: torch.Tensor, value: torch.Tensor) -> None:
        """Store ``key``/``value`` for one request's tokens at logical position ``start``."""
        span = self.token_slice(request, start, key.shape[0])
        self.key[span] = key
        self.value[span] = value


class Step4SparseIndexer:
    """The indexer's projections and scoring, borrowing its parameters from the attention.

    Not an ``nn.Module``: the checkpoint stores the indexer weights directly under
    ``self_attn`` (``self_attn.sparse_indexer_q.weight``), so wrapping them in a submodule
    would rename every key and require a conversion step to load. The upstream
    implementation splits the same way and for the same reason.
    """

    def __init__(self, owner, *, geometry: DSAGeometry, num_kv_groups: int) -> None:
        self.owner = owner
        self.geometry = geometry
        # Provider groups and attention KV groups are the same partition.  Under TP they
        # are both local counts; for TP=8, one provider group (four indexer heads) drives
        # one local KV group (eight attention Q heads).
        self.num_kv_groups = num_kv_groups
        self.num_heads = int(owner.sparse_indexer_w.weight.shape[0])
        q_out = int(owner.sparse_indexer_q.weight.shape[0])
        if q_out != self.num_heads * geometry.proxy_dim:
            raise ValueError(
                "local sparse_indexer_q/w shapes disagree: "
                f"q rows={q_out}, w heads={self.num_heads}, proxy_dim={geometry.proxy_dim}. "
                "This commonly means stale TP shards from the old replicated-provider layout."
            )
        if self.num_heads % num_kv_groups:
            raise ValueError(
                f"{self.num_heads} indexer heads do not divide into {num_kv_groups} groups"
            )
        self.heads_per_group = self.num_heads // num_kv_groups
        # The indexer's groups *are* the attention's KV groups, so this ratio is fixed under
        # every tensor-parallel degree and the prescale below is a TP invariant.
        self.weight_prescale = float(self.heads_per_group) ** -0.5

    def project(
        self, hidden_states: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project and normalise the indexer's q/k/z, and scale its per-head weights.

        Returns:
            ``(index_q [tokens, groups, heads_per_group, proxy], index_k [tokens, 1, proxy],
            index_z [tokens, 1, proxy], weights [tokens, groups, heads_per_group] fp32)``.
        """

        owner = self.owner
        proxy_dim = self.geometry.proxy_dim
        tokens = hidden_states.shape[0]

        index_q = owner.sparse_indexer_q(hidden_states)
        index_k = owner.sparse_indexer_k(hidden_states)
        index_z = owner.sparse_indexer_z(hidden_states)

        index_q, index_k, index_z = indexer_norm_rope(
            index_q.contiguous(),
            index_k.contiguous(),
            index_z.contiguous(),
            # The factory checkpoint stores these norm parameters in FP32, while
            # vLLM constructs the runtime modules under the model's BF16 default
            # dtype and therefore rounds them during load.  Keep FP32 registered
            # parameters for a lossless checkpoint load, but reproduce that BF16
            # runtime boundary before the fused kernel consumes them.
            owner.sparse_indexer_q_norm.weight.to(dtype=index_q.dtype),
            owner.sparse_indexer_k_norm.weight.to(dtype=index_k.dtype),
            owner.sparse_indexer_k_norm.bias.to(dtype=index_k.dtype),
            owner.sparse_indexer_rope_cos,
            owner.sparse_indexer_rope_sin,
            positions,
            head_dim=proxy_dim,
            num_q_heads=self.num_heads,
            num_k_heads=1,
            rotary_dim=owner.sparse_indexer_rope_cos.shape[1],
            eps=owner.sparse_indexer_q_norm.variance_epsilon,
            q_norm_weight_bias=1.0,
        )

        # The checkpoint stores this tiny matrix in fp32, but the NV reference does not run
        # an fp32 GEMM with it. ``Step3p5SparseIndexerIndexTPLinear`` allocates the runtime
        # parameter with the attention projection's BF16 ``params_dtype`` and its loader
        # copies (therefore rounds) the fp32 checkpoint tensor into that parameter; only the
        # linear result is widened to fp32 before the prescale.  See
        # ``origin/dev/sparse_attn`` step3p5.py:887-892 and 1291-1336.  Keep our registered
        # parameter fp32 so the factory checkpoint loads without conversion, but reproduce
        # the deployed arithmetic by rounding W to the activation dtype before the GEMM.
        weights = torch.nn.functional.linear(
            hidden_states,
            owner.sparse_indexer_w.weight.to(dtype=hidden_states.dtype),
        ).float()
        weights = weights.view(tokens, self.num_kv_groups, self.heads_per_group)
        weights = weights * self.weight_prescale

        return (
            index_q.view(tokens, self.num_kv_groups, self.heads_per_group, proxy_dim),
            index_k.view(tokens, 1, proxy_dim),
            index_z.view(tokens, 1, proxy_dim),
            weights.contiguous(),
        )


def update_summaries_prefill(
    cache: DSALayerCache,
    index_k: torch.Tensor,
    index_z: torch.Tensor,
    seq_lens: torch.Tensor,
) -> None:
    """Compress a packed prefill into per-region summaries, buffering the ragged tail.

    Only regions that are *complete* get a summary. The tail tokens are held instead, and
    the region they belong to is summarised later, once it fills -- see the module
    docstring for why a partial summary must never be stored.
    """
    geometry = cache.geometry
    region_size = geometry.region_size
    starts, counts = csa_region_layout(
        seq_lens, regions_per_seq=cache.max_regions, region_size=region_size
    )
    # A count below ``region_size`` marks the ragged tail; zero makes the kernel skip it.
    counts = torch.where(counts == region_size, counts, torch.zeros_like(counts))

    _, summary_fp8 = csa_compress_regions(
        index_k, index_z, starts, counts, region_size=region_size
    )
    live = counts > 0
    cache.summary[live] = summary_fp8[live]

    token_starts = torch.cumsum(seq_lens.to(torch.int64), dim=0) - seq_lens.to(torch.int64)
    for request in range(cache.batch):
        length = int(seq_lens[request])
        tail = length % region_size
        cache.pending_key[request, :tail] = index_k[
            int(token_starts[request]) + length - tail : int(token_starts[request]) + length
        ].to(cache.pending_key.dtype)
        cache.pending_z[request, :tail] = index_z[
            int(token_starts[request]) + length - tail : int(token_starts[request]) + length
        ].to(cache.pending_z.dtype)


def update_summaries_decode(
    cache: DSALayerCache, index_k: torch.Tensor, index_z: torch.Tensor
) -> None:
    """Append one token per request and summarise any region that just filled.

    ``cache.lengths`` must already have been advanced past the new token, so the pending
    slot is derived from it rather than tracked separately -- one counter that can be wrong
    instead of two that can disagree.
    """
    geometry = cache.geometry
    region_size = geometry.region_size
    lengths = cache.lengths.tolist()

    for request in range(cache.batch):
        length = int(lengths[request])
        slot = (length - 1) % region_size
        cache.pending_key[request, slot] = index_k[request].to(cache.pending_key.dtype)
        cache.pending_z[request, slot] = index_z[request].to(cache.pending_z.dtype)
        if slot != region_size - 1:
            continue

        region = (length - 1) // region_size
        starts = torch.zeros((1,), device=index_k.device, dtype=torch.int32)
        counts = torch.full((1,), region_size, device=index_k.device, dtype=torch.int32)
        _, summary_fp8 = csa_compress_regions(
            cache.pending_key[request].contiguous(),
            cache.pending_z[request].contiguous(),
            starts,
            counts,
            region_size=region_size,
        )
        cache.summary[request * cache.max_regions + region] = summary_fp8[0]


# ============================================================================
# merged source: inference/fp8_gemm.py
# ============================================================================
"""Block-scaled fp8 GEMM for the MoE experts.

Adapted from ``deepseek-ai/DeepSeek-V4-Flash/inference/kernel.py`` (MIT,
Copyright (c) 2023 DeepSeek). Only the fp8 ``act_quant`` + ``fp8_gemm`` paths are
kept; the fp4, sinkhorn, and attention kernels in that file are intentionally
not imported -- this is the one piece the MoE experts need, and a smaller
surface is easier to audit.

Two facts about the DeepSeek scheme are easy to misimplement and invisible when
you do -- they are stated up front because neither is guessable from the
shapes alone:

* **Activations and weights use different blocking.** Activations are
  quantized dynamically at runtime in **1-D blocks of 128 along K** (the
  reduction dim), one fp32 scale per row per 128-K block. Weights are
  quantized offline in **2-D blocks of [128, 128] over (out, K)**, one fp32
  scale per block. The kernel below special-cases ``BLOCK_N == GROUP_N`` and
  ``BLOCK_K == GROUP_K`` so one tile sits inside exactly one weight block and
  one K-block; this is what lets a single weight scale (a scalar per K-iter)
  apply to the whole tile.

* **The stored weight scale uses the *inverse* convention.** The checkpoint
  name is ``weight_scale_inv`` because to dequantize you *multiply* by it,
  not divide: ``real_weight = w_fp8 * weight_scale_inv[block]``. The
  activation scale is the same kind of value (the divisor used during
  quantization); the names differ for historical reasons but both multiply
  back to the real value. The kernel folds the two scales into the fp32
  accumulator after the fp8 dot, never into the fp8 operands.

The dispatch between the fp8 path and the BF16 fallback (layers 88/89/90 ship
BF16 experts with no scale tensor) is by **presence of the scale tensor**,
never by layer index -- a layer can graduate off ``modules_to_not_convert``
without code changes here, and a grouped caller can pass ``None`` per-expert
for just the BF16 experts.
"""


import torch
import triton
import triton.language as tl

# fp8_e4m3fn has 4 exponent bits and 3 mantissa bits, with range [-448, 448].
# The activation quantizer maps each block's range into that envelope by
# dividing by ``max(amax, 1e-10) / 448``.  This is the Optimus/NV Step4
# ``per_token_group_quant_fp8`` default.  DeepSeek's standalone reference uses
# 1e-4, but that larger floor zeros low-amplitude blocks that the deployed
# Step4 path preserves.
FP8_E4M3_MAX: float = 448.0
FP8_AMAX_FLOOR: float = 1e-10

# K-block size shared by the activation quantizer and the weight scale grid.
# The kernel asserts BLOCK_K == GROUP_K so one K-iteration consumes exactly
# one act scale entry and one weight scale entry per row.
BLOCK_K: int = 128
# Weight N-block. The kernel asserts BLOCK_N == GROUP_N so one output tile
# sits inside exactly one weight N-block, keeping the weight scale a scalar
# per (tile, K-iter) instead of a vector.
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

    One program handles a ``[BLOCK_M, BLOCK_K]`` tile -- exactly one quantization
    group wide. The amax is computed over the *live* elements of each row, so
    the kernel handles a final ragged K-block where the tile extends past K:
    the scale covers the live elements only, and the OOB quantized slots are
    not written (the GEMM's masked load will read them as zero, so the round
    trip stays numerically faithful).
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

    # OOB elements are zero via ``other=0.0``, so they do not inflate the amax.
    amax = tl.max(tl.abs(x), axis=1)
    amax = tl.maximum(amax, 1e-10)
    scale = amax * (1.0 / 448.0)
    inv_scale = 1.0 / scale

    # Clamp to [-448, 448] before the round-to-fp8: a value just above 448 would
    # otherwise wrap to NaN instead of saturating.
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

    Args:
        x: input tensor of shape ``[..., K]``. The last dim is the reduction
            dim K. Any leading dims are flattened; the result is reshaped back.
        block_size: K-block size, 128 by default. Must be a power of two so
            Triton accepts it as a constexpr bound on ``tl.arange``.

    Returns:
        ``(y, scale)`` where ``y`` has ``x``'s shape and dtype
        ``torch.float8_e4m3fn``, and ``scale`` has shape ``[..., ceil(K/block_size)]``
        and dtype ``torch.float32``. The recovery formula is
        ``real_x = y.float() * scale.unsqueeze(-1)``.

    Unlike the upstream DeepSeek kernel this does not assert ``K % block_size == 0``.
    A ragged last block is handled by computing the amax over the live elements
    only -- the MoE gate is exercised at K=4096 (a multiple of 128) in
    production, but the parity gate deliberately covers a ragged-K shape to
    keep the kernel honest.
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

    One program computes a ``[BLOCK_M, BLOCK_N]`` output tile. ``BLOCK_N`` must
    equal ``GROUP_N`` (128) so the tile sits inside exactly one weight N-block,
    and ``BLOCK_K`` must equal the activation block size (128) so one K-iter
    consumes exactly one act scale entry and one weight scale entry per row.

    Loads B transposed (as ``[BLOCK_K, BLOCK_N]``) so ``tl.dot`` sees ``A @ B``
    in the natural orientation. The strides are the *original* N/K strides of
    B; the transpose is by index, not by stride.

    The two scales are folded into the fp32 accumulator after the fp8 dot, not
    into the fp8 operands. Folding into the operands would re-quantize and
    lose the precision fp8 GEMM is supposed to keep; the post-dot multiply is
    the whole reason the accumulator is fp32.
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

    # Weight scale N index: tile (pid_n) sits in weight N-block ``pid_n`` when
    # ``BLOCK_N == GROUP_N``. Tiles never span weight N-block boundaries.
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
        # fp8 x fp8 -> fp32 on Hopper; this is the WGMMA path that makes the
        # scheme worth the precision cost. The accumulator is fp32 so the post-
        # dot scale multiply keeps full precision.
        partial = tl.dot(a, b, out_dtype=tl.float32)

        a_s = tl.load(
            A_scale_ptr + offs_m * stride_asm + k_idx * stride_ask,
            mask=m_mask,
            other=0.0,
        )
        # Scalar weight scale for this (n_block, k_idx) -- one fp32 per tile.
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
    """Block-scaled fp8 matmul ``y = x @ w.T``.

    Args:
        x_fp8: ``[M, K]`` ``torch.float8_e4m3fn`` activation, already quantized
            by :func:`act_quant`.
        x_scale: ``[M, ceil(K/128)]`` ``torch.float32`` activation scales.
        w_fp8: ``[N, K]`` ``torch.float8_e4m3fn`` weight (one expert's slice).
        w_scale_inv: ``[ceil(N/128), ceil(K/128)]`` ``torch.float32`` weight
            scales stored in the *inverse* convention
            (``real_weight = w_fp8 * w_scale_inv[block]``).

    Returns:
        ``y`` of shape ``[M, N]`` in ``torch.bfloat16``.
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
        raise ValueError(
            f"x_scale shape {tuple(x_scale.shape)} != ({M}, {k_blocks})"
        )
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
    """``y = x @ weight.T`` with fp8 path when a scale is present, BF16 fallback.

    The dispatcher is the only place that decides fp8 vs BF16: layers 88/89/90
    ship BF16 experts with no scale, and the criterion is *presence of the
    scale tensor*, never the layer index. A layer can graduate off the
    not-convert list without code changes here.

    Args:
        x: ``[..., K]`` activation. Cast to BF16 internally on the fallback path
            (the fp8 path quantizes through BF16 anyway).
        weight: ``[N, K]`` weight; ``torch.float8_e4m3fn`` when
            ``weight_scale_inv`` is present, otherwise BF16.
        weight_scale_inv: ``[ceil(N/128), ceil(K/128)]`` fp32 inverse scales, or
            ``None`` to select the BF16 fallback path.

    Returns:
        ``[..., N]`` BF16 activation.
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


# ============================================================================
# merged source: inference/moe_kernels.py
# ============================================================================
"""Small Triton primitives used by Step-4's MoE inference path.

The kernels in this module intentionally mirror the arithmetic of the deployed
vLLM/Optimus path rather than replacing it with mathematically equivalent
PyTorch reductions:

* SwiGLU loads the independent BF16 gate and up projections into FP32, applies
  the asymmetric clamps, multiplies in FP32, and rounds to BF16 only once.
* The top-k gather walks slots in logical order and maintains one FP32
  accumulator.  In particular, it is not a parallel reduction over the top-k
  dimension, whose reassociation could change the final BF16 value.

Importing this module does not initialize CUDA.  Triton is also an optional
import at module-import time so CPU-only tooling can inspect the modelling
code; attempting to execute either helper still requires CUDA and Triton.
"""


import math

import torch

try:
    import triton
    import triton.language as tl
except ImportError as exc:  # pragma: no cover - exercised only without Triton installed
    triton = None
    tl = None
    _TRITON_IMPORT_ERROR: ImportError | None = exc
else:
    _TRITON_IMPORT_ERROR = None


_BLOCK_SIZE = 1024


if triton is not None:

    @triton.jit
    def _clamped_swiglu_kernel(
        gate_ptr,
        up_ptr,
        output_ptr,
        num_elements,
        LIMIT: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """FP32 SWIGLUstep over two independent, contiguous input tensors."""
        block = tl.program_id(0).to(tl.int64)
        offsets = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_elements

        gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(up_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        gate_silu = gate * tl.sigmoid(gate)
        gate_clamped = tl.minimum(gate_silu, LIMIT)
        up_clamped = tl.minimum(tl.maximum(up, -LIMIT), LIMIT)
        output = gate_clamped * up_clamped

        tl.store(
            output_ptr + offsets,
            output.to(output_ptr.dtype.element_ty),
            mask=mask,
        )

    @triton.jit
    def _weighted_topk_gather_kernel(
        contributions_ptr,
        weights_ptr,
        output_ptr,
        hidden_size,
        TOP_K: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Sum one ``[top_k, hidden]`` slice in logical slot order."""
        token = tl.program_id(0).to(tl.int64)
        hidden_block = tl.program_id(1).to(tl.int64)
        hidden_offsets = hidden_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        hidden_mask = hidden_offsets < hidden_size

        accumulator = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        # TOP_K is constexpr, just as ``topk_num`` is in Optimus ep_gather.
        # Keeping the update in this loop produces the required slot-ordered
        # FP32 accumulation instead of a tree reduction over the K dimension.
        for slot in range(0, TOP_K):
            contribution_offset = (token * TOP_K + slot) * hidden_size + hidden_offsets
            contribution = tl.load(
                contributions_ptr + contribution_offset,
                mask=hidden_mask,
                other=0.0,
            ).to(tl.float32)
            weight = tl.load(weights_ptr + token * TOP_K + slot).to(tl.float32)
            accumulator += contribution * weight

        output_offset = token * hidden_size + hidden_offsets
        tl.store(
            output_ptr + output_offset,
            accumulator.to(output_ptr.dtype.element_ty),
            mask=hidden_mask,
        )

else:  # Keep names available to importers and type/introspection tools.
    _clamped_swiglu_kernel = None
    _weighted_topk_gather_kernel = None


def _require_cuda_and_triton(tensors: tuple[torch.Tensor, ...]) -> None:
    if _TRITON_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Triton is required to execute Step-4 MoE kernels"
        ) from _TRITON_IMPORT_ERROR
    if any(tensor.device.type != "cuda" for tensor in tensors):
        devices = ", ".join(str(tensor.device) for tensor in tensors)
        raise RuntimeError(
            f"Step-4 MoE Triton kernels require CUDA tensors, got [{devices}]"
        )


def clamped_swiglu(
    gate: torch.Tensor,
    up: torch.Tensor,
    limit: float = 7.0,
) -> torch.Tensor:
    """Apply Step-4's clamped SwiGLU to independent BF16 projections.

    Computes

    ``silu(gate.float()).clamp(max=limit) * up.float().clamp(-limit, limit)``

    in FP32 and casts the product to the input dtype once.  ``gate`` and ``up``
    must have the same shape and be contiguous BF16 CUDA tensors.
    """
    if gate.shape != up.shape:
        raise ValueError(
            f"gate/up shape mismatch: {tuple(gate.shape)} != {tuple(up.shape)}"
        )
    if gate.dtype != torch.bfloat16 or up.dtype != torch.bfloat16:
        raise TypeError(
            f"gate and up must both be torch.bfloat16, got {gate.dtype} and {up.dtype}"
        )
    if gate.device != up.device:
        raise ValueError(f"gate/up device mismatch: {gate.device} != {up.device}")
    if not gate.is_contiguous() or not up.is_contiguous():
        raise ValueError("gate and up must both be contiguous")

    try:
        limit_value = float(limit)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"limit must be a real scalar, got {limit!r}") from exc
    if not math.isfinite(limit_value) or limit_value < 0.0:
        raise ValueError(f"limit must be finite and non-negative, got {limit_value}")

    _require_cuda_and_triton((gate, up))
    output = torch.empty_like(gate)
    if gate.numel() == 0:
        return output

    assert triton is not None and _clamped_swiglu_kernel is not None
    grid = (triton.cdiv(gate.numel(), _BLOCK_SIZE),)
    _clamped_swiglu_kernel[grid](
        gate,
        up,
        output,
        gate.numel(),
        LIMIT=limit_value,
        BLOCK_SIZE=_BLOCK_SIZE,
        num_warps=4,
    )
    return output


def weighted_topk_gather(
    contributions: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Weight and sum top-k contributions in deterministic slot order.

    Args:
        contributions: contiguous BF16 tensor of shape ``[T, K, H]``.
        weights: contiguous FP32 tensor of shape ``[T, K]``.

    Returns:
        Contiguous BF16 tensor of shape ``[T, H]``.  For every token and hidden
        element the kernel starts from FP32 zero and performs
        ``acc += contributions[:, slot].float() * weights[:, slot]`` for slots
        ``0, 1, ..., K - 1``, then casts the accumulator to BF16 once.
    """
    if contributions.ndim != 3:
        raise ValueError(
            f"contributions must have shape [T, K, H], got {tuple(contributions.shape)}"
        )
    if weights.ndim != 2:
        raise ValueError(f"weights must have shape [T, K], got {tuple(weights.shape)}")

    tokens, top_k, hidden_size = contributions.shape
    if weights.shape != (tokens, top_k):
        raise ValueError(f"weights shape {tuple(weights.shape)} != ({tokens}, {top_k})")
    if top_k <= 0:
        raise ValueError(f"top-k dimension must be positive, got {top_k}")
    if hidden_size <= 0:
        raise ValueError(f"hidden dimension must be positive, got {hidden_size}")
    if contributions.dtype != torch.bfloat16:
        raise TypeError(
            f"contributions must be torch.bfloat16, got {contributions.dtype}"
        )
    if weights.dtype != torch.float32:
        raise TypeError(f"weights must be torch.float32, got {weights.dtype}")
    if contributions.device != weights.device:
        raise ValueError(
            "contributions/weights device mismatch: "
            f"{contributions.device} != {weights.device}"
        )
    if not contributions.is_contiguous() or not weights.is_contiguous():
        raise ValueError("contributions and weights must both be contiguous")

    _require_cuda_and_triton((contributions, weights))
    output = torch.empty(
        (tokens, hidden_size),
        dtype=torch.bfloat16,
        device=contributions.device,
    )
    if tokens == 0:
        return output

    assert triton is not None and _weighted_topk_gather_kernel is not None
    grid = (tokens, triton.cdiv(hidden_size, _BLOCK_SIZE))
    _weighted_topk_gather_kernel[grid](
        contributions,
        weights,
        output,
        hidden_size,
        TOP_K=top_k,
        BLOCK_SIZE=_BLOCK_SIZE,
        num_warps=4,
    )
    return output


# ============================================================================
# merged source: inference/qknorm_rope.py
# ============================================================================
"""Bitwise-oriented Triton Q/K RMSNorm + NeoX RoPE for Step-4.

The packed projection is laid out as ``[Q heads, K heads, V heads]``.  Q and K
are normalized head by head, rounded to BF16, reloaded as FP32, and only then
rotated.  V is copied without arithmetic.  Those details are observable: the
deployed CuTe kernel uses geometry-specific 16- or 32-lane RMS reductions and
has an activation-dtype materialization boundary between RMSNorm and RoPE.

This module intentionally does not depend on vLLM or CuTe.  Triton is optional
at import time so model/config tooling remains usable in CPU-only environments;
calling :func:`fused_qknorm_rope` still requires Triton and a CUDA device.
"""


import math

import torch

try:
    import triton
    import triton.language as tl

    # Triton moved libdevice from ``language`` to ``language.extra`` in 3.x.
    try:
        import triton.language.extra.libdevice as _libdevice
    except ImportError:  # pragma: no cover - exercised by the Triton 2.x runtime.
        _libdevice = tl.libdevice
except ImportError:  # pragma: no cover - covered by the CPU import contract.
    triton = None
    tl = None
    _libdevice = None


HEAD_DIM = 192
_SUPPORTED_ROTARY_PAIRS = frozenset((32, 96))


if triton is not None:

    @triton.jit
    def _lane_square_sum_192(
        qkv_ptr,
        head_base,
        LANE: tl.constexpr,
    ):
        """CuTe's 32-pair layout: three four-value vectors for one of 16 lanes."""
        acc = 0.0
        # The order is part of the numerical contract:
        #   block 0 dims [0, 64), block 1 [64, 128), block 2 [128, 192),
        # with four adjacent values owned by each logical lane in every block.
        for block in range(3):
            for element in range(4):
                dim = block * 64 + LANE * 4 + element
                value = tl.load(qkv_ptr + head_base + dim).to(tl.float32)
                acc = _libdevice.fma_rn(value, value, acc)
        return acc

    @triton.jit
    def _wide_lane_square_sum_192(
        qkv_ptr,
        head_base,
        LANE: tl.constexpr,
    ):
        """Installed vLLM CuTe layout for the 96-pair Step-4 path.

        Covering 96 RoPE pairs makes that kernel widen from 16 to 32 logical
        lanes.  Its 128-bit copy assigns eight adjacent values to each lane
        and pads the 192-value row to 256 values; lanes 24..31 therefore
        contribute eight explicit zeros apiece.
        """
        acc = 0.0
        for element in range(8):
            dim = LANE * 8 + element
            value = tl.load(
                qkv_ptr + head_base + dim,
                mask=dim < 192,
                other=0.0,
            ).to(tl.float32)
            acc = _libdevice.fma_rn(value, value, acc)
        return acc

    @triton.jit
    def _qknorm_rope_192_kernel(
        qkv_ptr,
        q_out_ptr,
        k_out_ptr,
        q_weight_ptr,
        k_weight_ptr,
        cos_ptr,
        sin_ptr,
        positions_ptr,
        stride_qkv_token,
        stride_q_out_token,
        stride_k_out_token,
        stride_cos_token,
        stride_sin_token,
        NUM_Q_HEADS: tl.constexpr,
        NUM_KV_HEADS: tl.constexpr,
        ROTARY_PAIRS: tl.constexpr,
        EPS: tl.constexpr,
        NORM_WEIGHT_BIAS: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_R: tl.constexpr,
    ):
        """One program per token and Q/K head.

        The physical Triton layout is deliberately not used to define the
        reduction.  Explicit logical lane sums followed by the exact ascending
        XOR tree make the result independent of Triton's evolving warp/layout
        lowering.  Step-4's 32-pair path uses 16 lanes; the 96-pair path uses
        32 so all rotary pairs are covered.
        """
        token = tl.program_id(0)
        packed_head = tl.program_id(1)
        is_q = packed_head < NUM_Q_HEADS
        head_base = token * stride_qkv_token + packed_head * 192

        if ROTARY_PAIRS == 96:
            # The production vLLM CuTe kernel widens to 32 lanes so four
            # RoPE values per lane cover all 96 pairs.  Its norm copy layout
            # changes at the same time to 8 contiguous values/lane, padded to
            # 256.  Preserve both the local FMA chains and XOR(1,2,4,8,16).
            p0 = _wide_lane_square_sum_192(qkv_ptr, head_base, LANE=0)
            p1 = _wide_lane_square_sum_192(qkv_ptr, head_base, LANE=1)
            p2 = _wide_lane_square_sum_192(qkv_ptr, head_base, LANE=2)
            p3 = _wide_lane_square_sum_192(qkv_ptr, head_base, LANE=3)
            p4 = _wide_lane_square_sum_192(qkv_ptr, head_base, LANE=4)
            p5 = _wide_lane_square_sum_192(qkv_ptr, head_base, LANE=5)
            p6 = _wide_lane_square_sum_192(qkv_ptr, head_base, LANE=6)
            p7 = _wide_lane_square_sum_192(qkv_ptr, head_base, LANE=7)
            p8 = _wide_lane_square_sum_192(qkv_ptr, head_base, LANE=8)
            p9 = _wide_lane_square_sum_192(qkv_ptr, head_base, LANE=9)
            p10 = _wide_lane_square_sum_192(qkv_ptr, head_base, LANE=10)
            p11 = _wide_lane_square_sum_192(qkv_ptr, head_base, LANE=11)
            p12 = _wide_lane_square_sum_192(qkv_ptr, head_base, LANE=12)
            p13 = _wide_lane_square_sum_192(qkv_ptr, head_base, LANE=13)
            p14 = _wide_lane_square_sum_192(qkv_ptr, head_base, LANE=14)
            p15 = _wide_lane_square_sum_192(qkv_ptr, head_base, LANE=15)
            p16 = _wide_lane_square_sum_192(qkv_ptr, head_base, LANE=16)
            p17 = _wide_lane_square_sum_192(qkv_ptr, head_base, LANE=17)
            p18 = _wide_lane_square_sum_192(qkv_ptr, head_base, LANE=18)
            p19 = _wide_lane_square_sum_192(qkv_ptr, head_base, LANE=19)
            p20 = _wide_lane_square_sum_192(qkv_ptr, head_base, LANE=20)
            p21 = _wide_lane_square_sum_192(qkv_ptr, head_base, LANE=21)
            p22 = _wide_lane_square_sum_192(qkv_ptr, head_base, LANE=22)
            p23 = _wide_lane_square_sum_192(qkv_ptr, head_base, LANE=23)
            p24 = _wide_lane_square_sum_192(qkv_ptr, head_base, LANE=24)
            p25 = _wide_lane_square_sum_192(qkv_ptr, head_base, LANE=25)
            p26 = _wide_lane_square_sum_192(qkv_ptr, head_base, LANE=26)
            p27 = _wide_lane_square_sum_192(qkv_ptr, head_base, LANE=27)
            p28 = _wide_lane_square_sum_192(qkv_ptr, head_base, LANE=28)
            p29 = _wide_lane_square_sum_192(qkv_ptr, head_base, LANE=29)
            p30 = _wide_lane_square_sum_192(qkv_ptr, head_base, LANE=30)
            p31 = _wide_lane_square_sum_192(qkv_ptr, head_base, LANE=31)

            s01 = _libdevice.add_rn(p0, p1)
            s23 = _libdevice.add_rn(p2, p3)
            s45 = _libdevice.add_rn(p4, p5)
            s67 = _libdevice.add_rn(p6, p7)
            s89 = _libdevice.add_rn(p8, p9)
            s1011 = _libdevice.add_rn(p10, p11)
            s1213 = _libdevice.add_rn(p12, p13)
            s1415 = _libdevice.add_rn(p14, p15)
            s1617 = _libdevice.add_rn(p16, p17)
            s1819 = _libdevice.add_rn(p18, p19)
            s2021 = _libdevice.add_rn(p20, p21)
            s2223 = _libdevice.add_rn(p22, p23)
            s2425 = _libdevice.add_rn(p24, p25)
            s2627 = _libdevice.add_rn(p26, p27)
            s2829 = _libdevice.add_rn(p28, p29)
            s3031 = _libdevice.add_rn(p30, p31)
            s03 = _libdevice.add_rn(s01, s23)
            s47 = _libdevice.add_rn(s45, s67)
            s811 = _libdevice.add_rn(s89, s1011)
            s1215 = _libdevice.add_rn(s1213, s1415)
            s1619 = _libdevice.add_rn(s1617, s1819)
            s2023 = _libdevice.add_rn(s2021, s2223)
            s2427 = _libdevice.add_rn(s2425, s2627)
            s2831 = _libdevice.add_rn(s2829, s3031)
            s07 = _libdevice.add_rn(s03, s47)
            s815 = _libdevice.add_rn(s811, s1215)
            s1623 = _libdevice.add_rn(s1619, s2023)
            s2431 = _libdevice.add_rn(s2427, s2831)
            s015 = _libdevice.add_rn(s07, s815)
            s1631 = _libdevice.add_rn(s1623, s2431)
            sum_squares = _libdevice.add_rn(s015, s1631)
        else:
            # 32-pair full-attention layers use 16 lanes.  Each pN is the
            # sequential 12-value fma.rn chain of logical lane N.
            p0 = _lane_square_sum_192(qkv_ptr, head_base, LANE=0)
            p1 = _lane_square_sum_192(qkv_ptr, head_base, LANE=1)
            p2 = _lane_square_sum_192(qkv_ptr, head_base, LANE=2)
            p3 = _lane_square_sum_192(qkv_ptr, head_base, LANE=3)
            p4 = _lane_square_sum_192(qkv_ptr, head_base, LANE=4)
            p5 = _lane_square_sum_192(qkv_ptr, head_base, LANE=5)
            p6 = _lane_square_sum_192(qkv_ptr, head_base, LANE=6)
            p7 = _lane_square_sum_192(qkv_ptr, head_base, LANE=7)
            p8 = _lane_square_sum_192(qkv_ptr, head_base, LANE=8)
            p9 = _lane_square_sum_192(qkv_ptr, head_base, LANE=9)
            p10 = _lane_square_sum_192(qkv_ptr, head_base, LANE=10)
            p11 = _lane_square_sum_192(qkv_ptr, head_base, LANE=11)
            p12 = _lane_square_sum_192(qkv_ptr, head_base, LANE=12)
            p13 = _lane_square_sum_192(qkv_ptr, head_base, LANE=13)
            p14 = _lane_square_sum_192(qkv_ptr, head_base, LANE=14)
            p15 = _lane_square_sum_192(qkv_ptr, head_base, LANE=15)

            # Lane 0 after ascending shfl_xor offsets 1, 2, 4, 8.
            s01 = _libdevice.add_rn(p0, p1)
            s23 = _libdevice.add_rn(p2, p3)
            s45 = _libdevice.add_rn(p4, p5)
            s67 = _libdevice.add_rn(p6, p7)
            s89 = _libdevice.add_rn(p8, p9)
            s1011 = _libdevice.add_rn(p10, p11)
            s1213 = _libdevice.add_rn(p12, p13)
            s1415 = _libdevice.add_rn(p14, p15)
            s03 = _libdevice.add_rn(s01, s23)
            s47 = _libdevice.add_rn(s45, s67)
            s811 = _libdevice.add_rn(s89, s1011)
            s1215 = _libdevice.add_rn(s1213, s1415)
            s07 = _libdevice.add_rn(s03, s47)
            s815 = _libdevice.add_rn(s811, s1215)
            sum_squares = _libdevice.add_rn(s07, s815)

        mean_square = _libdevice.div_rn(sum_squares, 192.0)
        inverse_rms = _libdevice.rsqrt(_libdevice.add_rn(mean_square, EPS))

        dims = tl.arange(0, BLOCK_D)
        dim_mask = dims < 192
        values = tl.load(
            qkv_ptr + head_base + dims,
            mask=dim_mask,
            other=0.0,
        ).to(tl.float32)
        q_weight = tl.load(
            q_weight_ptr + dims,
            mask=dim_mask,
            other=0.0,
        ).to(tl.float32)
        k_weight = tl.load(
            k_weight_ptr + dims,
            mask=dim_mask,
            other=0.0,
        ).to(tl.float32)
        weight = tl.where(is_q, q_weight, k_weight)
        weight = _libdevice.add_rn(weight, NORM_WEIGHT_BIAS)
        x_hat = _libdevice.mul_rn(values, inverse_rms)
        normalized = _libdevice.mul_rn(x_hat, weight).to(tl.bfloat16)

        q_base = token * stride_q_out_token + packed_head * 192
        k_base = token * stride_k_out_token + (packed_head - NUM_Q_HEADS) * 192
        tl.store(
            q_out_ptr + q_base + dims,
            normalized,
            mask=dim_mask & is_q,
        )
        tl.store(
            k_out_ptr + k_base + dims,
            normalized,
            mask=dim_mask & ~is_q,
        )

        # This global-memory round trip is the Triton equivalent of CuTe's
        # activation-typed shared-memory boundary.  A barrier prevents the
        # reload below from observing the pre-store contents.
        tl.debug_barrier()

        pairs = tl.arange(0, BLOCK_R)
        pair_mask = pairs < ROTARY_PAIRS
        dim0 = pairs
        dim1 = pairs + ROTARY_PAIRS
        q_value0 = tl.load(
            q_out_ptr + q_base + dim0,
            mask=pair_mask & is_q,
            other=0.0,
        ).to(tl.float32)
        q_value1 = tl.load(
            q_out_ptr + q_base + dim1,
            mask=pair_mask & is_q,
            other=0.0,
        ).to(tl.float32)
        k_value0 = tl.load(
            k_out_ptr + k_base + dim0,
            mask=pair_mask & ~is_q,
            other=0.0,
        ).to(tl.float32)
        k_value1 = tl.load(
            k_out_ptr + k_base + dim1,
            mask=pair_mask & ~is_q,
            other=0.0,
        ).to(tl.float32)
        value0 = tl.where(is_q, q_value0, k_value0)
        value1 = tl.where(is_q, q_value1, k_value1)

        position = tl.load(positions_ptr + token)
        cos_value = tl.load(
            cos_ptr + position * stride_cos_token + pairs,
            mask=pair_mask,
            other=0.0,
        ).to(tl.float32)
        sin_value = tl.load(
            sin_ptr + position * stride_sin_token + pairs,
            mask=pair_mask,
            other=0.0,
        ).to(tl.float32)

        # CuTe PTX contract:
        #   rot0 = mul(value0, cos) - mul(value1, sin)
        #   rot1 = fma(value0, sin, mul(value1, cos))
        rotated0 = _libdevice.sub_rn(
            _libdevice.mul_rn(value0, cos_value),
            _libdevice.mul_rn(value1, sin_value),
        ).to(tl.bfloat16)
        rotated1 = _libdevice.fma_rn(
            value0,
            sin_value,
            _libdevice.mul_rn(value1, cos_value),
        ).to(tl.bfloat16)

        tl.store(
            q_out_ptr + q_base + dim0,
            rotated0,
            mask=pair_mask & is_q,
        )
        tl.store(
            q_out_ptr + q_base + dim1,
            rotated1,
            mask=pair_mask & is_q,
        )
        tl.store(
            k_out_ptr + k_base + dim0,
            rotated0,
            mask=pair_mask & ~is_q,
        )
        tl.store(
            k_out_ptr + k_base + dim1,
            rotated1,
            mask=pair_mask & ~is_q,
        )

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
        mask = dims < 192
        input_base = (
            token * stride_qkv_token + (NUM_Q_HEADS + NUM_KV_HEADS + head) * 192
        )
        output_base = token * stride_v_out_token + head * 192
        # BF16 load/store only: V must retain its original bits.
        value = tl.load(qkv_ptr + input_base + dims, mask=mask)
        tl.store(v_out_ptr + output_base + dims, value, mask=mask)


def _validate_qknorm_rope_inputs(
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
    eps: float,
    norm_weight_bias: float,
) -> None:
    tensors = {
        "qkv": qkv,
        "q_weight": q_weight,
        "k_weight": k_weight,
        "cos": cos,
        "sin": sin,
        "positions": positions,
    }
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")

    if head_dim != HEAD_DIM:
        raise ValueError(f"head_dim must be {HEAD_DIM}, got {head_dim}")
    if num_q_heads <= 0 or num_kv_heads <= 0:
        raise ValueError("num_q_heads and num_kv_heads must be positive")
    if rotary_pairs not in _SUPPORTED_ROTARY_PAIRS:
        raise ValueError(
            "rotary_pairs must be 32 (full-attention layers) or 96 "
            f"(sliding-attention layers), got {rotary_pairs}"
        )
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError(f"eps must be finite and positive, got {eps}")
    if not math.isfinite(norm_weight_bias):
        raise ValueError(f"norm_weight_bias must be finite, got {norm_weight_bias}")

    packed_width = (num_q_heads + 2 * num_kv_heads) * head_dim
    if qkv.ndim != 2 or tuple(qkv.shape) != (qkv.shape[0], packed_width):
        raise ValueError(
            f"qkv must have shape [tokens, {packed_width}], got {tuple(qkv.shape)}"
        )
    if qkv.stride(1) != 1:
        raise ValueError("qkv must be contiguous in its last dimension")
    if qkv.dtype != torch.bfloat16:
        raise TypeError(f"qkv must have dtype torch.bfloat16, got {qkv.dtype}")

    for name, weight in (("q_weight", q_weight), ("k_weight", k_weight)):
        if weight.ndim != 1 or weight.numel() != head_dim:
            raise ValueError(
                f"{name} must have shape [{head_dim}], got {tuple(weight.shape)}"
            )
        if not weight.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
        if weight.dtype != torch.float32:
            raise TypeError(f"{name} must have dtype torch.float32, got {weight.dtype}")

    if cos.ndim != 2 or sin.ndim != 2:
        raise ValueError(
            f"cos and sin must be 2D, got {tuple(cos.shape)} and {tuple(sin.shape)}"
        )
    if cos.shape != sin.shape:
        raise ValueError(
            f"cos and sin must have equal shapes, got {tuple(cos.shape)} "
            f"and {tuple(sin.shape)}"
        )
    if cos.shape[1] < rotary_pairs:
        raise ValueError(
            f"cos/sin width must be at least {rotary_pairs}, got {cos.shape[1]}"
        )
    if cos.stride(1) != 1 or sin.stride(1) != 1:
        raise ValueError("cos and sin must be contiguous in their last dimension")
    if cos.dtype != torch.bfloat16 or sin.dtype != torch.bfloat16:
        raise TypeError(
            "cos and sin must have dtype torch.bfloat16, got "
            f"{cos.dtype} and {sin.dtype}"
        )

    if positions.ndim != 1 or positions.numel() != qkv.shape[0]:
        raise ValueError(
            "positions must have one entry per qkv token, got "
            f"shape {tuple(positions.shape)} for {qkv.shape[0]} tokens"
        )
    if not positions.is_contiguous():
        raise ValueError("positions must be contiguous")
    if positions.dtype not in (torch.int32, torch.int64):
        raise TypeError(
            "positions must have dtype torch.int32 or torch.int64, got "
            f"{positions.dtype}"
        )

    # Device checks intentionally come last.  Shape and dtype are properties of
    # the public contract too, and validating them first lets CPU-only tooling
    # diagnose malformed model inputs without needing a CUDA allocation.
    for name, tensor in tensors.items():
        if not tensor.is_cuda:
            raise ValueError(f"{name} must be a CUDA tensor")
    devices = {tensor.device for tensor in tensors.values()}
    if len(devices) != 1:
        raise ValueError("all inputs must be on the same CUDA device")


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
    eps: float = 1e-5,
    norm_weight_bias: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize and rotate packed TP-local Q/K heads, and copy V exactly.

    Args:
        qkv: BF16 ``[tokens, (num_q_heads + 2 * num_kv_heads) * 192]``.
            A wider physical row stride is allowed, but the last dimension must
            be contiguous.
        q_weight, k_weight: FP32 zero-centered RMSNorm weights of shape
            ``[192]``.  The effective scale is ``weight + norm_weight_bias``;
            Step-4 uses the default bias of ``1.0``.
        cos, sin: BF16 NeoX tables ``[max_position, >= rotary_pairs]``.
        positions: contiguous int32 or int64 position for every token.
        rotary_pairs: ``96`` on sliding layers and ``32`` on full-attention
            layers.  The leading ``2 * rotary_pairs`` dimensions are rotated.

    Returns:
        Flat contiguous BF16 tensors ``q``, ``k``, ``v`` with shapes
        ``[tokens, num_q_heads * 192]`` and
        ``[tokens, num_kv_heads * 192]`` (for both K and V).

    The older native BITWISE reference hard-codes four RoPE pairs per one of
    16 lanes, which only covers 64 pairs at D=192.  That is valid for its
    tested 48-pair geometry but incomplete for Step-4's 96-pair sliding
    layers.  The deployed CuTe/vLLM path widens that case to 32 lanes and a
    padded 256-value normalization tile.  This implementation reproduces that
    reduction and rotates all 96 pairs; 32-pair full-attention layers retain
    the 16-lane blocked layout.
    """
    _validate_qknorm_rope_inputs(
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
    if triton is None:
        raise RuntimeError("Triton is required to run fused_qknorm_rope")

    tokens = qkv.shape[0]
    q_out = torch.empty(
        (tokens, num_q_heads * HEAD_DIM),
        device=qkv.device,
        dtype=qkv.dtype,
    )
    k_out = torch.empty(
        (tokens, num_kv_heads * HEAD_DIM),
        device=qkv.device,
        dtype=qkv.dtype,
    )
    v_out = torch.empty_like(k_out)
    if tokens == 0:
        return q_out, k_out, v_out

    _qknorm_rope_192_kernel[(tokens, num_q_heads + num_kv_heads)](
        qkv,
        q_out,
        k_out,
        q_weight,
        k_weight,
        cos,
        sin,
        positions,
        qkv.stride(0),
        q_out.stride(0),
        k_out.stride(0),
        cos.stride(0),
        sin.stride(0),
        NUM_Q_HEADS=num_q_heads,
        NUM_KV_HEADS=num_kv_heads,
        ROTARY_PAIRS=rotary_pairs,
        EPS=float(eps),
        NORM_WEIGHT_BIAS=float(norm_weight_bias),
        BLOCK_D=256,
        BLOCK_R=128,
        num_warps=4,
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


__all__ = [
    "REGION_BLOCK_SIZE",
    "PROXY_DIM",
    "shared_table_stride",
    "indexer_norm_rope",
    "csa_region_layout",
    "csa_compress_regions",
    "quantize_activations_e4m3",
    "indexer_logits_paged",
    "region_topk_ids",
    "region_topk_pack",
    "prefill_sparse_meta",
    "decode_sparse_meta",
    "sparse_attention_prefill",
    "merge_split_states",
    "sparse_attention_decode",
    "build_rope_cache",
    "DSAGeometry",
    "DSALayerCache",
    "Step4SparseIndexer",
    "update_summaries_prefill",
    "update_summaries_decode",
    "act_quant",
    "fp8_gemm",
    "linear_fp8_or_bf16",
    "clamped_swiglu",
    "weighted_topk_gather",
    "HEAD_DIM",
    "fused_qknorm_rope",
]
