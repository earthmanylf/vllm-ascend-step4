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
        tile_max = tl.maximum(running_max, tl.max(scores, axis=1, propagate_nan=True),propagate_nan=tl.PropagateNan.ALL)
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
        enable_dynamic_cv_pipeline=False, set_workspace_multibuffer=1, enable_preload=False
    )
    return out, lse
__all__ = [
    "sparse_attention_prefill",
]
