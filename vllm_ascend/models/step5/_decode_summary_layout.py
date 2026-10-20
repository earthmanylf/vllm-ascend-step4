"""Fused Step3p5 DSA decode/prefill summary-cache runtime layout builder.

Replaces the ~40 tiny 1-D elementwise/index torch ops in
``Step3p5DSAAttentionImpl._get_or_build_runtime_layout`` +
``Step3p5SparseSummaryCache.flat_region_ids_from_token_slots`` with a single
Triton kernel. For each of ``T`` tokens it computes, in one launch:

  * token_positions  (logical position of the token in its sequence)
  * token_flat_slot  (physical summary region id)
  * token_valid      (whether the region id is usable)
  * reset_slots      (flat_slot at a region-block boundary, else -1)

``token_slots`` is just ``slot_mapping`` viewed as int64 on the unpadded path.
When a padded output buffer is supplied, the kernel writes token slots too so
CUDA graph replay can use a fixed row count without a separate copy/pad step.

The per-request metadata (which request owns a token, its logical position)
is recovered with an in-kernel binary search over ``query_start_loc`` +
``seq_lens``, mirroring the reference torch computation exactly, so the fused
path is bit-for-bit equivalent for both decode and prefill.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

# Keep the binary-search unroll independent of runtime batch size. This single
# variant supports request counts through 4095 without a runtime compile.
_SEARCH_STEPS = 12
_DECODE_SUMMARY_LAYOUT_ALIGNMENT_DYNAMIC_ARGS = (
    "slot_mapping_ptr",
    "query_start_loc_ptr",
    "seq_lens_ptr",
    "flat_slot_ptr",
    "token_slots_out_ptr",
    "token_positions_ptr",
    "reset_slots_ptr",
    "token_valid_ptr",
)


def _validate_search_batch(num_reqs: int) -> None:
    if int(num_reqs) >= (1 << _SEARCH_STEPS):
        raise ValueError(
            "Step3p5 DSA summary layout supports at most "
            f"{(1 << _SEARCH_STEPS) - 1} requests without runtime compilation, "
            f"got {int(num_reqs)}"
        )


@triton.jit
def _upper_bound_req_idx(
    query_start_loc_ptr,
    q_idx,
    num_reqs,
    live_mask,
    SEARCH_STEPS: tl.constexpr,
):
    lo = tl.full(q_idx.shape, 0, tl.int32)
    hi = tl.full(q_idx.shape, num_reqs, tl.int32)
    for _ in range(SEARCH_STEPS):
        active = live_mask & (lo < hi)
        mid = tl.where(active, (lo + hi) // 2, lo)
        mid_val = tl.load(
            query_start_loc_ptr + mid,
            mask=active,
            other=0,
        ).to(tl.int32)
        go_right = active & (mid_val <= q_idx)
        go_left = active & ~go_right
        lo = tl.where(go_right, mid + 1, lo)
        hi = tl.where(go_left, mid, hi)
    return lo - 1


@triton.jit(
    do_not_specialize=[
        "num_tokens",
        "output_rows",
        "num_reqs",
        "total_regions",
        "max_slot",
    ],
    do_not_specialize_on_alignment=_DECODE_SUMMARY_LAYOUT_ALIGNMENT_DYNAMIC_ARGS,
)
def _decode_summary_layout_kernel(
    slot_mapping_ptr,        # int64 [T]
    query_start_loc_ptr,     # int32 [num_reqs + 1]
    seq_lens_ptr,            # int32 [num_reqs]
    flat_slot_ptr,           # int64 [T]  (out)
    token_slots_out_ptr,     # int64 [T]  (out)
    token_positions_ptr,     # int64 [T]  (out)
    reset_slots_ptr,         # int64 [T]  (out)
    token_valid_ptr,         # bool  [T]  (out)
    num_tokens,
    output_rows,
    num_reqs,
    total_regions,
    max_slot,
    page_size: tl.constexpr,
    region_block_size: tl.constexpr,
    summaries_per_page: tl.constexpr,
    BLOCK_R: tl.constexpr,
    SEARCH_STEPS: tl.constexpr,
) -> None:
    pid = tl.program_id(0)
    idx = pid * BLOCK_R + tl.arange(0, BLOCK_R)
    write_mask = idx < output_rows
    actual_tokens = tl.load(query_start_loc_ptr + num_reqs).to(tl.int32)
    token_mask = write_mask & (idx < num_tokens) & (idx < actual_tokens)
    q_idx = tl.where(token_mask, idx, 0).to(tl.int32)

    # Which request owns each token, and the token's logical position.
    req_idx = _upper_bound_req_idx(
        query_start_loc_ptr,
        q_idx,
        num_reqs,
        token_mask,
        SEARCH_STEPS=SEARCH_STEPS,
    )
    req_idx = tl.maximum(tl.minimum(req_idx, num_reqs - 1), 0)
    q_start = tl.load(query_start_loc_ptr + req_idx, mask=token_mask, other=0).to(tl.int32)
    q_end = tl.load(query_start_loc_ptr + req_idx + 1, mask=token_mask, other=0).to(tl.int32)
    seq_len = tl.load(seq_lens_ptr + req_idx, mask=token_mask, other=0).to(tl.int32)
    q_len = q_end - q_start
    q_local = q_idx - q_start
    pos = (seq_len - q_len + q_local).to(tl.int64)   # token_positions

    slot = tl.load(slot_mapping_ptr + idx, mask=token_mask, other=-1).to(tl.int64)

    # flat_region_ids_from_token_slots (position-aware fast path).
    valid = token_mask & (slot >= 0) & (slot < max_slot) & (pos >= 0)
    safe_slot = tl.maximum(tl.minimum(slot, max_slot - 1), 0)
    safe_pos = tl.maximum(pos, 0)
    physical_pages = safe_slot // page_size
    logical_pages = safe_pos // page_size
    logical_regions = safe_pos // region_block_size
    first_region_in_page = (logical_pages * page_size) // region_block_size
    local_fragments = logical_regions - first_region_in_page
    valid = valid & (local_fragments >= 0) & (local_fragments < summaries_per_page)
    frag_clamped = tl.maximum(tl.minimum(local_fragments, summaries_per_page - 1), 0)
    flat_region = physical_pages * summaries_per_page + frag_clamped
    valid = valid & (flat_region < total_regions)
    flat_region = tl.maximum(tl.minimum(flat_region, total_regions - 1), 0)

    # region_start = token_valid & (token_positions % region_block_size == 0)
    rem = safe_pos - (safe_pos // region_block_size) * region_block_size
    region_start = valid & (rem == 0)
    reset = tl.where(region_start, flat_region, tl.full(flat_region.shape, -1, tl.int64))
    flat_region = tl.where(token_mask, flat_region, -1)
    slot = tl.where(token_mask, slot, -1)
    pos = tl.where(token_mask, pos, 0)
    reset = tl.where(token_mask, reset, -1)
    valid = token_mask & valid
    tl.store(flat_slot_ptr + idx, flat_region, mask=write_mask)
    tl.store(token_slots_out_ptr + idx, slot, mask=write_mask)
    tl.store(token_positions_ptr + idx, pos, mask=write_mask)
    tl.store(reset_slots_ptr + idx, reset, mask=write_mask)
    tl.store(token_valid_ptr + idx, valid, mask=write_mask)


def build_decode_summary_layout_step3p5(
    slot_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    num_actual_tokens: int,
    num_pages: int,
    page_size: int,
    region_block_size: int,
    summaries_per_page: int,
    padded_rows: int | None = None,
    out_flat_slot: torch.Tensor | None = None,
    out_token_slots: torch.Tensor | None = None,
    out_token_positions: torch.Tensor | None = None,
    out_reset_slots: torch.Tensor | None = None,
    out_token_valid: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (token_flat_slot, token_slots, token_positions, reset_slots, token_valid).

    All int64 except token_valid (bool), shape [output_rows]. Bit-for-bit
    equivalent to the reference torch layout builder for both decode and prefill.
    FULL CUDA graph rows beyond ``query_start_loc[-1]`` are initialized as
    invalid padding instead of participating in request lookup.

    The five ``out_*`` arguments are optional caller-owned storage. When
    supplied, the kernel writes directly into those buffers so graph refreshes
    and steady-state requests do not allocate transient layout tensors.
    """
    device = slot_mapping.device
    T = int(num_actual_tokens)
    output_rows = max(T, int(padded_rows) if padded_rows is not None else T)
    num_reqs = int(query_start_loc.numel()) - 1
    _validate_search_batch(num_reqs)
    total_regions = int(num_pages) * int(summaries_per_page)
    max_slot = int(num_pages) * int(page_size)

    if int(slot_mapping.numel()) < T:
        raise ValueError(
            "slot_mapping is shorter than num_actual_tokens: "
            f"{int(slot_mapping.numel())} < {T}"
        )
    if int(seq_lens.numel()) < num_reqs:
        raise ValueError(
            "seq_lens is shorter than query_start_loc request count: "
            f"{int(seq_lens.numel())} < {num_reqs}"
        )

    slot_mapping_source = slot_mapping[:T].to(
        device=device, dtype=torch.int64
    ).contiguous()

    def _output_buffer(
        buffer: torch.Tensor | None,
        *,
        dtype: torch.dtype,
        name: str,
    ) -> torch.Tensor:
        if buffer is None:
            return torch.empty((output_rows,), dtype=dtype, device=device)
        if (buffer.device != device or buffer.dtype != dtype
                or buffer.ndim != 1 or int(buffer.numel()) < output_rows
                or not buffer.is_contiguous()):
            raise ValueError(
                f"{name} must be contiguous {dtype} storage with at least "
                f"{output_rows} elements on {device}, got shape={tuple(buffer.shape)}, "
                f"dtype={buffer.dtype}, device={buffer.device}, "
                f"stride={buffer.stride()}."
            )
        return buffer[:output_rows]

    flat_slot = _output_buffer(
        out_flat_slot, dtype=torch.int64, name="out_flat_slot")
    if out_token_slots is None and output_rows == T:
        token_slots = slot_mapping_source
    else:
        token_slots = _output_buffer(
            out_token_slots, dtype=torch.int64, name="out_token_slots")
    token_positions = _output_buffer(
        out_token_positions, dtype=torch.int64, name="out_token_positions")
    reset_slots = _output_buffer(
        out_reset_slots, dtype=torch.int64, name="out_reset_slots")
    token_valid = _output_buffer(
        out_token_valid, dtype=torch.bool, name="out_token_valid")
    if T == 0 or num_reqs <= 0:
        flat_slot.zero_()
        token_slots.fill_(-1)
        token_positions.zero_()
        reset_slots.fill_(-1)
        token_valid.zero_()
        return flat_slot, token_slots, token_positions, reset_slots, token_valid

    qsl = query_start_loc[: num_reqs + 1].to(device=device, dtype=torch.int32).contiguous()
    sl = seq_lens[:num_reqs].to(device=device, dtype=torch.int32).contiguous()

    BLOCK_R = 256
    grid = (triton.cdiv(output_rows, BLOCK_R),)
    _decode_summary_layout_kernel[grid](
        slot_mapping_source,
        qsl,
        sl,
        flat_slot,
        token_slots,
        token_positions,
        reset_slots,
        token_valid,
        T,
        output_rows,
        num_reqs,
        total_regions,
        max_slot,
        page_size=int(page_size),
        region_block_size=int(region_block_size),
        summaries_per_page=int(summaries_per_page),
        BLOCK_R=BLOCK_R,
        SEARCH_STEPS=_SEARCH_STEPS,
        num_warps=4,
    )
    return flat_slot, token_slots, token_positions, reset_slots, token_valid
