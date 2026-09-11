# SPDX-License-Identifier: Apache-2.0
"""Triton kernels for the Ascend Step-4 paged sparse summary cache.

These kernels are the NPU counterparts of the upstream SM90 CuTeDSL operators
(``csa_compact_*_sm90_gqa``); where upstream ships a Triton fallback with the
same contract this file follows its structure. The summary cache is page
backed: ``mean_cache`` is addressed by physical region id
``page * summaries_per_page + fragment`` so prefix-cache hits reuse summaries
through the block table, exactly like upstream.

Semantics pinned against upstream:

* One fp8 (e4m3) summary per *completed* ``region_block_size``-token region,
  written with the fp32 -> bf16 -> e4m3 double rounding of the CuTe kernels.
* A region being filled keeps its state in a fixed-capacity active slot:
  decode stages raw tokens (``active_token_k/z/valid``) and folds them at
  completion; prefill folds immediately and keeps the running
  numerator/denominator/max (online softmax, one scalar max shared by all
  proxy dims of the region). Both forms are consumed by both kernels.
* ``active_region_ids`` is the authoritative slot->region map;
  ``active_slot_by_region`` is a reverse hint that may go stale. Slot
  allocation is a device-side CAS probe; exhaustion publishes
  ``allocation_success=0`` and the caller fails closed.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# Reuse the upstream Triton layout builder unchanged: it is device-agnostic
# (pure Triton + torch) and bit-exact with the reference computation.
from vllm.models.step4.nvidia.ops.cute_dsl.indexer_ops.decode_summary_layout_step3p5 import (  # noqa: E501
    build_decode_summary_layout_step3p5,
)

__all__ = [
    "build_decode_summary_layout_step3p5",
    "csa_compact_decode_update",
    "csa_compact_prefill_update",
    "csa_reset_blocks",
]

# CAS probe window for active-slot allocation. The slot table is sized at
# 8x max_num_seqs and a probe starts at region % capacity; upstream's CuTe
# allocator probes 32 entries, the Triton fast path a fixed window too.
_ALLOC_PROBE = 64


@triton.jit
def _csa_store_fp8_mean(mean_cache_ptr, offset, value, mask):
    """Store an fp32 mean vector through the bf16 -> e4m3 double rounding.

    The CuTe producer rounds fp32 -> bf16 -> e4m3; going straight from fp32
    flips values sitting on an e4m3 midpoint. The parity gate compares
    bytes, so this is not cosmetic.
    """
    fp8_value = value.to(tl.bfloat16).to(tl.float8e4nv)
    tl.store(mean_cache_ptr + offset, fp8_value, mask=mask)


# ---------------------------------------------------------------------------
# Block reset (KV zeroing hook)
# ---------------------------------------------------------------------------


@triton.jit
def _csa_reset_blocks_kernel(
    block_ids,
    mean_cache,
    active_region_ids,
    active_slot_by_region,
    active_numerator,
    denominator,
    max_logits,
    staging_valid,
    num_block_ids,
    num_pages,
    mean_stride_page,
    numerator_stride_slot,
    denominator_stride_slot,
    max_stride_slot,
    SUMMARIES_PER_PAGE: tl.constexpr,
    PROXY_DIM: tl.constexpr,
    ACTIVE_CAPACITY: tl.constexpr,
    REGION_BLOCK_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
) -> None:
    """Reset one summary fragment of one physical page (grid: block, frag).

    Clears the page-backed fp8 payload and the active slot that owns a region
    of the page (if any). The reverse map is only a hint, so slot ownership
    is validated against ``active_region_ids`` before clearing slot state.
    ``blocks_per_scheduler_block`` is 1 on Ascend: the scheduler block id is
    the owner KV page id.
    """
    block_index = tl.program_id(0)
    fragment = tl.program_id(1)
    valid_id = block_index < num_block_ids
    page = tl.load(block_ids + block_index, mask=valid_id, other=-1).to(tl.int64)
    valid = valid_id & (page >= 0) & (page < num_pages)
    safe_page = tl.where(valid, page, 0)

    dims = tl.arange(0, BLOCK_D)
    dim_mask = dims < PROXY_DIM
    tl.store(
        mean_cache + safe_page * mean_stride_page + fragment * PROXY_DIM + dims,
        0,
        mask=valid & dim_mask,
    )

    region = page * SUMMARIES_PER_PAGE + fragment
    safe_region = tl.where(valid, region, 0)
    slot = tl.load(active_slot_by_region + safe_region, mask=valid, other=-1).to(
        tl.int64
    )
    slot_ok = valid & (slot >= 0) & (slot < ACTIVE_CAPACITY)
    safe_slot = tl.where(slot_ok, slot, 0)
    owner = tl.load(active_region_ids + safe_slot, mask=slot_ok, other=-1)
    owns = slot_ok & (owner == region)

    tl.store(active_slot_by_region + safe_region, -1, mask=valid)
    tl.store(active_region_ids + safe_slot, -1, mask=owns)
    tl.store(
        active_numerator + safe_slot * numerator_stride_slot + dims,
        0.0,
        mask=owns & dim_mask,
    )
    tl.store(
        denominator + safe_slot * denominator_stride_slot + dims,
        0.0,
        mask=owns & dim_mask,
    )
    tl.store(max_logits + safe_slot * max_stride_slot, -float("inf"), mask=owns)
    offsets = tl.arange(0, REGION_BLOCK_SIZE)
    tl.store(
        staging_valid + safe_slot * REGION_BLOCK_SIZE + offsets,
        0,
        mask=owns & (offsets < REGION_BLOCK_SIZE),
    )


def csa_reset_blocks(
    block_ids: torch.Tensor,
    mean_cache: torch.Tensor,
    active_region_ids: torch.Tensor,
    active_slot_by_region: torch.Tensor,
    active_numerator: torch.Tensor,
    denominator: torch.Tensor,
    max_logits: torch.Tensor,
    staging_valid: torch.Tensor,
    *,
    num_pages: int,
    summaries_per_page: int,
    proxy_dim: int,
    region_block_size: int,
) -> None:
    """Reset the summary payload and active-slot state of physical pages."""
    if block_ids.numel() == 0:
        return
    _csa_reset_blocks_kernel[(int(block_ids.numel()), int(summaries_per_page))](
        block_ids,
        mean_cache,
        active_region_ids,
        active_slot_by_region,
        active_numerator,
        denominator,
        max_logits,
        staging_valid,
        int(block_ids.numel()),
        int(num_pages),
        int(mean_cache.stride(0)),
        int(active_numerator.stride(0)),
        int(denominator.stride(0)),
        int(max_logits.stride(0)),
        SUMMARIES_PER_PAGE=int(summaries_per_page),
        PROXY_DIM=int(proxy_dim),
        ACTIVE_CAPACITY=int(active_region_ids.numel()),
        REGION_BLOCK_SIZE=int(region_block_size),
        BLOCK_D=triton.next_power_of_2(int(proxy_dim)),
        num_warps=4,
    )


# ---------------------------------------------------------------------------
# Decode summary update (stage raw tokens, flush on region completion)
# ---------------------------------------------------------------------------


@triton.jit
def _csa_decode_stage_flush_kernel(
    mean_cache,
    active_region_ids,
    active_slot_by_region,
    allocation_success,
    active_numerator,
    denominator,
    max_logits,
    staging_k,
    staging_z,
    staging_valid,
    flat_slot,
    reset_slots,
    token_valid,
    token_positions,
    index_k,
    index_z,
    live_rows,
    total_regions,
    mean_stride_page,
    numerator_stride_slot,
    denominator_stride_slot,
    max_stride_slot,
    SUMMARIES_PER_PAGE: tl.constexpr,
    PROXY_DIM: tl.constexpr,
    REGION_BLOCK_SIZE: tl.constexpr,
    ACTIVE_CAPACITY: tl.constexpr,
    ALLOC_PROBE: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
) -> None:
    """One decode token (one row): stage it, or flush the region it completes.

    Grid is (rows,); each row is one request's single new token. Decode rows
    of distinct requests never share a physical region (their tail tokens
    live in privately owned pages), so one program per row is race free.

    Non-completing tokens only enter the staging buffer; the page-backed fp8
    mean is written exactly once, when the region fills, folding the staged
    tokens, the current token, and any prefill-left prefix numerator/
    denominator/max with one shared scalar max.
    """
    row = tl.program_id(0)
    if row >= live_rows:
        return
    valid = tl.load(token_valid + row).to(tl.int1)
    region = tl.load(flat_slot + row)
    region_ok = (region >= 0) & (region < total_regions)
    if ~(valid & region_ok):
        return
    position = tl.load(token_positions + row)
    offset = position % REGION_BLOCK_SIZE

    dims = tl.arange(0, BLOCK_D)
    dim_mask = dims < PROXY_DIM

    # Slot resolve: reverse-map hint validated against the authoritative
    # forward map. The maps are int32; the region id is int64 for address
    # arithmetic, so comparisons go through an explicit int32 view.
    region_i32 = region.to(tl.int32)
    hint = tl.load(active_slot_by_region + region).to(tl.int64)
    hint_ok = (hint >= 0) & (hint < ACTIVE_CAPACITY)
    safe_hint = tl.where(hint_ok, hint, 0)
    hint_owner = tl.load(active_region_ids + safe_hint, mask=hint_ok, other=-1)
    slot = tl.where(hint_ok & (hint_owner == region_i32), hint, ACTIVE_CAPACITY)
    has_slot = slot != ACTIVE_CAPACITY
    safe_slot = tl.where(has_slot, slot, 0)

    prefix_num = tl.load(
        active_numerator + safe_slot * numerator_stride_slot + dims,
        mask=has_slot & dim_mask,
        other=0.0,
    )
    prefix_den = tl.load(
        denominator + safe_slot * denominator_stride_slot + dims,
        mask=has_slot & dim_mask,
        other=0.0,
    )
    prefix_max = tl.load(
        max_logits + safe_slot * max_stride_slot, mask=has_slot, other=-float("inf")
    ).to(tl.float32)

    # Region restart: a token at offset 0 resets all state for this region,
    # including the staged tail of an abandoned generation.
    reset = tl.load(reset_slots + row) == region
    prefix_num = tl.where(reset, 0.0, prefix_num)
    prefix_den = tl.where(reset, 0.0, prefix_den)
    prefix_max = tl.where(reset, -float("inf"), prefix_max)
    if reset:
        toks = tl.arange(0, BLOCK_T)
        tl.store(
            staging_valid + safe_slot * REGION_BLOCK_SIZE + toks,
            0,
            mask=has_slot & (toks < REGION_BLOCK_SIZE),
        )
        _csa_store_fp8_mean(
            mean_cache
            + (region // SUMMARIES_PER_PAGE) * mean_stride_page
            + (region % SUMMARIES_PER_PAGE) * PROXY_DIM,
            dims,
            tl.zeros((BLOCK_D,), dtype=tl.float32),
            dim_mask,
        )

    cur_k = tl.load(index_k + row * PROXY_DIM + dims, mask=dim_mask, other=0.0).to(
        tl.float32
    )
    cur_z = tl.load(index_z + row * PROXY_DIM + dims, mask=dim_mask, other=0.0).to(
        tl.float32
    )

    if offset == REGION_BLOCK_SIZE - 1:
        # Region completes: fold staged tokens [0, REGION_BLOCK_SIZE-1), the
        # current token, and the prefill prefix with one shared scalar max.
        toks = tl.arange(0, BLOCK_T)
        staged_ok = (
            tl.load(
                staging_valid + safe_slot * REGION_BLOCK_SIZE + toks,
                mask=has_slot & (toks < REGION_BLOCK_SIZE - 1),
                other=0,
            )
            != 0
        )
        staged_z = tl.load(
            staging_z
            + safe_slot * (REGION_BLOCK_SIZE * PROXY_DIM)
            + toks[:, None] * PROXY_DIM
            + dims[None, :],
            mask=staged_ok[:, None] & dim_mask[None, :],
            other=-float("inf"),
        ).to(tl.float32)
        staged_k = tl.load(
            staging_k
            + safe_slot * (REGION_BLOCK_SIZE * PROXY_DIM)
            + toks[:, None] * PROXY_DIM
            + dims[None, :],
            mask=staged_ok[:, None] & dim_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        has_prefix = tl.max(prefix_den) > 0.0
        base_max = tl.where(has_prefix, prefix_max, -float("inf"))
        shared_max = tl.maximum(
            base_max,
            tl.maximum(
                tl.max(tl.where(staged_ok[:, None], staged_z, -float("inf"))),
                tl.max(cur_z),
            ),
        )
        scale = tl.where(has_prefix, tl.exp(prefix_max - shared_max), 0.0)
        staged_w = tl.where(staged_ok[:, None], tl.exp(staged_z - shared_max), 0.0)
        cur_w = tl.exp(cur_z - shared_max)
        numerator = prefix_num * scale + tl.sum(staged_w * staged_k, axis=0) + (
            cur_w * cur_k
        )
        denominator_acc = prefix_den * scale + tl.sum(staged_w, axis=0) + cur_w
        mean = tl.where(
            denominator_acc > 0.0,
            numerator / tl.maximum(denominator_acc, 1.0e-20),
            0.0,
        )
        _csa_store_fp8_mean(
            mean_cache
            + (region // SUMMARIES_PER_PAGE) * mean_stride_page
            + (region % SUMMARIES_PER_PAGE) * PROXY_DIM,
            dims,
            mean,
            dim_mask,
        )
        # Release the slot: completed regions live only in the page cache.
        tl.store(active_slot_by_region + region, -1)
        if has_slot:
            tl.store(active_region_ids + safe_slot, -1)
            tl.store(
                active_numerator + safe_slot * numerator_stride_slot + dims,
                0.0,
                mask=dim_mask,
            )
            tl.store(
                denominator + safe_slot * denominator_stride_slot + dims,
                0.0,
                mask=dim_mask,
            )
            tl.store(max_logits + safe_slot * max_stride_slot, -float("inf"))
            tl.store(
                staging_valid + safe_slot * REGION_BLOCK_SIZE + toks,
                0,
                mask=toks < REGION_BLOCK_SIZE,
            )
        return

    # Non-completing token: stage it, allocating a slot on first touch.
    if ~has_slot:
        start = region % ACTIVE_CAPACITY
        empty = tl.full((), -1, tl.int32)
        for probe in tl.range(0, ALLOC_PROBE):
            if slot == ACTIVE_CAPACITY:
                cand = (start + probe) % ACTIVE_CAPACITY
                old = tl.atomic_cas(active_region_ids + cand, empty, region_i32)
                if (old == empty) | (old == region_i32):
                    slot = cand
        if slot == ACTIVE_CAPACITY:
            tl.store(allocation_success, 0)
    slot_ok = (slot >= 0) & (slot < ACTIVE_CAPACITY)
    safe_slot = tl.where(slot_ok, slot, 0)
    if slot_ok:
        tl.store(active_slot_by_region + region, slot.to(tl.int32))
        tl.store(
            staging_k
            + safe_slot * (REGION_BLOCK_SIZE * PROXY_DIM)
            + offset * PROXY_DIM
            + dims,
            cur_k.to(staging_k.dtype.element_ty),
            mask=dim_mask,
        )
        tl.store(
            staging_z
            + safe_slot * (REGION_BLOCK_SIZE * PROXY_DIM)
            + offset * PROXY_DIM
            + dims,
            cur_z.to(staging_z.dtype.element_ty),
            mask=dim_mask,
        )
        tl.store(staging_valid + safe_slot * REGION_BLOCK_SIZE + offset, 1)


def csa_compact_decode_update(
    mean_cache: torch.Tensor,
    active_region_ids: torch.Tensor,
    active_slot_by_region: torch.Tensor,
    allocation_success: torch.Tensor,
    active_numerator: torch.Tensor,
    denominator: torch.Tensor,
    max_logits: torch.Tensor,
    staging_k: torch.Tensor,
    staging_z: torch.Tensor,
    staging_valid: torch.Tensor,
    flat_slot: torch.Tensor,
    reset_slots: torch.Tensor,
    token_valid: torch.Tensor,
    token_positions: torch.Tensor,
    index_k: torch.Tensor,
    index_z: torch.Tensor,
    *,
    live_rows: int,
    num_pages: int,
    summaries_per_page: int,
    proxy_dim: int,
    region_block_size: int,
) -> None:
    """Fold one decode token per row into the paged summary state."""
    rows = int(flat_slot.numel())
    if rows == 0 or live_rows <= 0:
        return
    _csa_decode_stage_flush_kernel[(rows,)](
        mean_cache,
        active_region_ids,
        active_slot_by_region,
        allocation_success,
        active_numerator,
        denominator,
        max_logits,
        staging_k,
        staging_z,
        staging_valid,
        flat_slot,
        reset_slots,
        token_valid,
        token_positions,
        index_k,
        index_z,
        int(live_rows),
        int(num_pages) * int(summaries_per_page),
        int(mean_cache.stride(0)),
        int(active_numerator.stride(0)),
        int(denominator.stride(0)),
        int(max_logits.stride(0)),
        SUMMARIES_PER_PAGE=int(summaries_per_page),
        PROXY_DIM=int(proxy_dim),
        REGION_BLOCK_SIZE=int(region_block_size),
        ACTIVE_CAPACITY=int(active_region_ids.numel()),
        ALLOC_PROBE=_ALLOC_PROBE,
        BLOCK_T=triton.next_power_of_2(int(region_block_size)),
        BLOCK_D=triton.next_power_of_2(int(proxy_dim)),
        num_warps=4,
    )


# ---------------------------------------------------------------------------
# Prefill / mixed-batch summary update (two-pass ordered fold)
# ---------------------------------------------------------------------------


@triton.jit
def _csa_prefill_clear_scratch_kernel(
    scratch_region_ids,
    scratch_row_map,
    scratch_reset_map,
    scratch_rows,
    total_items,
    REGION_BLOCK_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
) -> None:
    pid = tl.program_id(0)
    offsets = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    tl.store(scratch_row_map + offsets, -1, mask=offsets < total_items)
    row_mask = offsets < scratch_rows
    tl.store(scratch_region_ids + offsets, -1, mask=row_mask)
    tl.store(scratch_reset_map + offsets, 0, mask=row_mask)


@triton.jit
def _csa_prefill_fill_scratch_kernel(
    scratch_region_ids,
    scratch_row_map,
    scratch_reset_map,
    flat_slot,
    reset_slots,
    token_valid,
    token_positions,
    live_rows,
    total_regions,
    scratch_rows,
    REGION_BLOCK_SIZE: tl.constexpr,
) -> None:
    """Hash each token row to its region's scratch slot (CAS, linear probe)."""
    row = tl.program_id(0)
    if row >= live_rows:
        return
    valid = tl.load(token_valid + row).to(tl.int1)
    region = tl.load(flat_slot + row)
    if ~(valid & (region >= 0) & (region < total_regions)):
        return
    region_i32 = region.to(tl.int32)
    found = scratch_rows
    start_slot = region_i32 % scratch_rows
    # Bounded probe: a ``while`` on the CAS result miscompiles on
    # triton-ascend when the surrounding launches run back to back.
    for probe in tl.range(0, scratch_rows):
        if found == scratch_rows:
            cand = (start_slot + probe) % scratch_rows
            old = tl.atomic_cas(scratch_region_ids + cand, -1, region_i32)
            found = tl.where((old == -1) | (old == region_i32), cand, found)
    if found == scratch_rows:
        return
    position = tl.load(token_positions + row)
    offset = position % REGION_BLOCK_SIZE
    if (offset >= 0) & (offset < REGION_BLOCK_SIZE):
        tl.store(scratch_row_map + found * REGION_BLOCK_SIZE + offset, row)
    if tl.load(reset_slots + row) == region:
        tl.store(scratch_reset_map + found, 1)


@triton.jit
def _csa_prefill_update_ordered_kernel(
    mean_cache,
    active_region_ids,
    active_slot_by_region,
    allocation_success,
    active_numerator,
    denominator,
    max_logits,
    staging_valid,
    scratch_region_ids,
    scratch_row_map,
    scratch_reset_map,
    index_k,
    index_z,
    live_rows,
    total_regions,
    mean_stride_page,
    numerator_stride_slot,
    denominator_stride_slot,
    max_stride_slot,
    SUMMARIES_PER_PAGE: tl.constexpr,
    PROXY_DIM: tl.constexpr,
    REGION_BLOCK_SIZE: tl.constexpr,
    ACTIVE_CAPACITY: tl.constexpr,
    PROCESS_COMPLETE: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
) -> None:
    """Fold one scratch region group per program.

    Runs twice per update (PROCESS_COMPLETE True then False): completed
    regions release their slots in the first pass so the second pass's
    allocations can reuse them without a grid-wide barrier. Unlike the
    decode path there is no staging: a region's tokens are folded into the
    slot's numerator/denominator/max immediately, with one shared scalar
    max over the region's tokens and the prefix.
    """
    group_slot = tl.program_id(0)
    region = tl.load(scratch_region_ids + group_slot).to(tl.int64)
    if region < 0:
        return
    dims = tl.arange(0, BLOCK_D)
    dim_mask = dims < PROXY_DIM
    toks = tl.arange(0, BLOCK_T)
    tok_mask = toks < REGION_BLOCK_SIZE
    rows = tl.load(
        scratch_row_map + group_slot * REGION_BLOCK_SIZE + toks,
        mask=tok_mask,
        other=-1,
    )
    if tl.max(rows) < 0:
        return
    last_row = tl.load(
        scratch_row_map + group_slot * REGION_BLOCK_SIZE + (REGION_BLOCK_SIZE - 1)
    )
    complete = last_row >= 0
    if complete != PROCESS_COMPLETE:
        return

    # Slot resolve: reverse-map hint validated against the forward map.
    # The maps are int32; the region id is int64 for address arithmetic.
    region_i32 = region.to(tl.int32)
    hint = tl.load(active_slot_by_region + region).to(tl.int64)
    hint_ok = (hint >= 0) & (hint < ACTIVE_CAPACITY)
    safe_hint = tl.where(hint_ok, hint, 0)
    hint_owner = tl.load(active_region_ids + safe_hint, mask=hint_ok, other=-1)
    slot = tl.where(hint_ok & (hint_owner == region_i32), hint, ACTIVE_CAPACITY)
    has_slot = slot != ACTIVE_CAPACITY
    safe_slot = tl.where(has_slot, slot, 0)

    prefix_num = tl.load(
        active_numerator + safe_slot * numerator_stride_slot + dims,
        mask=has_slot & dim_mask,
        other=0.0,
    )
    prefix_den = tl.load(
        denominator + safe_slot * denominator_stride_slot + dims,
        mask=has_slot & dim_mask,
        other=0.0,
    )
    prefix_max = tl.load(
        max_logits + safe_slot * max_stride_slot, mask=has_slot, other=-float("inf")
    ).to(tl.float32)

    reset = tl.load(scratch_reset_map + group_slot) != 0
    prefix_num = tl.where(reset, 0.0, prefix_num)
    prefix_den = tl.where(reset, 0.0, prefix_den)
    prefix_max = tl.where(reset, -float("inf"), prefix_max)
    if reset:
        tl.store(
            staging_valid + safe_slot * REGION_BLOCK_SIZE + toks,
            0,
            mask=has_slot & tok_mask,
        )

    valid_row = (rows >= 0) & (rows < live_rows) & tok_mask
    safe_rows = tl.where(valid_row, rows, 0)
    values = tl.load(
        index_k + safe_rows[:, None] * PROXY_DIM + dims[None, :],
        mask=valid_row[:, None] & dim_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    logits = tl.load(
        index_z + safe_rows[:, None] * PROXY_DIM + dims[None, :],
        mask=valid_row[:, None] & dim_mask[None, :],
        other=-float("inf"),
    ).to(tl.float32)

    has_prefix = tl.max(prefix_den) > 0.0
    base_max = tl.where(has_prefix, prefix_max, -float("inf"))
    shared_max = tl.maximum(
        base_max, tl.max(tl.where(valid_row[:, None], logits, -float("inf")))
    )
    scale = tl.where(has_prefix, tl.exp(prefix_max - shared_max), 0.0)
    weights = tl.where(valid_row[:, None], tl.exp(logits - shared_max), 0.0)
    numerator = prefix_num * scale + tl.sum(weights * values, axis=0)
    denominator_acc = prefix_den * scale + tl.sum(weights, axis=0)
    mean = tl.where(
        denominator_acc > 0.0,
        numerator / tl.maximum(denominator_acc, 1.0e-20),
        0.0,
    )

    if complete:
        _csa_store_fp8_mean(
            mean_cache
            + (region // SUMMARIES_PER_PAGE) * mean_stride_page
            + (region % SUMMARIES_PER_PAGE) * PROXY_DIM,
            dims,
            mean,
            dim_mask,
        )
        tl.store(active_slot_by_region + region, -1)
        if has_slot:
            tl.store(active_region_ids + safe_slot, -1)
            tl.store(
                active_numerator + safe_slot * numerator_stride_slot + dims,
                0.0,
                mask=dim_mask,
            )
            tl.store(
                denominator + safe_slot * denominator_stride_slot + dims,
                0.0,
                mask=dim_mask,
            )
            tl.store(max_logits + safe_slot * max_stride_slot, -float("inf"))
        return

    # Incomplete region: allocate a slot on first touch and keep the running
    # numerator/denominator/max for the next chunk (or the decode flush).
    if ~has_slot:
        empty = tl.full((), -1, tl.int32)
        for cand in tl.range(0, ACTIVE_CAPACITY):
            if slot == ACTIVE_CAPACITY:
                old = tl.atomic_cas(active_region_ids + cand, empty, region_i32)
                if (old == empty) | (old == region_i32):
                    slot = cand.to(tl.int64)
        if slot == ACTIVE_CAPACITY:
            tl.store(allocation_success, 0)
    slot_ok = (slot >= 0) & (slot < ACTIVE_CAPACITY)
    safe_slot = tl.where(slot_ok, slot, 0)
    if slot_ok:
        tl.store(active_slot_by_region + region, slot.to(tl.int32))
        tl.store(
            active_numerator + safe_slot * numerator_stride_slot + dims,
            numerator,
            mask=dim_mask,
        )
        tl.store(
            denominator + safe_slot * denominator_stride_slot + dims,
            denominator_acc,
            mask=dim_mask,
        )
        tl.store(max_logits + safe_slot * max_stride_slot, shared_max)


def csa_compact_prefill_update(
    mean_cache: torch.Tensor,
    active_region_ids: torch.Tensor,
    active_slot_by_region: torch.Tensor,
    allocation_success: torch.Tensor,
    active_numerator: torch.Tensor,
    denominator: torch.Tensor,
    max_logits: torch.Tensor,
    staging_valid: torch.Tensor,
    scratch_region_ids: torch.Tensor,
    scratch_row_map: torch.Tensor,
    scratch_reset_map: torch.Tensor,
    flat_slot: torch.Tensor,
    reset_slots: torch.Tensor,
    token_valid: torch.Tensor,
    token_positions: torch.Tensor,
    index_k: torch.Tensor,
    index_z: torch.Tensor,
    *,
    live_rows: int,
    num_pages: int,
    summaries_per_page: int,
    proxy_dim: int,
    region_block_size: int,
) -> None:
    """Fold a packed token chunk (prefill or mixed batch) into the summary state."""
    rows = int(flat_slot.numel())
    live_rows = int(live_rows)
    if rows == 0 or live_rows <= 0:
        return
    scratch_rows = rows
    total_items = scratch_rows * int(region_block_size)
    clear_block = 256
    _csa_prefill_clear_scratch_kernel[
        (triton.cdiv(max(total_items, scratch_rows), clear_block),)
    ](
        scratch_region_ids,
        scratch_row_map,
        scratch_reset_map,
        scratch_rows,
        total_items,
        REGION_BLOCK_SIZE=int(region_block_size),
        BLOCK_N=clear_block,
    )
    _csa_prefill_fill_scratch_kernel[(rows,)](
        scratch_region_ids,
        scratch_row_map,
        scratch_reset_map,
        flat_slot,
        reset_slots,
        token_valid,
        token_positions,
        live_rows,
        int(num_pages) * int(summaries_per_page),
        scratch_rows,
        REGION_BLOCK_SIZE=int(region_block_size),
    )
    for process_complete in (True, False):
        _csa_prefill_update_ordered_kernel[(scratch_rows,)](
            mean_cache,
            active_region_ids,
            active_slot_by_region,
            allocation_success,
            active_numerator,
            denominator,
            max_logits,
            staging_valid,
            scratch_region_ids,
            scratch_row_map,
            scratch_reset_map,
            index_k,
            index_z,
            live_rows,
            int(num_pages) * int(summaries_per_page),
            int(mean_cache.stride(0)),
            int(active_numerator.stride(0)),
            int(denominator.stride(0)),
            int(max_logits.stride(0)),
            SUMMARIES_PER_PAGE=int(summaries_per_page),
            PROXY_DIM=int(proxy_dim),
            REGION_BLOCK_SIZE=int(region_block_size),
            ACTIVE_CAPACITY=int(active_region_ids.numel()),
            PROCESS_COMPLETE=process_complete,
            BLOCK_T=triton.next_power_of_2(int(region_block_size)),
            BLOCK_D=triton.next_power_of_2(int(proxy_dim)),
            num_warps=4,
        )

