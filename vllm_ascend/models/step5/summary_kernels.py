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
  ``active_slot_by_region`` is a reverse hint that may go stale. Prompt
  default prefill and decode/verify use the original device allocator;
  runner prefill uses a separate reserved pool. Allocation failure publishes
  ``allocation_success=0`` and the caller fails closed.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from vllm_ascend.worker.step5_csa_slot_manager import CSAMetadata

# Reuse the upstream Triton layout builder unchanged: it is device-agnostic
# (pure Triton + torch) and bit-exact with the reference computation.
from vllm.models.step5.nvidia.ops.cute_dsl.indexer_ops.decode_summary_layout_step3p5 import (  # noqa: E501
    build_decode_summary_layout_step3p5,
)

__all__ = [
    "build_decode_summary_layout_step3p5",
    "csa_compact_decode_update",
    "csa_compact_prefill_update",
    "csa_reset_blocks",
]


@triton.jit
def _csa_store_fp8_mean(mean_cache_ptr, offset, value, mask):
    """Store an fp32 mean vector through the bf16 -> e4m3 double rounding.

    The CuTe producer rounds fp32 -> bf16 -> e4m3; going straight from fp32
    flips values sitting on an e4m3 midpoint. The parity gate compares
    bytes, so this is not cosmetic.
    """
    fp8_value = value.to(tl.bfloat16).to(tl.float8e4nv)
    tl.store(mean_cache_ptr + offset, fp8_value, mask=mask)


def _get_num_vector_cores() -> int:
    """Return the number of physical vector cores on the current NPU device."""
    try:
        device = torch.npu.current_device()
        props = triton.runtime.driver.active.utils.get_device_properties(device)
        n = props.get("num_vectorcore", -1)
        if n > 0:
            return int(n)
    except Exception:
        pass
    return 64


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
    CLEAR_MEAN: tl.constexpr,
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
    if CLEAR_MEAN:
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
    clear_mean: bool = True,
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
        CLEAR_MEAN=bool(clear_mean),
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
        ALLOC_PROBE=int(active_region_ids.numel()),
        BLOCK_T=triton.next_power_of_2(int(region_block_size)),
        BLOCK_D=triton.next_power_of_2(int(proxy_dim)),
        num_warps=4,
    )


# ---------------------------------------------------------------------------
# Prefill / mixed-batch summary update (slot preparation + single update)
# ---------------------------------------------------------------------------


@triton.jit
def _csa_prefill_clear_scratch_kernel(
    scratch_region_ids,
    scratch_row_map,
    scratch_reset_map,
    unique_regions,
    scratch_rows,
    total_items,
    REGION_BLOCK_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
) -> None:
    pid = tl.program_id(0)
    offsets = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    tl.store(scratch_row_map + offsets, -1, mask=offsets < total_items)
    row_mask = offsets < scratch_rows
    region_vals = tl.load(unique_regions + offsets, mask=row_mask, other=0).to(tl.int32)
    tl.store(scratch_region_ids + offsets, region_vals, mask=row_mask)
    tl.store(scratch_reset_map + offsets, 0, mask=row_mask)


@triton.jit
def _csa_prefill_fill_direct_kernel(
    scratch_row_map,
    scratch_reset_map,
    inverse_indices,
    flat_slot,
    reset_slots,
    token_valid,
    token_positions,
    live_rows,
    REGION_BLOCK_SIZE: tl.constexpr,
) -> None:
    """Direct-index fill: each token writes to its group via inverse_indices.

    Replaces the O(n^2) CAS hash-table with O(n) direct stores. The
    ``inverse_indices`` tensor (from ``torch.unique``) maps each token to
    its de-duplicated group slot, eliminating all atomic operations and
    linear probing.
    """
    row = tl.program_id(0)
    if row >= live_rows:
        return
    valid = tl.load(token_valid + row).to(tl.int1)
    if ~valid:
        return
    group_idx = tl.load(inverse_indices + row).to(tl.int64)
    position = tl.load(token_positions + row)
    offset = position % REGION_BLOCK_SIZE
    if (offset >= 0) & (offset < REGION_BLOCK_SIZE):
        tl.store(scratch_row_map + group_idx * REGION_BLOCK_SIZE + offset, row)
    region = tl.load(flat_slot + row)
    if tl.load(reset_slots + row) == region:
        tl.store(scratch_reset_map + group_idx, 1)


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
    num_tasks,
    NUM_CORES: tl.constexpr,
    MEAN_STRIDE_PAGE: tl.constexpr,
    NUMERATOR_STRIDE_SLOT: tl.constexpr,
    DENOMINATOR_STRIDE_SLOT: tl.constexpr,
    MAX_STRIDE_SLOT: tl.constexpr,
    SUMMARIES_PER_PAGE: tl.constexpr,
    PROXY_DIM: tl.constexpr,
    REGION_BLOCK_SIZE: tl.constexpr,
    active_capacity,
    PROCESS_COMPLETE: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
) -> None:
    """Fold scratch region groups with grid collapsed to physical cores.

    Each core iterates over multiple region groups via a strided loop,
    eliminating the overhead of launching num_unique programs (which can
    far exceed the physical vector core count).
    """
    pid = tl.program_id(0)
    dims = tl.arange(0, BLOCK_D)
    dim_mask = dims < PROXY_DIM
    toks = tl.arange(0, BLOCK_T)
    tok_mask = toks < REGION_BLOCK_SIZE

    for group_slot in range(pid, num_tasks, NUM_CORES):
        region = tl.load(scratch_region_ids + group_slot).to(tl.int64)
        if region >= 0:
            rows = tl.load(
                scratch_row_map + group_slot * REGION_BLOCK_SIZE + toks,
                mask=tok_mask,
                other=-1,
            )
            if tl.max(rows) >= 0:
                last_row = tl.load(
                    scratch_row_map + group_slot * REGION_BLOCK_SIZE + (REGION_BLOCK_SIZE - 1)
                )
                complete = last_row >= 0
                if complete == PROCESS_COMPLETE:
                    region_i32 = region.to(tl.int32)
                    hint = tl.load(active_slot_by_region + region).to(tl.int64)
                    hint_ok = (hint >= 0) & (hint < active_capacity)
                    safe_hint = tl.where(hint_ok, hint, 0)
                    hint_owner = tl.load(active_region_ids + safe_hint, mask=hint_ok, other=-1)
                    slot = tl.where(hint_ok & (hint_owner == region_i32), hint, active_capacity)
                    has_slot = slot != active_capacity
                    safe_slot = tl.where(has_slot, slot, 0)

                    prefix_num = tl.load(
                        active_numerator + safe_slot * NUMERATOR_STRIDE_SLOT + dims,
                        mask=has_slot & dim_mask,
                        other=0.0,
                    )
                    prefix_den = tl.load(
                        denominator + safe_slot * DENOMINATOR_STRIDE_SLOT + dims,
                        mask=has_slot & dim_mask,
                        other=0.0,
                    )
                    prefix_max = tl.load(
                        max_logits + safe_slot * MAX_STRIDE_SLOT, mask=has_slot, other=-float("inf")
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
                            + (region // SUMMARIES_PER_PAGE) * MEAN_STRIDE_PAGE
                            + (region % SUMMARIES_PER_PAGE) * PROXY_DIM,
                            dims,
                            mean,
                            dim_mask,
                        )
                        tl.store(active_slot_by_region + region, -1)
                        if has_slot:
                            tl.store(active_region_ids + safe_slot, -1)
                            tl.store(
                                active_numerator + safe_slot * NUMERATOR_STRIDE_SLOT + dims,
                                0.0,
                                mask=dim_mask,
                            )
                            tl.store(
                                denominator + safe_slot * DENOMINATOR_STRIDE_SLOT + dims,
                                0.0,
                                mask=dim_mask,
                            )
                            tl.store(max_logits + safe_slot * MAX_STRIDE_SLOT, -float("inf"))
                    else:
                        if ~has_slot:
                            empty = tl.full((), -1, tl.int32)
                            for cand in tl.range(0, active_capacity):
                                if slot == active_capacity:
                                    old = tl.atomic_cas(active_region_ids + cand, empty, region_i32)
                                    if (old == empty) | (old == region_i32):
                                        slot = cand.to(tl.int64)
                        if slot == active_capacity:
                            tl.store(allocation_success, 0)
                        slot_ok = (slot >= 0) & (slot < active_capacity)
                        safe_slot = tl.where(slot_ok, slot, 0)
                        if slot_ok:
                            tl.store(active_slot_by_region + region, slot.to(tl.int32))
                            tl.store(
                                active_numerator + safe_slot * NUMERATOR_STRIDE_SLOT + dims,
                                numerator,
                                mask=dim_mask,
                            )
                            tl.store(
                                denominator + safe_slot * DENOMINATOR_STRIDE_SLOT + dims,
                                denominator_acc,
                                mask=dim_mask,
                            )
                            tl.store(max_logits + safe_slot * MAX_STRIDE_SLOT, shared_max)


def csa_compact_prefill_update_default(
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
    fill_compile_mode: str = "unstructured_in_simt",
) -> None:
    """Fold a packed token chunk (prefill or mixed batch) into the summary state.

    Uses ``torch.unique`` to de-duplicate region ids and produce an
    inverse-index, replacing the original O(n^2) CAS hash-table with an
    O(n) direct-index fill. The update kernel grid shrinks from
    ``num_tokens`` to ``num_unique_regions``, eliminating idle programs.
    """
    rows = int(flat_slot.numel())
    live_rows = int(live_rows)
    if rows == 0 or live_rows <= 0:
        return

    # Step 1: torch.unique — de-duplicate region ids + inverse index
    unique_regions, inverse_indices = torch.unique(
        flat_slot[:live_rows], return_inverse=True
    )
    num_unique = int(unique_regions.numel())

    # Step 2: clear scratch buffers + write unique region ids (fused in kernel)
    total_items = num_unique * int(region_block_size)
    clear_block = 256
    _csa_prefill_clear_scratch_kernel[
        (triton.cdiv(max(total_items, num_unique), clear_block),)
    ](
        scratch_region_ids,
        scratch_row_map,
        scratch_reset_map,
        unique_regions,
        num_unique,
        total_items,
        REGION_BLOCK_SIZE=int(region_block_size),
        BLOCK_N=clear_block,
    )

    # Step 3: direct-index fill — no CAS, no probing (O(n))
    _csa_prefill_fill_direct_kernel[(rows,)](
        scratch_row_map,
        scratch_reset_map,
        inverse_indices,
        flat_slot,
        reset_slots,
        token_valid,
        token_positions,
        live_rows,
        REGION_BLOCK_SIZE=int(region_block_size),
        compile_mode=fill_compile_mode,
    )

    # Step 4: ordered fold — two-pass (complete then incomplete), grid = num_cores
    num_cores = _get_num_vector_cores()
    for process_complete in (True, False):
        _csa_prefill_update_ordered_kernel[(num_cores,)](
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
            num_tasks=num_unique,
            NUM_CORES=num_cores,
            MEAN_STRIDE_PAGE=int(mean_cache.stride(0)),
            NUMERATOR_STRIDE_SLOT=int(active_numerator.stride(0)),
            DENOMINATOR_STRIDE_SLOT=int(denominator.stride(0)),
            MAX_STRIDE_SLOT=int(max_logits.stride(0)),
            SUMMARIES_PER_PAGE=int(summaries_per_page),
            PROXY_DIM=int(proxy_dim),
            REGION_BLOCK_SIZE=int(region_block_size),
            active_capacity=int(active_region_ids.numel()),
            PROCESS_COMPLETE=process_complete,
            BLOCK_T=triton.next_power_of_2(int(region_block_size)),
            BLOCK_D=triton.next_power_of_2(int(proxy_dim)),
            num_warps=4,
        )


# ---------------------------------------------------------------------------
# MTP verify batches: staged tokens + deferred region flush
# ---------------------------------------------------------------------------


@triton.jit
def _csa_prefill_task_rows(
    request, ordinal, query_start_loc, seq_lens,
    flat_slot, token_valid, token_positions, live_rows, total_regions,
    REGION_BLOCK_SIZE: tl.constexpr, BLOCK_T: tl.constexpr,
):
    # 请求和 region 序号由二维 grid 直接给出。
    row_base = tl.load(query_start_loc)
    query_start = tl.load(query_start_loc + request) - row_base
    query_end = tl.load(query_start_loc + request + 1) - row_base
    query_length = query_end - query_start
    context_length = tl.load(seq_lens + request) - query_length
    context_offset = context_length % REGION_BLOCK_SIZE
    # 第一个任务从本次请求首行开始；后续任务跨过首个 region 的剩余位置。
    start_row = query_start + tl.where(
        ordinal == 0,
        0,
        REGION_BLOCK_SIZE - context_offset + (ordinal - 1) * REGION_BLOCK_SIZE,
    )
    start_ok = (start_row >= 0) & (start_row < query_end) & (start_row < live_rows)
    region = tl.load(flat_slot + start_row, mask=start_ok, other=-1).to(tl.int64)
    offsets = tl.arange(0, BLOCK_T)
    rows = start_row + offsets
    span = tl.where(ordinal == 0, REGION_BLOCK_SIZE - context_offset, REGION_BLOCK_SIZE)
    region_ok = start_ok & (region >= 0) & (region < total_regions)
    in_bounds = region_ok & (offsets < span) & (offsets < REGION_BLOCK_SIZE)
    in_bounds = in_bounds & (rows >= 0) & (rows < query_end) & (rows < live_rows)
    row_regions = tl.load(flat_slot + rows, mask=in_bounds, other=-1)
    in_region = in_bounds & (row_regions == region)
    valid_row = in_region & tl.load(token_valid + rows, mask=in_region, other=0).to(tl.int1)
    positions = tl.load(token_positions + rows, mask=valid_row, other=0)
    complete = tl.sum(
        (valid_row & (positions % REGION_BLOCK_SIZE == REGION_BLOCK_SIZE - 1)).to(tl.int32), axis=0
    ) > 0
    return region, rows, valid_row, complete


@triton.jit
def _csa_prefill_update_kernel(
    mean_cache,
    active_region_ids,
    active_slot_by_region,
    allocation_success,
    active_numerator,
    denominator,
    max_logits,
    staging_valid,
    flat_slot,
    reset_slots,
    token_valid,
    token_positions,
    index_k,
    index_z,
    query_start_loc,
    seq_lens,
    region_count,
    planned_targets, planned_sources,
    num_requests,
    live_rows,
    total_regions,
    MEAN_STRIDE_PAGE: tl.constexpr,
    NUMERATOR_STRIDE_SLOT: tl.constexpr,
    DENOMINATOR_STRIDE_SLOT: tl.constexpr,
    MAX_STRIDE_SLOT: tl.constexpr,
    SUMMARIES_PER_PAGE: tl.constexpr,
    PROXY_DIM: tl.constexpr,
    REGION_BLOCK_SIZE: tl.constexpr,
    active_capacity,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
) -> None:
    """Fold all prepared region tasks in one launch without slot allocation.

    Query rows must follow contiguous sequence positions within each request.
    Each physical region must have only one updating task in the batch.
    """
    # 每个执行单元处理一个请求的一个 region，短请求的多余任务直接返回。
    request = tl.program_id(0)
    ordinal = tl.program_id(1)
    if request >= num_requests:
        return
    request_region_count = tl.load(region_count + request)
    if ordinal >= request_region_count:
        return
    dims = tl.arange(0, BLOCK_D)
    dim_mask = dims < PROXY_DIM
    toks = tl.arange(0, BLOCK_T)
    tok_mask = toks < REGION_BLOCK_SIZE

    # 同批次前序状态更新已报错时，停止继续写入。
    if tl.load(allocation_success) == 0:
        return
    region, rows, valid_row, complete = _csa_prefill_task_rows(
        request, ordinal, query_start_loc, seq_lens,
        flat_slot, token_valid, token_positions, live_rows, total_regions,
        REGION_BLOCK_SIZE, BLOCK_T,
    )
    if tl.sum(valid_row.to(tl.int32), axis=0) > 0:
        # 反向映射只是 slot 位置提示，还要核对 slot 的实际拥有者。
        # 无有效 slot 时使用安全地址，后续通过 mask 禁止读取旧状态。
        region_i32 = region.to(tl.int32)
        hint = tl.load(active_slot_by_region + region).to(tl.int64)
        hint_ok = (hint >= 0) & (hint < active_capacity)
        safe_hint = tl.where(hint_ok, hint, 0)
        hint_owner = tl.load(active_region_ids + safe_hint, mask=hint_ok, other=-1)
        slot = tl.where(hint_ok & (hint_owner == region_i32), hint, active_capacity)
        # Runner knows the old reserved address even if the reverse hint
        # is stale. Confirm physical ownership before reading any state.
        planned_source = tl.load(planned_sources + request, mask=ordinal == 0, other=-1).to(tl.int64)
        source_ok = (planned_source >= 0) & (planned_source < active_capacity)
        source_owner = tl.load(active_region_ids + tl.where(source_ok, planned_source, 0),
                               mask=source_ok, other=-1)
        slot = tl.where(source_ok & (source_owner == region_i32), planned_source, slot)
        has_slot = slot != active_capacity
        safe_slot = tl.where(has_slot, slot, 0)

        # 读取跨批次保留的分子、分母和最大分数；没有旧状态时从零累计。
        prefix_num = tl.load(
            active_numerator + safe_slot * NUMERATOR_STRIDE_SLOT + dims,
            mask=has_slot & dim_mask,
            other=0.0,
        )
        prefix_den = tl.load(
            denominator + safe_slot * DENOMINATOR_STRIDE_SLOT + dims,
            mask=has_slot & dim_mask,
            other=0.0,
        )
        prefix_max = tl.load(
            max_logits + safe_slot * MAX_STRIDE_SLOT, mask=has_slot, other=-float("inf")
        ).to(tl.float32)

        # 任一有效 token 要求 reset，就忽略本次之前的累计值，
        # 并清除该 slot 的 token 暂存有效标记；无效行不能触发 reset。
        resets = tl.load(reset_slots + rows, mask=valid_row, other=-1)
        reset = tl.sum((valid_row & (resets == region)).to(tl.int32), axis=0) > 0
        prefix_num = tl.where(reset, 0.0, prefix_num)
        prefix_den = tl.where(reset, 0.0, prefix_den)
        prefix_max = tl.where(reset, -float("inf"), prefix_max)
        if reset:
            tl.store(
                staging_valid + safe_slot * REGION_BLOCK_SIZE + toks,
                0,
                mask=has_slot & tok_mask,
            )

        # 直接按输入行读取数值和分数，转为 FP32 后累计。
        # 无效行不读取数据，也不参与后面的权重计算。
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

        # 统一旧状态与当前分数的最大值，避免指数计算数值过大。
        # 先把旧分子、分母缩放到同一尺度，再加上当前 token 的贡献。
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

        # 完成后按实际 region 编号定位页内缓存，写入最终 summary，
        # 然后清空并释放已有 slot；完整 region 不需要新申请 slot。
        if complete:
            _csa_store_fp8_mean(
                mean_cache
                + (region // SUMMARIES_PER_PAGE) * MEAN_STRIDE_PAGE
                + (region % SUMMARIES_PER_PAGE) * PROXY_DIM,
                dims,
                mean,
                dim_mask,
            )
            tl.store(active_slot_by_region + region, -1)
            if has_slot:
                tl.store(active_region_ids + safe_slot, -1)
                tl.store(
                    active_numerator + safe_slot * NUMERATOR_STRIDE_SLOT + dims,
                    0.0,
                    mask=dim_mask,
                )
                tl.store(
                    denominator + safe_slot * DENOMINATOR_STRIDE_SLOT + dims,
                    0.0,
                    mask=dim_mask,
                )
                tl.store(max_logits + safe_slot * MAX_STRIDE_SLOT, -float("inf"))
        else:
            # CPU 计划直接给目标地址，旧状态与目标地址都属于 runner 预留区。
            # 双 slot 交替保证新尾部不会覆盖其他任务仍要读取的旧尾部。
            target = tl.load(planned_targets + request, mask=ordinal == request_region_count - 1, other=-1).to(tl.int64)
            target_ok = (target >= 0) & (target < active_capacity)
            if target_ok:
                if has_slot & (slot != target):
                    tl.store(active_region_ids + safe_slot, -1)
                    tl.store(active_numerator + safe_slot * NUMERATOR_STRIDE_SLOT + dims,
                             0.0, mask=dim_mask)
                    tl.store(denominator + safe_slot * DENOMINATOR_STRIDE_SLOT + dims,
                             0.0, mask=dim_mask)
                    tl.store(max_logits + safe_slot * MAX_STRIDE_SLOT, -float("inf"))
                    tl.store(staging_valid + safe_slot * REGION_BLOCK_SIZE + toks,
                             0, mask=tok_mask)
                # 累计值由本次结果覆盖；清除目标里上一代 token 的有效标记。
                tl.store(staging_valid + target * REGION_BLOCK_SIZE + toks, 0, mask=tok_mask)
                tl.store(active_region_ids + target, region_i32)
                tl.store(active_slot_by_region + region, target.to(tl.int32))
            safe_slot = tl.where(target_ok, target, 0)
            has_slot = target_ok
            # 准备阶段或 CPU 计划已确定 slot；这里只写累计结果，不再申请空位。
            if has_slot:
                tl.store(
                    active_numerator + safe_slot * NUMERATOR_STRIDE_SLOT + dims,
                    numerator,
                    mask=dim_mask,
                )
                tl.store(
                    denominator + safe_slot * DENOMINATOR_STRIDE_SLOT + dims,
                    denominator_acc,
                    mask=dim_mask,
                )
                tl.store(max_logits + safe_slot * MAX_STRIDE_SLOT, shared_max)
            else:
                tl.store(allocation_success, 0)


def csa_compact_prefill_update(
    mean_cache: torch.Tensor,
    active_region_ids: torch.Tensor,
    active_slot_by_region: torch.Tensor,
    allocation_success: torch.Tensor,
    active_numerator: torch.Tensor,
    denominator: torch.Tensor,
    max_logits: torch.Tensor,
    staging_valid: torch.Tensor,
    flat_slot: torch.Tensor,
    reset_slots: torch.Tensor,
    token_valid: torch.Tensor,
    token_positions: torch.Tensor,
    index_k: torch.Tensor,
    index_z: torch.Tensor,
    *,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    live_rows: int,
    num_pages: int,
    summaries_per_page: int,
    proxy_dim: int,
    region_block_size: int,
    slot_plan: CSAMetadata,
) -> None:
    """Fold request-derived region tasks without unique or token-row scratch.

    Query rows must follow consecutive sequence positions within each request,
    with each physical region updated by only one task. Query boundaries may
    have a nonzero origin when processing a sliced prefill tail. The caller
    provides a runner slot plan containing per-request int32 region counts and addresses.
    Only the update kernel is launched; there is no device allocation path.
    Slot capacity must cover existing states plus all new incomplete regions;
    preparation cannot reuse slots released by this call's update.
    """
    live_rows = int(live_rows)
    if live_rows < 0 or live_rows > flat_slot.numel():
        raise ValueError("live_rows must be within flat_slot")
    if region_block_size <= 0:
        raise ValueError("region_block_size must be positive")
    if live_rows == 0:
        return

    num_requests = int(query_start_loc.numel()) - 1
    if num_requests <= 0 or seq_lens.numel() < num_requests:
        raise ValueError("prefill requires query boundaries and sequence lengths")
    if slot_plan is None:
        raise ValueError("CSA prefill requires a runner-provided slot plan")
    if any(values is None or values.numel() != num_requests for values in
           (slot_plan.region_count, slot_plan.target_slot, slot_plan.source_slot)):
        raise ValueError("CSA slot plan does not match the prefill request count")
    if slot_plan.max_regions_per_request < 0:
        raise ValueError("CSA slot plan has invalid maximum region count")
    if slot_plan.max_regions_per_request == 0:
        return
    region_count = slot_plan.region_count
    planned_targets = slot_plan.target_slot
    planned_sources = slot_plan.source_slot
    _csa_prefill_update_kernel[(num_requests, slot_plan.max_regions_per_request)](
        mean_cache,
        active_region_ids,
        active_slot_by_region,
        allocation_success,
        active_numerator,
        denominator,
        max_logits,
        staging_valid,
        flat_slot,
        reset_slots,
        token_valid,
        token_positions,
        index_k,
        index_z,
        query_start_loc,
        seq_lens,
        region_count,
        planned_targets, planned_sources,
        num_requests,
        live_rows,
        int(num_pages) * int(summaries_per_page),
        MEAN_STRIDE_PAGE=int(mean_cache.stride(0)),
        NUMERATOR_STRIDE_SLOT=int(active_numerator.stride(0)),
        DENOMINATOR_STRIDE_SLOT=int(denominator.stride(0)),
        MAX_STRIDE_SLOT=int(max_logits.stride(0)),
        SUMMARIES_PER_PAGE=int(summaries_per_page),
        PROXY_DIM=int(proxy_dim),
        REGION_BLOCK_SIZE=int(region_block_size),
        active_capacity=int(active_region_ids.numel()),
        BLOCK_T=triton.next_power_of_2(int(region_block_size)),
        BLOCK_D=triton.next_power_of_2(int(proxy_dim)),
        num_warps=4,
    )


# ---------------------------------------------------------------------------
# MTP verify batches: staged tokens + deferred region flush
# ---------------------------------------------------------------------------


@triton.jit
def _csa_prefill_handoff_kernel(
    prefill_owners, prefill_reverse, prefill_n, prefill_d, prefill_m,
    owners, reverse, numerator, denominator, max_logits, staging_valid,
    allocation_success, target_slots, handoff,
    PREFILL_CAPACITY: tl.constexpr, ACTIVE_CAPACITY: tl.constexpr,
    TOTAL_REGIONS: tl.constexpr, PROXY_DIM: tl.constexpr,
    REGION_SIZE: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_T: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    request = tl.program_id(0)
    if tl.load(handoff + request) == 0:
        return
    source = tl.load(target_slots + request).to(tl.int64)
    if (source < 0) | (source >= PREFILL_CAPACITY):
        tl.store(allocation_success, 0)
        return
    region = tl.load(prefill_owners + source)
    if (region < 0) | (region >= TOTAL_REGIONS):
        tl.store(allocation_success, 0)
        return

    # Prefill 收尾时交接；只剩一个提示词 token 时也提前交接给原单 token 路径。
    # 先找原缓存中的同一 region，避免重复申请。
    hint = tl.load(reverse + region).to(tl.int64)
    hint_ok = (hint >= 0) & (hint < ACTIVE_CAPACITY)
    hint_owner = tl.load(owners + tl.where(hint_ok, hint, 0), mask=hint_ok, other=-1)
    slot = tl.where(hint_ok & (hint_owner == region), hint, ACTIVE_CAPACITY)
    if slot == ACTIVE_CAPACITY:
        slots = tl.arange(0, BLOCK_S)
        old_owners = tl.load(owners + slots, mask=slots < ACTIVE_CAPACITY, other=-1)
        slot = tl.min(tl.where((slots < ACTIVE_CAPACITY) & (old_owners == region),
                               slots, ACTIVE_CAPACITY), axis=0).to(tl.int64)
    if slot == ACTIVE_CAPACITY:
        # 请求之间可能同时交接，沿用原缓存的 CAS 占位协议。
        for probe in tl.range(0, ACTIVE_CAPACITY):
            if slot == ACTIVE_CAPACITY:
                candidate = (region + probe) % ACTIVE_CAPACITY
                old = tl.atomic_cas(owners + candidate, -1, region)
                if (old == -1) | (old == region):
                    slot = candidate.to(tl.int64)
    if slot == ACTIVE_CAPACITY:
        tl.store(allocation_success, 0)
        return

    dims = tl.arange(0, BLOCK_D)
    mask = dims < PROXY_DIM
    n = tl.load(prefill_n + source * PROXY_DIM + dims, mask=mask, other=0.0)
    d = tl.load(prefill_d + source * PROXY_DIM + dims, mask=mask, other=0.0)
    m = tl.load(prefill_m + source)
    tl.store(numerator + slot * PROXY_DIM + dims, n, mask=mask)
    tl.store(denominator + slot * PROXY_DIM + dims, d, mask=mask)
    tl.store(max_logits + slot, m)
    tokens = tl.arange(0, BLOCK_T)
    tl.store(staging_valid + slot * REGION_SIZE + tokens, 0, mask=tokens < REGION_SIZE)
    tl.store(reverse + region, slot.to(tl.int32))
    # 后续 decode/MTP 只看到原缓存；独立 prefill 状态已经完成交接。
    tl.store(prefill_reverse + region, -1)
    tl.store(prefill_owners + source, -1)
    tl.store(prefill_n + source * PROXY_DIM + dims, 0.0, mask=mask)
    tl.store(prefill_d + source * PROXY_DIM + dims, 0.0, mask=mask)
    tl.store(prefill_m + source, -float("inf"))


def csa_prefill_handoff(
    prefill_owners: torch.Tensor, prefill_reverse: torch.Tensor,
    prefill_n: torch.Tensor, prefill_d: torch.Tensor, prefill_m: torch.Tensor,
    owners: torch.Tensor, reverse: torch.Tensor,
    numerator: torch.Tensor, denominator: torch.Tensor, max_logits: torch.Tensor,
    staging_valid: torch.Tensor, allocation_success: torch.Tensor,
    metadata: CSAMetadata, *, proxy_dim: int, region_size: int,
) -> None:
    """将最后一块 prefill 的未完成状态安装到原 CSA 缓存，供原生成路径接续。"""
    if not metadata.needs_handoff:
        return
    if metadata.handoff is None or metadata.handoff.numel() != metadata.target_slot.numel():
        raise ValueError("CSA prefill handoff requires one flag per request")
    _csa_prefill_handoff_kernel[(metadata.handoff.numel(),)](
        prefill_owners, prefill_reverse, prefill_n, prefill_d, prefill_m,
        owners, reverse, numerator, denominator, max_logits, staging_valid,
        allocation_success, metadata.target_slot, metadata.handoff,
        PREFILL_CAPACITY=prefill_owners.numel(), ACTIVE_CAPACITY=owners.numel(),
        TOTAL_REGIONS=reverse.numel(), PROXY_DIM=proxy_dim, REGION_SIZE=region_size,
        BLOCK_D=triton.next_power_of_2(proxy_dim),
        BLOCK_T=triton.next_power_of_2(region_size),
        BLOCK_S=triton.next_power_of_2(owners.numel()), num_warps=4,
    )


@triton.jit
def _csa_verify_stage_kernel(
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
    num_blocks,
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
    SLOTS_PER_REQ: tl.constexpr,
    K_ROWS: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
) -> None:
    """Stage one request's verify rows in position order; never flush.

    Spec-decode verify rows are provisional: the rejection sampler commits
    only an accepted prefix, and the next verify round rewrites the rejected
    positions at the same slots. Staging is last-write-wins per
    (region, offset), so a rewrite cleanly replaces a rejected token's
    contribution -- this is the Ascend equivalent of the upstream
    ``Step5DSAMTPTransaction`` rollback, with the accepted-prefix state
    recovered by replay instead of by fp32 snapshots. Regions therefore only
    materialize into the fp8 page cache through the deferred flush below,
    once they are provably fully committed.

    One program per request stages its K+1 rows sequentially in row order:
    a region-start row's reset must be ordered before the staging of the
    region's later offsets, which parallel per-token programs cannot
    guarantee.
    """
    block = tl.program_id(0)
    if block >= num_blocks:
        return
    dims = tl.arange(0, BLOCK_D)
    dim_mask = dims < PROXY_DIM
    toks = tl.arange(0, BLOCK_T)

    # Same-region carry: the K+1 rows of one request usually fall in one
    # or two regions, and a region's slot cannot change inside this
    # program (reset keeps ownership, fold keeps the slot live), so a row
    # whose region repeats the previous row's skips the hint/owner lookup
    # -- two dependent scalar loads off the critical path.
    prev_region = tl.full((), -1, tl.int64)
    prev_slot = tl.full((), ACTIVE_CAPACITY, tl.int64)

    for j in tl.static_range(K_ROWS):
        # triton-ascend rejects `continue` in static_range; guard the body.
        row = block * K_ROWS + j
        row_ok = row < live_rows
        valid = tl.load(token_valid + row).to(tl.int1)
        region = tl.load(flat_slot + row)
        region_ok = (region >= 0) & (region < total_regions)
        if row_ok & valid & region_ok:
            position = tl.load(token_positions + row)
            offset = position % REGION_BLOCK_SIZE

            region_i32 = region.to(tl.int32)
            if region == prev_region:
                slot = prev_slot
            else:
                hint = tl.load(active_slot_by_region + region).to(tl.int64)
                hint_ok = (hint >= 0) & (hint < ACTIVE_CAPACITY)
                safe_hint = tl.where(hint_ok, hint, 0)
                hint_owner = tl.load(
                    active_region_ids + safe_hint, mask=hint_ok, other=-1
                )
                slot = tl.where(
                    hint_ok & (hint_owner == region_i32), hint, ACTIVE_CAPACITY
                )
            has_slot = slot != ACTIVE_CAPACITY
            safe_slot = tl.where(has_slot, slot, 0)

            # Region restart at offset 0 clears any older state for this
            # region (fresh region after a page flush, or an abandoned
            # generation tail).
            reset = tl.load(reset_slots + row) == region
            if reset:
                if has_slot:
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
                _csa_store_fp8_mean(
                    mean_cache
                    + (region // SUMMARIES_PER_PAGE) * mean_stride_page
                    + (region % SUMMARIES_PER_PAGE) * PROXY_DIM,
                    dims,
                    tl.zeros((BLOCK_D,), dtype=tl.float32),
                    dim_mask,
                )
                # The slot keeps its ownership: staging continues in the
                # same slot. Nulling it here would force a re-allocation
                # whose CAS probe can claim a second slot for the same
                # region and orphan the old one.

            cur_k = tl.load(
                index_k + row * PROXY_DIM + dims, mask=dim_mask, other=0.0
            ).to(tl.float32)
            cur_z = tl.load(
                index_z + row * PROXY_DIM + dims, mask=dim_mask, other=0.0
            ).to(tl.float32)

            # Allocate a slot on first touch (the region may still be
            # slotless when the verify batch is the first to touch it after
            # a page flush). Slots are partitioned per request:
            # [block*SLOTS_PER_REQ, +SLOTS_PER_REQ) is written only by this
            # program within this launch -- regions are request-private,
            # and every other slot-table writer (deferred flush, reset,
            # prefill) runs in a separate, sequentially ordered launch --
            # so a plain read/claim probe is race-free and skips the
            # atomic_cas hardware sync entirely (measured at ~3.6us fixed
            # per issue plus cross-program serialization at n=16). The
            # 8-slots-per-request budget is the documented capacity policy;
            # partition exhaustion keeps the fail-closed allocation_success
            # contract of the global probe.
            if slot == ACTIVE_CAPACITY:
                part_lo = block * SLOTS_PER_REQ
                if part_lo + SLOTS_PER_REQ <= ACTIVE_CAPACITY:
                    empty = tl.full((), -1, tl.int32)
                    for probe in tl.range(0, SLOTS_PER_REQ):
                        if slot == ACTIVE_CAPACITY:
                            cand = part_lo + probe
                            own = tl.load(active_region_ids + cand)
                            if (own == empty) | (own == region_i32):
                                tl.store(active_region_ids + cand, region_i32)
                                slot = cand.to(tl.int64)
                if slot == ACTIVE_CAPACITY:
                    tl.store(allocation_success, 0)
            # Carry the post-allocation slot so the next row of this same
            # region skips both the hint lookup and any re-allocation.
            prev_region = region
            prev_slot = slot
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

                if offset == REGION_BLOCK_SIZE - 1:
                    # Upstream alignment: fold the region the moment its last
                    # offset is staged and install the mean into the fp8 page
                    # cache, so the indexer scoring in this same forward sees
                    # a real summary for it (the GPU MTP transaction kernel
                    # installs every touched region the same way). The slot
                    # and the staging stay live: a rejected position is
                    # rewritten by the next round, whose last row re-runs this
                    # fold with the corrected values, and the deferred flush
                    # below still re-folds and releases the slot once
                    # region_end <= committed ctx.
                    staged_ok = (
                        tl.load(
                            staging_valid
                            + safe_slot * REGION_BLOCK_SIZE
                            + toks,
                            mask=toks < REGION_BLOCK_SIZE,
                            other=0,
                        )
                        != 0
                    )
                    prefix_num = tl.load(
                        active_numerator
                        + safe_slot * numerator_stride_slot
                        + dims,
                        mask=dim_mask,
                        other=0.0,
                    )
                    prefix_den = tl.load(
                        denominator
                        + safe_slot * denominator_stride_slot
                        + dims,
                        mask=dim_mask,
                        other=0.0,
                    )
                    prefix_max = tl.load(
                        max_logits + safe_slot * max_stride_slot
                    ).to(tl.float32)
                    # The staging rows are contiguous in memory, so the
                    # (token, dim) tile is fetched as one flat contiguous
                    # vector and reshaped -- bit-identical values and masks,
                    # wider memory transactions than the 2D gather (the
                    # contiguous load is what makes this kernel fast on the
                    # A5 vector cores).
                    flat_toks = tl.arange(0, BLOCK_T * BLOCK_D)
                    flat_mask = tl.reshape(
                        staged_ok[:, None] & dim_mask[None, :],
                        (BLOCK_T * BLOCK_D,),
                    )
                    staged_z = tl.reshape(
                        tl.load(
                            staging_z
                            + safe_slot * (REGION_BLOCK_SIZE * PROXY_DIM)
                            + flat_toks,
                            mask=flat_mask,
                            other=-float("inf"),
                        ).to(tl.float32),
                        (BLOCK_T, BLOCK_D),
                    )
                    staged_k = tl.reshape(
                        tl.load(
                            staging_k
                            + safe_slot * (REGION_BLOCK_SIZE * PROXY_DIM)
                            + flat_toks,
                            mask=flat_mask,
                            other=0.0,
                        ).to(tl.float32),
                        (BLOCK_T, BLOCK_D),
                    )
                    has_prefix = tl.max(prefix_den) > 0.0
                    base_max = tl.where(has_prefix, prefix_max, -float("inf"))
                    shared_max = tl.maximum(
                        base_max,
                        tl.max(
                            tl.where(staged_ok[:, None], staged_z, -float("inf"))
                        ),
                    )
                    prefix_scale = tl.where(
                        has_prefix, tl.exp(prefix_max - shared_max), 0.0
                    )
                    staged_w = tl.where(
                        staged_ok[:, None], tl.exp(staged_z - shared_max), 0.0
                    )
                    fold_num = prefix_num * prefix_scale + tl.sum(
                        staged_w * staged_k, axis=0
                    )
                    fold_den = prefix_den * prefix_scale + tl.sum(staged_w, axis=0)
                    mean = tl.where(
                        fold_den > 0.0,
                        fold_num / tl.maximum(fold_den, 1.0e-20),
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


def csa_verify_stage_update(
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
    query_len: int,
    num_pages: int,
    summaries_per_page: int,
    proxy_dim: int,
    region_block_size: int,
) -> None:
    """Stage the verify rows of an MTP batch into the active slot state."""
    rows = int(flat_slot.numel())
    if rows == 0 or live_rows <= 0 or query_len <= 1:
        return
    num_blocks = (min(rows, live_rows) + query_len - 1) // query_len
    _csa_verify_stage_kernel[(num_blocks,)](
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
        int(num_blocks),
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
        ALLOC_PROBE=int(active_region_ids.numel()),
        SLOTS_PER_REQ=8,
        K_ROWS=int(query_len),
        BLOCK_T=triton.next_power_of_2(int(region_block_size)),
        BLOCK_D=triton.next_power_of_2(int(proxy_dim)),
        num_warps=4,
        compile_mode="simd",
    )


@triton.jit
def _csa_deferred_flush_kernel(
    mean_cache,
    active_region_ids,
    active_slot_by_region,
    active_numerator,
    denominator,
    max_logits,
    staging_k,
    staging_z,
    staging_valid,
    seq_lens,
    block_table,
    num_flush_reqs,
    query_len,
    max_table_cols,
    total_regions,
    mean_stride_page,
    numerator_stride_slot,
    denominator_stride_slot,
    max_stride_slot,
    SUMMARIES_PER_PAGE: tl.constexpr,
    PROXY_DIM: tl.constexpr,
    REGION_BLOCK_SIZE: tl.constexpr,
    REGIONS_PER_PAGE: tl.constexpr,
    ACTIVE_CAPACITY: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
) -> None:
    """Flush one request's highest fully-committed region, if any.

    A region is committable when its last position is below the request's
    committed context ``ctx = seq_len - query_len``: verify acceptance is at
    least one token per round and never retreats, so once ``region_end <=
    ctx`` the region's staged tokens can no longer be rewritten. The fold is
    the same shared-max online-softmax merge the decode flush uses (prefill
    prefix N/D/M plus every staged token).
    """
    req = tl.program_id(0)
    if req >= num_flush_reqs:
        return
    seq_len = tl.load(seq_lens + req).to(tl.int32)
    ctx = seq_len - query_len
    region_idx = ctx // REGION_BLOCK_SIZE - 1
    if region_idx < 0:
        return
    page_col = region_idx // REGIONS_PER_PAGE
    if page_col >= max_table_cols:
        return
    page = tl.load(block_table + req * max_table_cols + page_col).to(tl.int32)
    if page < 0:
        return
    region = (page * REGIONS_PER_PAGE + region_idx % REGIONS_PER_PAGE).to(tl.int64)
    if (region < 0) | (region >= total_regions):
        return

    hint = tl.load(active_slot_by_region + region).to(tl.int64)
    hint_ok = (hint >= 0) & (hint < ACTIVE_CAPACITY)
    safe_hint = tl.where(hint_ok, hint, 0)
    hint_owner = tl.load(active_region_ids + safe_hint, mask=hint_ok, other=-1)
    if ~(hint_ok & (hint_owner == region.to(tl.int32))):
        # Not staged (already flushed, or owned by prefill state): nothing
        # to do. Prefill-flushed complete regions release their slots, and
        # an N/D/M-only tail region belongs to the current context.
        return
    slot = safe_hint

    dims = tl.arange(0, BLOCK_D)
    dim_mask = dims < PROXY_DIM
    toks = tl.arange(0, BLOCK_T)
    tok_mask = toks < REGION_BLOCK_SIZE

    # Eligibility (region_end <= ctx) already proves every offset of the
    # region was written: positions are covered contiguously by successive
    # verify batches, and never rewritten once committed. Prefill-written
    # offsets live in the prefix N/D/M, verify-written offsets in staging,
    # so the fold below covers the region exactly.
    staged_ok = (
        tl.load(
            staging_valid + slot * REGION_BLOCK_SIZE + toks, mask=tok_mask, other=0
        )
        != 0
    )

    prefix_num = tl.load(
        active_numerator + slot * numerator_stride_slot + dims,
        mask=dim_mask,
        other=0.0,
    )
    prefix_den = tl.load(
        denominator + slot * denominator_stride_slot + dims,
        mask=dim_mask,
        other=0.0,
    )
    prefix_max = tl.load(max_logits + slot * max_stride_slot).to(tl.float32)

    staged_z = tl.load(
        staging_z + slot * (REGION_BLOCK_SIZE * PROXY_DIM) + toks[:, None] * PROXY_DIM + dims[None, :],
        mask=staged_ok[:, None] & dim_mask[None, :],
        other=-float("inf"),
    ).to(tl.float32)
    staged_k = tl.load(
        staging_k + slot * (REGION_BLOCK_SIZE * PROXY_DIM) + toks[:, None] * PROXY_DIM + dims[None, :],
        mask=staged_ok[:, None] & dim_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    has_prefix = tl.max(prefix_den) > 0.0
    base_max = tl.where(has_prefix, prefix_max, -float("inf"))
    shared_max = tl.maximum(base_max, tl.max(tl.where(staged_ok[:, None], staged_z, -float("inf"))))
    scale = tl.where(has_prefix, tl.exp(prefix_max - shared_max), 0.0)
    staged_w = tl.where(staged_ok[:, None], tl.exp(staged_z - shared_max), 0.0)
    numerator = prefix_num * scale + tl.sum(staged_w * staged_k, axis=0)
    denominator_acc = prefix_den * scale + tl.sum(staged_w, axis=0)
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
    # Release the slot: the region now lives only in the page cache.
    tl.store(active_slot_by_region + region, -1)
    tl.store(active_region_ids + slot, -1)
    tl.store(
        active_numerator + slot * numerator_stride_slot + dims, 0.0, mask=dim_mask
    )
    tl.store(
        denominator + slot * denominator_stride_slot + dims, 0.0, mask=dim_mask
    )
    tl.store(max_logits + slot * max_stride_slot, -float("inf"))
    tl.store(
        staging_valid + slot * REGION_BLOCK_SIZE + toks, 0, mask=tok_mask
    )


def csa_deferred_flush(
    mean_cache: torch.Tensor,
    active_region_ids: torch.Tensor,
    active_slot_by_region: torch.Tensor,
    active_numerator: torch.Tensor,
    denominator: torch.Tensor,
    max_logits: torch.Tensor,
    staging_k: torch.Tensor,
    staging_z: torch.Tensor,
    staging_valid: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    *,
    num_reqs: int,
    query_len: int,
    num_pages: int,
    summaries_per_page: int,
    region_block_size: int,
    page_size: int,
) -> None:
    """Materialize fully-committed tail regions of MTP verify requests."""
    rows = int(seq_lens.numel())
    if rows == 0 or num_reqs <= 0 or query_len <= 1:
        return
    _csa_deferred_flush_kernel[(rows,)](
        mean_cache,
        active_region_ids,
        active_slot_by_region,
        active_numerator,
        denominator,
        max_logits,
        staging_k,
        staging_z,
        staging_valid,
        seq_lens,
        block_table,
        int(num_reqs),
        int(query_len),
        int(block_table.stride(0)),
        int(num_pages) * int(summaries_per_page),
        int(mean_cache.stride(0)),
        int(active_numerator.stride(0)),
        int(denominator.stride(0)),
        int(max_logits.stride(0)),
        SUMMARIES_PER_PAGE=int(summaries_per_page),
        PROXY_DIM=int(staging_k.shape[-1]),
        REGION_BLOCK_SIZE=int(region_block_size),
        REGIONS_PER_PAGE=int(page_size) // int(region_block_size),
        ACTIVE_CAPACITY=int(active_region_ids.numel()),
        BLOCK_T=triton.next_power_of_2(int(region_block_size)),
        BLOCK_D=triton.next_power_of_2(int(staging_k.shape[-1])),
        num_warps=4,
    )
