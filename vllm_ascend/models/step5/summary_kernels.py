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
  prefill consumes a runner plan; fallback prefill assigns compacted free slots;
  decode/verify use CAS probes. Exhaustion publishes
  ``allocation_success=0`` and the caller fails closed.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from vllm_ascend.worker.step5_csa_slot_manager import CSAPrefillSlotPlan

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
        start = region % ALLOC_PROBE
        empty = tl.full((), -1, tl.int32)
        for probe in tl.range(0, ALLOC_PROBE):
            if slot == ACTIVE_CAPACITY:
                cand = (start + probe) % ALLOC_PROBE
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
    allocation_capacity: int | None = None,
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
        ALLOC_PROBE=int(active_region_ids.numel() if allocation_capacity is None else allocation_capacity),
        BLOCK_T=triton.next_power_of_2(int(region_block_size)),
        BLOCK_D=triton.next_power_of_2(int(proxy_dim)),
        num_warps=4,
    )


# ---------------------------------------------------------------------------
# Prefill / mixed-batch summary update (slot preparation + single update)
# ---------------------------------------------------------------------------


@triton.jit
def _csa_prefill_task_prefix_kernel(
    query_start_loc,
    seq_lens,
    task_prefix,
    num_requests,
    active_region_ids, active_slot_by_region, free_slots, active_capacity, total_regions, allocation_capacity,
    REGION_BLOCK_SIZE: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_S: tl.constexpr,
) -> None:
    # 每批只扫描一次拥有者表，同时修复反向映射、按下标顺序收集空位。
    # 此时不释放旧状态，update 结束前不能把旧尾部的 slot 分给新尾部。
    slots = tl.arange(0, BLOCK_S)
    owners = tl.load(active_region_ids + slots, mask=slots < active_capacity, other=-1)
    occupied = (slots < active_capacity) & (owners >= 0) & (owners < total_regions)
    tl.store(active_slot_by_region + owners, slots, mask=occupied)
    free = (slots < allocation_capacity) & (owners == -1)
    ranks = tl.cumsum(free.to(tl.int32), axis=0) - 1
    tl.store(free_slots + ranks, slots, mask=free)
    tl.store(free_slots + active_capacity, tl.sum(free.to(tl.int32), axis=0))
    # 每个请求需要的任务数：本次 token 覆盖了多少个逻辑 region。
    requests = tl.arange(0, BLOCK_R)
    valid = requests < num_requests
    starts = tl.load(query_start_loc + requests, mask=valid, other=0)
    ends = tl.load(query_start_loc + requests + 1, mask=valid, other=0)
    lengths = tl.load(seq_lens + requests, mask=valid, other=0)
    query_lengths = ends - starts
    context_offsets = (lengths - query_lengths) % REGION_BLOCK_SIZE
    counts = tl.where(
        valid & (query_lengths > 0),
        tl.cdiv(context_offsets + query_lengths, REGION_BLOCK_SIZE),
        0,
    ).to(tl.int32)
    # 例如任务数 [3, 1] 得到累计边界 [0, 3, 4]，由 slot 准备和 update 复用。
    tl.store(task_prefix, 0)
    tl.store(task_prefix + requests + 1, tl.cumsum(counts, axis=0), mask=valid)


@triton.jit
def _csa_prefill_task_rows(
    task, query_start_loc, seq_lens, task_prefix, num_requests,
    flat_slot, token_valid, token_positions, live_rows, total_regions,
    REGION_BLOCK_SIZE: tl.constexpr, BLOCK_T: tl.constexpr,
):
    # 准备和计算阶段共用任务定位及有效性规则，避免两阶段对同一任务判断不同。
    row_base = tl.load(query_start_loc)
    # 找到 prefix[req] <= task < prefix[req + 1] 的请求，跳过零任务请求。
    lo = tl.full((), 0, tl.int32)
    hi = num_requests
    while lo < hi:
        mid = (lo + hi) // 2
        task_end = tl.load(task_prefix + mid + 1)
        go_left = task < task_end
        hi = tl.where(go_left, mid, hi)
        lo = tl.where(go_left, lo, mid + 1)
    request = lo
    task_base = tl.load(task_prefix + request)
    query_start = tl.load(query_start_loc + request) - row_base
    query_end = tl.load(query_start_loc + request + 1) - row_base
    query_length = query_end - query_start
    context_length = tl.load(seq_lens + request) - query_length
    context_offset = context_length % REGION_BLOCK_SIZE
    ordinal = task - task_base
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
def _csa_prefill_prepare_slots_kernel(
    active_region_ids, active_slot_by_region, slot_requests,
    flat_slot, token_valid, token_positions,
    query_start_loc, seq_lens, task_prefix, num_requests, live_rows, total_regions,
    active_capacity,
    NUM_CORES: tl.constexpr,
    REGION_BLOCK_SIZE: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    # 并行判断哪些任务需要新 slot。反向映射已经由前一个 kernel 修复，
    # 此处只需一次提示读取和一次拥有者核对，不扫描 slot 表、不抢占空位。
    pid = tl.program_id(0)
    num_tasks = tl.load(task_prefix + num_requests)
    for task in range(pid, num_tasks, NUM_CORES):
        region, rows, valid_row, complete = _csa_prefill_task_rows(
            task, query_start_loc, seq_lens, task_prefix, num_requests,
            flat_slot, token_valid, token_positions, live_rows, total_regions,
            REGION_BLOCK_SIZE, BLOCK_T,
        )
        requested_region = tl.full((), -1, tl.int32)
        if (tl.sum(valid_row.to(tl.int32), axis=0) > 0) & ~complete:
            hint = tl.load(active_slot_by_region + region).to(tl.int64)
            hint_ok = (hint >= 0) & (hint < active_capacity)
            owner = tl.load(active_region_ids + tl.where(hint_ok, hint, 0), mask=hint_ok, other=-1)
            if ~(hint_ok & (owner == region)):
                requested_region = region.to(tl.int32)
        tl.store(slot_requests + task, requested_region)


@triton.jit
def _csa_prefill_assign_slots_kernel(
    active_region_ids, active_slot_by_region, allocation_success,
    active_numerator, denominator, max_logits, staging_valid,
    free_slots, slot_requests, task_prefix, num_requests, active_capacity,
    NUMERATOR_STRIDE_SLOT: tl.constexpr,
    DENOMINATOR_STRIDE_SLOT: tl.constexpr,
    MAX_STRIDE_SLOT: tl.constexpr,
    PROXY_DIM: tl.constexpr,
    REGION_BLOCK_SIZE: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_A: tl.constexpr,
):
    # 一个程序按小批次累计申请数：第 k 个申请者直接拿 free_slots[k]。
    # 空位编号唯一，无需 atomic_cas；小批次限制初始化张量的临时空间。
    num_tasks = tl.load(task_prefix + num_requests)
    num_free = tl.load(free_slots + active_capacity)
    allocated = tl.full((), 0, tl.int32)
    offsets = tl.arange(0, BLOCK_A)
    dims = tl.arange(0, BLOCK_D)
    toks = tl.arange(0, BLOCK_T)
    for start in range(0, num_tasks, BLOCK_A):
        tasks = start + offsets
        regions = tl.load(slot_requests + tasks, mask=tasks < num_tasks, other=-1)
        need = (tasks < num_tasks) & (regions >= 0)
        ranks = allocated + tl.cumsum(need.to(tl.int32), axis=0) - 1
        assigned = need & (ranks < num_free)
        slots = tl.load(free_slots + ranks, mask=assigned, other=0)
        if tl.sum((need & ~assigned).to(tl.int32), axis=0) > 0:
            tl.store(allocation_success, 0)
        # 每批只初始化确实申请成功的新 slot，旧状态不受影响。
        if tl.sum(assigned.to(tl.int32), axis=0) > 0:
            tl.store(active_region_ids + slots, regions, mask=assigned)
            tl.store(active_slot_by_region + regions, slots, mask=assigned)
            tl.store(active_numerator + slots[:, None] * NUMERATOR_STRIDE_SLOT + dims[None, :],
                     0.0, mask=assigned[:, None] & (dims[None, :] < PROXY_DIM))
            tl.store(denominator + slots[:, None] * DENOMINATOR_STRIDE_SLOT + dims[None, :],
                     0.0, mask=assigned[:, None] & (dims[None, :] < PROXY_DIM))
            tl.store(max_logits + slots * MAX_STRIDE_SLOT, -float("inf"), mask=assigned)
            tl.store(staging_valid + slots[:, None] * REGION_BLOCK_SIZE + toks[None, :],
                     0, mask=assigned[:, None] & (toks[None, :] < REGION_BLOCK_SIZE))
        allocated += tl.sum(need.to(tl.int32), axis=0)


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
    task_prefix,
    planned_targets, planned_sources,
    num_requests,
    live_rows,
    total_regions,
    NUM_CORES: tl.constexpr,
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
    HOST_PLANNED: tl.constexpr,
) -> None:
    """Fold all prepared region tasks in one launch without slot allocation.

    Query rows must follow contiguous sequence positions within each request.
    Each physical region must have only one updating task in the batch.
    """
    # 按请求边界和长度定位 region，不再逐行比较 flat_slot 寻找首行。
    pid = tl.program_id(0)
    dims = tl.arange(0, BLOCK_D)
    dim_mask = dims < PROXY_DIM
    toks = tl.arange(0, BLOCK_T)
    tok_mask = toks < REGION_BLOCK_SIZE

    # 准备阶段分配失败时停止本次更新，由调用方报告容量不足。
    if tl.load(allocation_success) == 0:
        return
    num_tasks = tl.load(task_prefix + num_requests)
    for task in range(pid, num_tasks, NUM_CORES):
        region, rows, valid_row, complete = _csa_prefill_task_rows(
            task, query_start_loc, seq_lens, task_prefix, num_requests,
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
            if HOST_PLANNED:
                # Runner knows the old reserved address even if the reverse hint
                # is stale. Confirm physical ownership before reading any state.
                planned_source = tl.load(planned_sources + task).to(tl.int64)
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
                # CPU 计划直接给目标地址，旧状态可以来自 prefill 预留区或 decode 池。
                # 双 slot 交替保证新尾部不会覆盖其他任务仍要读取的旧尾部。
                if HOST_PLANNED:
                    target = tl.load(planned_targets + task).to(tl.int64)
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
    task_prefix: torch.Tensor,
    free_slots: torch.Tensor,
    slot_requests: torch.Tensor,
    live_rows: int,
    num_pages: int,
    summaries_per_page: int,
    proxy_dim: int,
    region_block_size: int,
    allocation_capacity: int | None = None,
    slot_plan: CSAPrefillSlotPlan | None = None,
) -> None:
    """Fold request-derived region tasks without unique or token-row scratch.

    Query rows must follow consecutive sequence positions within each request,
    with each physical region updated by only one task. Query boundaries may
    have a nonzero origin when processing a sliced prefill tail. The caller
    provides int32 task-prefix, free-slot, and per-task allocation buffers.
    With a runner slot plan, only the update kernel is launched. Otherwise,
    free slots in the device partition are collected and assigned without CAS.
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
    num_cores = _get_num_vector_cores()
    if slot_plan is not None:
        # CPU 已生成任务边界和分配计划，各层只需一次 update 启动。
        if slot_plan.task_prefix.numel() != num_requests + 1:
            raise ValueError("CSA slot plan does not match the prefill request count")
        if slot_plan.target_slots.numel() != slot_plan.num_tasks or slot_plan.source_slots.numel() != slot_plan.num_tasks:
            raise ValueError("CSA slot plan does not match its task count")
        task_prefix = slot_plan.task_prefix
        planned_targets = slot_plan.target_slots
        planned_sources = slot_plan.source_slots
    else:
        planned_targets = task_prefix
        planned_sources = task_prefix
        if task_prefix.dtype != torch.int32 or task_prefix.numel() < num_requests + 1:
            raise ValueError("task_prefix must be int32 with num_requests + 1 entries")
        if free_slots.dtype != torch.int32 or free_slots.numel() < active_region_ids.numel() + 1:
            raise ValueError("free_slots must be int32 with active_capacity + 1 entries")
        if slot_requests.dtype != torch.int32 or slot_requests.numel() < live_rows:
            raise ValueError("slot_requests must be int32 with at least live_rows entries")
        # 每次调用刷新累计任务数，避免复用缓冲区时读取上一批次的请求长度。
        _csa_prefill_task_prefix_kernel[(1,)](
            query_start_loc,
            seq_lens,
            task_prefix,
            num_requests,
            active_region_ids, active_slot_by_region, free_slots, int(active_region_ids.numel()),
            int(num_pages) * int(summaries_per_page),
            int(active_region_ids.numel() if allocation_capacity is None else allocation_capacity),
            BLOCK_S=triton.next_power_of_2(int(active_region_ids.numel())),
            REGION_BLOCK_SIZE=int(region_block_size),
            BLOCK_R=triton.next_power_of_2(num_requests),
        )

        # 四次顺序提交：任务计数和空位收集、申请判断、按序分配、一次全量 update。
        _csa_prefill_prepare_slots_kernel[(num_cores,)](
            active_region_ids, active_slot_by_region, slot_requests,
            flat_slot, token_valid, token_positions,
            query_start_loc, seq_lens, task_prefix, num_requests, live_rows,
            int(num_pages) * int(summaries_per_page), int(active_region_ids.numel()),
            NUM_CORES=num_cores,
            REGION_BLOCK_SIZE=int(region_block_size),
            BLOCK_T=triton.next_power_of_2(int(region_block_size)),
            num_warps=4,
        )
        _csa_prefill_assign_slots_kernel[(1,)](
            active_region_ids, active_slot_by_region, allocation_success,
            active_numerator, denominator, max_logits, staging_valid,
            free_slots, slot_requests, task_prefix, num_requests, int(active_region_ids.numel()),
            NUMERATOR_STRIDE_SLOT=int(active_numerator.stride(0)),
            DENOMINATOR_STRIDE_SLOT=int(denominator.stride(0)),
            MAX_STRIDE_SLOT=int(max_logits.stride(0)),
            PROXY_DIM=int(proxy_dim),
            REGION_BLOCK_SIZE=int(region_block_size),
            BLOCK_T=triton.next_power_of_2(int(region_block_size)),
            BLOCK_D=triton.next_power_of_2(int(proxy_dim)),
            BLOCK_A=16,
            num_warps=4,
        )
    _csa_prefill_update_kernel[(num_cores,)](
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
        task_prefix,
        planned_targets, planned_sources,
        num_requests,
        live_rows,
        int(num_pages) * int(summaries_per_page),
        NUM_CORES=num_cores,
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
        HOST_PLANNED=slot_plan is not None,
        num_warps=4,
    )


# ---------------------------------------------------------------------------
# MTP verify batches: staged tokens + deferred region flush
# ---------------------------------------------------------------------------


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
            hint = tl.load(active_slot_by_region + region).to(tl.int64)
            hint_ok = (hint >= 0) & (hint < ACTIVE_CAPACITY)
            safe_hint = tl.where(hint_ok, hint, 0)
            hint_owner = tl.load(active_region_ids + safe_hint, mask=hint_ok, other=-1)
            slot = tl.where(hint_ok & (hint_owner == region_i32), hint, ACTIVE_CAPACITY)
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
            # a page flush).
            if slot == ACTIVE_CAPACITY:
                start = region % ALLOC_PROBE
                empty = tl.full((), -1, tl.int32)
                for probe in tl.range(0, ALLOC_PROBE):
                    if slot == ACTIVE_CAPACITY:
                        cand = (start + probe) % ALLOC_PROBE
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
    allocation_capacity: int | None = None,
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
        ALLOC_PROBE=int(active_region_ids.numel() if allocation_capacity is None else allocation_capacity),
        K_ROWS=int(query_len),
        BLOCK_T=triton.next_power_of_2(int(region_block_size)),
        BLOCK_D=triton.next_power_of_2(int(proxy_dim)),
        num_warps=4,
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
