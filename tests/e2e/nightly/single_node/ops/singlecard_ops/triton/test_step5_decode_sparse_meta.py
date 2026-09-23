#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
"""NPU test: Step5 DSA decode metadata (selector + pack) vs torch reference.

Covers ``decode_sparse_meta`` / ``_decode_pack_kernel``: batched top-k region
selection followed by int64 metadata packing for the sparse-attention decode
kernel (``start_token | phys_region << 32`` words, compacted in ascending
logical region id order, current region force-appended).

Regression background: the previous rank/sort compaction staged a transposed
scratch tile *inside* the ``ids`` tensor and walked past its end whenever
``rows % BLOCK_ROWS != 0`` (any odd decode batch with the production
``topk=512``), scribbling up to ``(BLOCK_ROWS - 1) * BLOCK_K`` int32 onto the
neighbouring allocation. A corrupted metadata word carrying the sort sentinel
(``~2**30``) in its physical half then sent ``_sparse_attn_decode_kernel``'s
K/V gather gigabytes out of the KV pool, surfacing as an MTE "DDR address out
of range" aicore exception (plog errcode 95) inside the *consumer* kernel.
The replacement cumsum compaction stages nothing; the canary test below pins
that memory behaviour directly, and the reference test pins the metadata
contract (including the ascending-logical-id ordering the fix introduces).

Usage:
    pytest test_step5_decode_sparse_meta.py -v
"""

import torch
import triton

from vllm_ascend.models.step5.dsa_kernels import (
    _decode_pack_kernel,
    decode_sparse_meta,
)

DEVICE = "npu"

TOPK = 512  # production sparse_config.topk (exercises the old USE_SORT path)
REGION_SIZE = 8  # production sparse_config.region_block_size
REGIONS_PER_PAGE = 16  # block_size 128 / region_size 8
CANARY = 0x5EED5EED


def _pack_block_rows(rows: int, block_k: int) -> int:
    """Launch-tile height, mirroring ``decode_sparse_meta``."""
    block_rows = min(128, max(2, triton.next_power_of_2(triton.cdiv(rows, 64))))
    return min(block_rows, max(2, 8192 // block_k))


def _fill_selector_ids(
    ids: torch.Tensor, kv_seqlens: torch.Tensor, num_reqs: int
) -> None:
    """Selector-style rows: ascending live region ids, -1 padding the tail."""
    rows = ids.shape[0]
    history = ((kv_seqlens - 1).clamp_min(0) // REGION_SIZE).cpu()
    ids_cpu = torch.full((rows, ids.shape[1]), -1, dtype=torch.int32)
    for r in range(rows):
        live = min(int(history[r % num_reqs]), ids.shape[1])
        if live > 0:
            ids_cpu[r, :live] = torch.arange(live, dtype=torch.int32)
    ids.copy_(ids_cpu)


def test_decode_pack_does_not_write_past_ids() -> None:
    """Canary regression test for the staging-tile OOB write.

    ``ids`` is carved out of one allocation with a canary tail, so any write
    past the tensor lands deterministically in the canary. Row counts cover
    odd/even batches and single/multi program grids; every case must leave
    the canary untouched.
    """
    torch.npu.set_device(0)
    for rows in (1, 2, 3, 15, 16, 17, 31, 32, 33):
        num_reqs = rows  # Step5 DSA runs one local KV head (groups == 1)
        block_k = triton.next_power_of_2(TOPK)
        block_rows = _pack_block_rows(rows, block_k)

        kv_seqlens = torch.randint(
            64, 4096, (num_reqs,), dtype=torch.int32, device=DEVICE
        )
        pages_per_req = 32
        block_table = (
            torch.arange(num_reqs * pages_per_req, dtype=torch.int32)
            .view(num_reqs, pages_per_req)
            .to(DEVICE)
        )

        buf = torch.full(
            (rows * TOPK + block_k * block_rows + 4096,),
            CANARY,
            dtype=torch.int32,
            device=DEVICE,
        )
        ids = buf[: rows * TOPK].view(rows, TOPK)
        canary = buf[rows * TOPK :]
        _fill_selector_ids(ids, kv_seqlens, num_reqs)

        packed = torch.zeros((rows, TOPK + 1), dtype=torch.int64, device=DEVICE)
        counts = torch.zeros((rows,), dtype=torch.int32, device=DEVICE)
        _decode_pack_kernel[(triton.cdiv(rows, block_rows),)](
            ids,
            kv_seqlens,
            block_table,
            packed,
            counts,
            rows,
            num_reqs,
            ids.stride(0),
            block_table.stride(0),
            packed.stride(0),
            topk=TOPK,
            region_size=REGION_SIZE,
            regions_per_page=REGIONS_PER_PAGE,
            BLOCK_ROWS=block_rows,
            BLOCK_K=block_k,
        )
        torch.npu.synchronize()
        corrupted = int((canary != CANARY).sum().item())
        assert corrupted == 0, (
            f"rows={rows}: {corrupted} canary elements past `ids` were "
            "overwritten by _decode_pack_kernel"
        )


def _reference_decode_sparse_meta(
    logits: torch.Tensor,
    kv_seqlens: torch.Tensor,
    block_table: torch.Tensor,
    *,
    topk: int,
    region_size: int,
    regions_per_page: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Host reference: top-k over visible regions, ascending-id compaction."""
    rows, seq_regions = logits.shape
    num_reqs = kv_seqlens.shape[0]
    packed_ref = torch.zeros((rows, topk + 1), dtype=torch.int64)
    counts_ref = torch.zeros((rows,), dtype=torch.int32)
    for r in range(rows):
        req = r % num_reqs
        sl = int(kv_seqlens[req])
        valid_regions = (sl + region_size - 1) // region_size
        history = max(sl - 1, 0) // region_size
        visible = min(history, seq_regions)
        k = min(topk, visible)
        sel: list[int] = []
        if k > 0:
            sel = torch.topk(logits[r, :visible].cpu(), k).indices.tolist()
            sel.sort()
        words = []
        for i in sel:
            if i >= valid_regions:
                continue
            page = int(block_table[req, i // regions_per_page])
            if page < 0:
                continue
            phys = page * regions_per_page + i % regions_per_page
            words.append((i * region_size) | (phys << 32))
        count = len(words)
        current_region = max((sl - 1) // region_size, 0)
        current_page = int(block_table[req, current_region // regions_per_page])
        current_live = current_region < valid_regions and current_page >= 0
        current_word = (current_region * region_size) | (
            (current_page * regions_per_page + current_region % regions_per_page)
            << 32
        )
        for pos, word in enumerate(words):
            packed_ref[r, pos] = word
        if current_live:
            slot = topk if count == topk else count
            packed_ref[r, slot] = current_word
        counts_ref[r] = count + (1 if current_live else 0)
    return packed_ref, counts_ref


def _run_wrapper_case(
    rows: int,
    num_reqs: int,
    seq_regions: int,
    topk: int,
    seed: int,
) -> None:
    """Full ``decode_sparse_meta`` vs host reference for one batch shape."""
    g = torch.Generator().manual_seed(seed)
    kv_seqlens = torch.randint(
        1, seq_regions * REGION_SIZE, (num_reqs,), generator=g, dtype=torch.int32
    )
    # Distinct, exactly-representable scores per row -> tie-free top-k.
    logits = torch.stack(
        [torch.randperm(seq_regions, generator=g) for r in range(rows)]
    ).to(torch.float32)
    pages = (seq_regions + REGIONS_PER_PAGE - 1) // REGIONS_PER_PAGE + 8
    block_table = torch.randperm(pages * 2, generator=g)[:pages].repeat(num_reqs, 1)
    # Punch unmapped holes; make sure every request keeps its first page.
    holes = torch.rand((num_reqs, pages), generator=g) < 0.1
    holes[:, 0] = False
    block_table = torch.where(holes, torch.full_like(block_table, -1), block_table)

    packed, counts = decode_sparse_meta(
        logits.to(DEVICE),
        kv_seqlens.to(DEVICE),
        block_table.to(DEVICE),
        topk=topk,
        region_size=REGION_SIZE,
        regions_per_page=REGIONS_PER_PAGE,
    )
    packed_ref, counts_ref = _reference_decode_sparse_meta(
        logits,
        kv_seqlens,
        block_table,
        topk=topk,
        region_size=REGION_SIZE,
        regions_per_page=REGIONS_PER_PAGE,
    )
    torch.testing.assert_close(counts.cpu(), counts_ref)
    torch.testing.assert_close(packed.cpu(), packed_ref)


def test_decode_sparse_meta_matches_reference_odd_rows() -> None:
    """Production shape: odd decode batch, topk=512, tiled selector path."""
    _run_wrapper_case(rows=15, num_reqs=15, seq_regions=9000, topk=TOPK, seed=0)


def test_decode_sparse_meta_matches_reference_groups() -> None:
    """Two KV groups sharing per-request lengths via group-major rows."""
    _run_wrapper_case(rows=16, num_reqs=8, seq_regions=9000, topk=TOPK, seed=1)


def test_decode_sparse_meta_matches_reference_small_topk() -> None:
    """Unified cumsum path below the old USE_SORT threshold (topk<=256)."""
    _run_wrapper_case(rows=33, num_reqs=33, seq_regions=300, topk=64, seed=2)


def test_decode_sparse_meta_single_request_min_len() -> None:
    """Degenerate batch: one request, one token of history."""
    _run_wrapper_case(rows=1, num_reqs=1, seq_regions=64, topk=16, seed=3)
