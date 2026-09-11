#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
"""NPU test: Step4 DSA paged summary cache Triton kernels vs torch reference.

Covers the kernels in ``vllm_ascend.models.step4.summary_kernels``:

* ``build_decode_summary_layout_step3p5`` (token -> physical region layout),
* ``csa_compact_prefill_update`` / ``csa_compact_decode_update`` (page-backed
  fp8 region summaries + active-slot runtime state),
* ``csa_reset_blocks`` (block lifecycle reset).

The paged indexer scoring against this cache is covered separately in
``test_step4_indexer_logits_paged.py``; the decode-side region-completion
flush scenario in ``test_step4_csa_decode_flush.py``.

The reference recomputes region summaries from the raw token stream:
``s_d = sum_t exp(z_td - m) * k_td / sum_t exp(z_td - m)`` with one scalar
``m`` per region, stored through the fp32 -> bf16 -> e4m3 double rounding.
"""

import torch

from vllm_ascend.models.step4.summary_kernels import (
    build_decode_summary_layout_step3p5,
    csa_compact_prefill_update,
    csa_reset_blocks,
)

DEVICE = "npu"

NUM_PAGES = 8
PAGE_SIZE = 16
REGION_SIZE = 8
SUMMARIES_PER_PAGE = PAGE_SIZE // REGION_SIZE
PROXY_DIM = 256
ACTIVE_CAPACITY = 16
TOTAL_REGIONS = NUM_PAGES * SUMMARIES_PER_PAGE


def _alloc_state():
    return dict(
        mean_cache=torch.zeros(
            (NUM_PAGES, SUMMARIES_PER_PAGE, 1, PROXY_DIM),
            device=DEVICE,
            dtype=torch.float8_e4m3fn,
        ),
        active_region_ids=torch.full(
            (ACTIVE_CAPACITY,), -1, device=DEVICE, dtype=torch.int32
        ),
        active_slot_by_region=torch.full(
            (TOTAL_REGIONS,), -1, device=DEVICE, dtype=torch.int32
        ),
        allocation_success=torch.ones((1,), device=DEVICE, dtype=torch.int32),
        numerator=torch.zeros(
            (ACTIVE_CAPACITY, 1, PROXY_DIM), device=DEVICE, dtype=torch.float32
        ),
        denominator=torch.zeros(
            (ACTIVE_CAPACITY, 1, PROXY_DIM), device=DEVICE, dtype=torch.float32
        ),
        max_logits=torch.full(
            (ACTIVE_CAPACITY, 1),
            float("-inf"),
            device=DEVICE,
            dtype=torch.float32,
        ),
        staging_k=torch.zeros(
            (ACTIVE_CAPACITY, REGION_SIZE, 1, PROXY_DIM),
            device=DEVICE,
            dtype=torch.bfloat16,
        ),
        staging_z=torch.zeros(
            (ACTIVE_CAPACITY, REGION_SIZE, 1, PROXY_DIM),
            device=DEVICE,
            dtype=torch.bfloat16,
        ),
        staging_valid=torch.zeros(
            (ACTIVE_CAPACITY, REGION_SIZE), device=DEVICE, dtype=torch.uint8
        ),
    )


def _slot_mapping(block_table_row, positions):
    return torch.tensor(
        [
            int(block_table_row[p // PAGE_SIZE]) * PAGE_SIZE + p % PAGE_SIZE
            for p in positions
        ],
        device=DEVICE,
        dtype=torch.int64,
    )


def _layout(slot_mapping, qsl, seq_lens, live):
    return build_decode_summary_layout_step3p5(
        slot_mapping,
        qsl,
        seq_lens,
        num_actual_tokens=live,
        num_pages=NUM_PAGES,
        page_size=PAGE_SIZE,
        region_block_size=REGION_SIZE,
        summaries_per_page=SUMMARIES_PER_PAGE,
    )


def _update_prefill(state, slot_mapping, qsl, seq_lens, index_k, index_z, live):
    flat_slot, _, positions, reset, valid = _layout(
        slot_mapping, qsl, seq_lens, live
    )
    rows = flat_slot.numel()
    csa_compact_prefill_update(
        state["mean_cache"],
        state["active_region_ids"],
        state["active_slot_by_region"],
        state["allocation_success"],
        state["numerator"],
        state["denominator"],
        state["max_logits"],
        state["staging_valid"],
        torch.empty((rows,), device=DEVICE, dtype=torch.int32),
        torch.empty((rows * REGION_SIZE,), device=DEVICE, dtype=torch.int32),
        torch.empty((rows,), device=DEVICE, dtype=torch.int32),
        flat_slot,
        reset,
        valid,
        positions,
        index_k.contiguous(),
        index_z.contiguous(),
        live_rows=live,
        num_pages=NUM_PAGES,
        summaries_per_page=SUMMARIES_PER_PAGE,
        proxy_dim=PROXY_DIM,
        region_block_size=REGION_SIZE,
    )
    assert int(state["allocation_success"].item()) == 1


def _prefill_chunk(state, block_table_row, k, z, start, end, seq_len):
    _update_prefill(
        state,
        _slot_mapping(block_table_row, range(start, end)),
        torch.tensor([0, end - start], device=DEVICE, dtype=torch.int32),
        torch.tensor([seq_len], device=DEVICE, dtype=torch.int32),
        k.view(end - start, 1, PROXY_DIM),
        z.view(end - start, 1, PROXY_DIM),
        end - start,
    )


def _ref_means(all_k, all_z):
    """{logical_region: fp8 mean} for every completed region of the stream."""
    out = {}
    for r in range(all_k.shape[0] // REGION_SIZE):
        zr = all_z[r * REGION_SIZE : (r + 1) * REGION_SIZE]
        kr = all_k[r * REGION_SIZE : (r + 1) * REGION_SIZE]
        m = zr.max()
        w = torch.exp(zr - m)
        mean = (w * kr).sum(0) / w.sum(0).clamp_min(1e-20)
        out[r] = mean.to(torch.bfloat16).to(torch.float8_e4m3fn)
    return out


def _check_means(state, block_table_row, ref):
    mean_cache = state["mean_cache"].view(NUM_PAGES * SUMMARIES_PER_PAGE, PROXY_DIM)
    for logical_region, ref_fp8 in ref.items():
        page = int(block_table_row[logical_region // SUMMARIES_PER_PAGE])
        frag = logical_region % SUMMARIES_PER_PAGE
        got = mean_cache[page * SUMMARIES_PER_PAGE + frag]
        assert torch.equal(got.view(torch.uint8), ref_fp8.view(torch.uint8)), (
            f"region {logical_region} mismatch"
        )


def test_layout_builder_matches_torch():
    torch.manual_seed(0)
    block_table = torch.tensor([[3, 5]], device=DEVICE, dtype=torch.int32)
    positions = list(range(20))
    slot_mapping = _slot_mapping(block_table[0], positions)
    qsl = torch.tensor([0, 20], device=DEVICE, dtype=torch.int32)
    seq_lens = torch.tensor([20], device=DEVICE, dtype=torch.int32)

    flat_slot, token_slots, token_positions, reset_slots, token_valid = _layout(
        slot_mapping, qsl, seq_lens, 20
    )

    # Reference: region = page * 2 + (pos % 16) // 8, page = bt[pos // 16].
    ref_flat = torch.tensor(
        [
            int(block_table[0, p // PAGE_SIZE]) * SUMMARIES_PER_PAGE
            + (p % PAGE_SIZE) // REGION_SIZE
            for p in positions
        ],
        device=DEVICE,
        dtype=torch.int64,
    )
    ref_reset = torch.tensor(
        [
            int(ref_flat[p]) if p % REGION_SIZE == 0 else -1
            for p in positions
        ],
        device=DEVICE,
        dtype=torch.int64,
    )
    assert torch.equal(flat_slot, ref_flat)
    assert torch.equal(token_positions, torch.arange(20, device=DEVICE))
    assert torch.equal(reset_slots, ref_reset)
    assert bool(token_valid.all())


def test_prefill_update_matches_reference():
    torch.manual_seed(0)
    state = _alloc_state()
    block_table = torch.tensor([[3, 5]], device=DEVICE, dtype=torch.int32)
    k = torch.randn(20, PROXY_DIM, device=DEVICE, dtype=torch.bfloat16)
    z = torch.randn(20, PROXY_DIM, device=DEVICE, dtype=torch.bfloat16)

    _prefill_chunk(state, block_table[0], k, z, 0, 20, 20)

    ref = _ref_means(k.float(), z.float())
    assert set(ref) == {0, 1}  # region 2 stays pending (4 tokens)
    _check_means(state, block_table[0], ref)


def test_chunked_prefill_prefix_continuation():
    torch.manual_seed(2)
    state = _alloc_state()
    block_table = torch.tensor([[3, 5]], device=DEVICE, dtype=torch.int32)
    all_k = torch.randn(28, PROXY_DIM, device=DEVICE, dtype=torch.bfloat16)
    all_z = torch.randn(28, PROXY_DIM, device=DEVICE, dtype=torch.bfloat16)

    # Chunk 1 completes region 0 and leaves region 1 pending as slot N/D/M.
    _prefill_chunk(state, block_table[0], all_k[:12], all_z[:12], 0, 12, 12)
    # Chunk 2 folds the prefix and completes regions 1 and 2.
    _prefill_chunk(state, block_table[0], all_k[12:], all_z[12:], 12, 28, 28)

    ref = _ref_means(all_k.float(), all_z.float())
    assert set(ref) == {0, 1, 2}
    _check_means(state, block_table[0], ref)


def test_mixed_decode_and_prefill_update():
    """A mixed batch runs the prefill update over decode rows too."""
    torch.manual_seed(3)
    state = _alloc_state()
    block_table = torch.tensor(
        [[3, 5, 0, 0], [1, 7, 0, 0]], device=DEVICE, dtype=torch.int32
    )
    k0 = torch.randn(20, PROXY_DIM, device=DEVICE, dtype=torch.bfloat16)
    z0 = torch.randn(20, PROXY_DIM, device=DEVICE, dtype=torch.bfloat16)
    _prefill_chunk(state, block_table[0], k0, z0, 0, 20, 20)

    # req0 contributes one decode token (pos 20), req1 a 10-token chunk.
    k1 = torch.randn(10, PROXY_DIM, device=DEVICE, dtype=torch.bfloat16)
    z1 = torch.randn(10, PROXY_DIM, device=DEVICE, dtype=torch.bfloat16)
    kd = torch.randn(1, PROXY_DIM, device=DEVICE, dtype=torch.bfloat16)
    zd = torch.randn(1, PROXY_DIM, device=DEVICE, dtype=torch.bfloat16)
    slot_mapping = torch.cat(
        [
            _slot_mapping(block_table[0], [20]),
            _slot_mapping(block_table[1], range(10)),
        ]
    )
    qsl = torch.tensor([0, 1, 11], device=DEVICE, dtype=torch.int32)
    seq_lens = torch.tensor([21, 10], device=DEVICE, dtype=torch.int32)
    _update_prefill(
        state,
        slot_mapping,
        qsl,
        seq_lens,
        torch.cat([kd, k1]).view(11, 1, PROXY_DIM),
        torch.cat([zd, z1]).view(11, 1, PROXY_DIM),
        11,
    )

    ref0 = _ref_means(torch.cat([k0, kd]).float(), torch.cat([z0, zd]).float())
    assert set(ref0) == {0, 1}  # region 2 pending with 5 tokens
    _check_means(state, block_table[0], ref0)
    ref1 = _ref_means(k1.float(), z1.float())
    assert set(ref1) == {0}
    _check_means(state, block_table[1], ref1)


def test_reset_blocks_clears_payload_and_slots():
    torch.manual_seed(4)
    state = _alloc_state()
    block_table = torch.tensor([[3, 5]], device=DEVICE, dtype=torch.int32)
    k = torch.randn(24, PROXY_DIM, device=DEVICE, dtype=torch.bfloat16)
    z = torch.randn(24, PROXY_DIM, device=DEVICE, dtype=torch.bfloat16)
    _prefill_chunk(state, block_table[0], k, z, 0, 24, 24)
    assert int((state["active_region_ids"] >= 0).sum()) == 0  # 3 full regions

    csa_reset_blocks(
        torch.tensor([3], device=DEVICE, dtype=torch.long),
        state["mean_cache"].view(torch.uint8),
        state["active_region_ids"],
        state["active_slot_by_region"],
        state["numerator"],
        state["denominator"],
        state["max_logits"],
        state["staging_valid"],
        num_pages=NUM_PAGES,
        summaries_per_page=SUMMARIES_PER_PAGE,
        proxy_dim=PROXY_DIM,
        region_block_size=REGION_SIZE,
    )

    mean_cache = state["mean_cache"].view(NUM_PAGES * SUMMARIES_PER_PAGE, PROXY_DIM)
    # Page 3 (logical regions 0-1) is cleared; page 5 (region 2) is not.
    assert int(mean_cache[3 * SUMMARIES_PER_PAGE : 3 * SUMMARIES_PER_PAGE + 2].view(torch.uint8).abs().sum()) == 0
    assert int(mean_cache[5 * SUMMARIES_PER_PAGE].view(torch.uint8).abs().sum()) != 0
