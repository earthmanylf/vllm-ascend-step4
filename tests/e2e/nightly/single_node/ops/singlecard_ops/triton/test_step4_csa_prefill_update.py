#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
"""NPU test: standalone CSA prefill summary update kernel vs torch reference.

Covers ``csa_compact_prefill_update`` through the real layout builder
(``decode_summary_layout_step3p5``):
page-backed fp8 region summaries written with the fp32 -> bf16 -> e4m3
double rounding, online-softmax running state for a region in progress,
and the active-slot runtime state.

Scenarios:

* a single prefill chunk completing whole regions,
* a chunked prefill whose second chunk continues the running state the
  first chunk left in the open region,
* a mixed batch where one request's decode token runs through the prefill
  update next to another request's fresh chunk.

The reference recomputes region means from the raw token stream:
``s_d = sum_t exp(z_td - m) * k_td / sum_t exp(z_td - m)`` with one scalar
``m`` per region.

Usage:
    # correctness tests (pytest)
    pytest test_step4_csa_prefill_update.py -v

    # timing benchmark
    python test_step4_csa_prefill_update.py --tokens 20,128,512
"""

from __future__ import annotations

import argparse
import os
import time

import pytest
import torch

from vllm_ascend.models.step4.summary_kernels import (
    build_decode_summary_layout_step3p5,
    csa_compact_prefill_update,
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
        staging_valid=torch.zeros(
            (ACTIVE_CAPACITY, REGION_SIZE), device=DEVICE, dtype=torch.uint8
        ),
    )


def _alloc_state_scaled(num_pages, total_regions, active_capacity):
    """Allocate state for large token counts (parameterized capacities)."""
    return dict(
        mean_cache=torch.zeros(
            (num_pages, SUMMARIES_PER_PAGE, 1, PROXY_DIM),
            device=DEVICE, dtype=torch.float8_e4m3fn,
        ),
        active_region_ids=torch.full(
            (active_capacity,), -1, device=DEVICE, dtype=torch.int32,
        ),
        active_slot_by_region=torch.full(
            (total_regions,), -1, device=DEVICE, dtype=torch.int32,
        ),
        allocation_success=torch.ones((1,), device=DEVICE, dtype=torch.int32),
        numerator=torch.zeros(
            (active_capacity, 1, PROXY_DIM), device=DEVICE, dtype=torch.float32,
        ),
        denominator=torch.zeros(
            (active_capacity, 1, PROXY_DIM), device=DEVICE, dtype=torch.float32,
        ),
        max_logits=torch.full(
            (active_capacity, 1), float("-inf"), device=DEVICE, dtype=torch.float32,
        ),
        staging_valid=torch.zeros(
            (active_capacity, REGION_SIZE), device=DEVICE, dtype=torch.uint8,
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

    # Chunk 1 leaves region 1 open with 4 tokens; chunk 2 folds the prefix
    # and completes regions 1 and 2.
    _prefill_chunk(state, block_table[0], all_k[:12], all_z[:12], 0, 12, 12)
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


def _test_large_prefill(num_tokens: int) -> None:
    """Large-scale prefill correctness: all regions complete, check every mean."""
    torch.manual_seed(42)
    pages_needed = (num_tokens + PAGE_SIZE - 1) // PAGE_SIZE
    num_pages = max(NUM_PAGES, pages_needed)
    total_regions = num_pages * SUMMARIES_PER_PAGE
    active_capacity = max(ACTIVE_CAPACITY, total_regions)

    block_table = torch.arange(num_pages, device=DEVICE, dtype=torch.int32).unsqueeze(0)
    k = torch.randn(num_tokens, PROXY_DIM, device=DEVICE, dtype=torch.bfloat16)
    z = torch.randn(num_tokens, PROXY_DIM, device=DEVICE, dtype=torch.bfloat16)

    slot_mapping = _slot_mapping(block_table[0], range(num_tokens))
    qsl = torch.tensor([0, num_tokens], device=DEVICE, dtype=torch.int32)
    seq_lens = torch.tensor([num_tokens], device=DEVICE, dtype=torch.int32)

    flat_slot, _, positions, reset, valid = build_decode_summary_layout_step3p5(
        slot_mapping, qsl, seq_lens,
        num_actual_tokens=num_tokens,
        num_pages=num_pages,
        page_size=PAGE_SIZE,
        region_block_size=REGION_SIZE,
        summaries_per_page=SUMMARIES_PER_PAGE,
    )
    rows = flat_slot.numel()

    state = _alloc_state_scaled(num_pages, total_regions, active_capacity)

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
        k.contiguous(),
        z.contiguous(),
        live_rows=num_tokens,
        num_pages=num_pages,
        summaries_per_page=SUMMARIES_PER_PAGE,
        proxy_dim=PROXY_DIM,
        region_block_size=REGION_SIZE,
    )
    assert int(state["allocation_success"].item()) == 1

    # Check all fully-completed regions (those with REGION_SIZE tokens)
    num_complete = num_tokens // REGION_SIZE
    mean_cache = state["mean_cache"].view(
        num_pages * SUMMARIES_PER_PAGE, PROXY_DIM
    )
    bt = block_table[0]
    mismatches = 0
    for r in range(num_complete):
        zr = z[r * REGION_SIZE : (r + 1) * REGION_SIZE].float()
        kr = k[r * REGION_SIZE : (r + 1) * REGION_SIZE].float()
        m = zr.max()
        w = torch.exp(zr - m)
        ref_mean = (
            (w * kr).sum(0) / w.sum(0).clamp_min(1e-20)
        ).to(torch.bfloat16).to(torch.float8_e4m3fn)

        page = int(bt[r // SUMMARIES_PER_PAGE])
        frag = r % SUMMARIES_PER_PAGE
        got = mean_cache[page * SUMMARIES_PER_PAGE + frag]
        if not torch.equal(got.view(torch.uint8), ref_mean.view(torch.uint8)):
            diff = (got.float() - ref_mean.float()).abs()
            max_diff = diff.max().item()
            mismatches += 1
            if mismatches <= 3:
                print(f"    region {r} mismatch (max_abs_diff={max_diff:.2e})")
    # Allow rare fp8 rounding edge cases (< 0.1% of regions)
    assert mismatches <= num_complete // 1000 + 1, (
        f"{mismatches}/{num_complete} regions mismatched"
    )
    print(f"  large prefill {num_tokens} tokens: {num_complete} regions verified OK")


@pytest.mark.skipif(
    not torch.npu.is_available(), reason="NPU required for large prefill test"
)
def test_large_prefill_16k():
    _test_large_prefill(16384)


@pytest.mark.skipif(
    not torch.npu.is_available(), reason="NPU required for large prefill test"
)
def test_large_prefill_32k():
    _test_large_prefill(32768)


@pytest.mark.skipif(
    not torch.npu.is_available(), reason="NPU required for large prefill test"
)
def test_large_prefill_64k():
    _test_large_prefill(65536)


# ---- performance benchmark infrastructure --------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))

TOKEN_TABLE: list[dict[str, int]] = [
    {"num_tokens": 8},
    {"num_tokens": 20},
    {"num_tokens": 64},
    {"num_tokens": 128},
    {"num_tokens": 256},
    {"num_tokens": 512},
    {"num_tokens": 1024},
    {"num_tokens": 4096},
]


def _synchronize() -> None:
    if DEVICE == "npu":
        torch.npu.synchronize()
    elif DEVICE == "cuda":
        torch.cuda.synchronize()


def _bench(fn, *, warmup: int = 10, iters: int = 50) -> float:
    """Average wall-clock time of a single ``fn()`` call, in milliseconds."""
    for _ in range(warmup):
        fn()
    _synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    _synchronize()
    return (time.perf_counter() - start) / iters * 1e3


def _make_bench_case(num_tokens: int):
    """Create a self-contained benchmark callable for a single prefill chunk."""
    torch.manual_seed(42)
    pages_needed = (num_tokens + PAGE_SIZE - 1) // PAGE_SIZE
    num_pages = max(NUM_PAGES, pages_needed)
    total_regions = num_pages * SUMMARIES_PER_PAGE
    active_capacity = max(ACTIVE_CAPACITY, total_regions)

    block_table = torch.arange(num_pages, device=DEVICE, dtype=torch.int32).unsqueeze(0)
    k = torch.randn(num_tokens, PROXY_DIM, device=DEVICE, dtype=torch.bfloat16)
    z = torch.randn(num_tokens, PROXY_DIM, device=DEVICE, dtype=torch.bfloat16)

    slot_mapping = _slot_mapping(block_table[0], range(num_tokens))
    qsl = torch.tensor([0, num_tokens], device=DEVICE, dtype=torch.int32)
    seq_lens = torch.tensor([num_tokens], device=DEVICE, dtype=torch.int32)

    def fn():
        state = dict(
            mean_cache=torch.zeros(
                (num_pages, SUMMARIES_PER_PAGE, 1, PROXY_DIM),
                device=DEVICE, dtype=torch.float8_e4m3fn,
            ),
            active_region_ids=torch.full(
                (active_capacity,), -1, device=DEVICE, dtype=torch.int32,
            ),
            active_slot_by_region=torch.full(
                (total_regions,), -1, device=DEVICE, dtype=torch.int32,
            ),
            allocation_success=torch.ones((1,), device=DEVICE, dtype=torch.int32),
            numerator=torch.zeros(
                (active_capacity, 1, PROXY_DIM), device=DEVICE, dtype=torch.float32,
            ),
            denominator=torch.zeros(
                (active_capacity, 1, PROXY_DIM), device=DEVICE, dtype=torch.float32,
            ),
            max_logits=torch.full(
                (active_capacity, 1), float("-inf"), device=DEVICE, dtype=torch.float32,
            ),
            staging_valid=torch.zeros(
                (active_capacity, REGION_SIZE), device=DEVICE, dtype=torch.uint8,
            ),
        )
        flat_slot, _, positions, reset, valid = build_decode_summary_layout_step3p5(
            slot_mapping, qsl, seq_lens,
            num_actual_tokens=num_tokens,
            num_pages=num_pages,
            page_size=PAGE_SIZE,
            region_block_size=REGION_SIZE,
            summaries_per_page=SUMMARIES_PER_PAGE,
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
            k.contiguous(),
            z.contiguous(),
            live_rows=num_tokens,
            num_pages=num_pages,
            summaries_per_page=SUMMARIES_PER_PAGE,
            proxy_dim=PROXY_DIM,
            region_block_size=REGION_SIZE,
        )

    desc = f"num_tokens={num_tokens}  proxy_dim={PROXY_DIM}  regions={total_regions}"
    return fn, desc


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CSA compact prefill update performance test -- Ascend version"
    )
    parser.add_argument(
        "--tokens",
        default="",
        help="comma-separated num_tokens values to run (e.g. 20,128,512); empty = all",
    )
    parser.add_argument("--warmup", type=int, default=10, help="warmup iterations")
    parser.add_argument("--iters", type=int, default=50, help="timed iterations")
    return parser


def _filter_cases(tokens_arg: str) -> list[dict[str, int]]:
    if not tokens_arg:
        return list(TOKEN_TABLE)
    wanted = [int(x) for x in tokens_arg.split(",") if x.strip()]
    result = [c for c in TOKEN_TABLE if c["num_tokens"] in wanted]
    known = {c["num_tokens"] for c in TOKEN_TABLE}
    for w in wanted:
        if w not in known:
            result.append({"num_tokens": w})
    return result


def main() -> None:
    args = _build_parser().parse_args()

    print(
        f"csa_compact_prefill_update (Ascend)   device={DEVICE}  "
        f"proxy_dim={PROXY_DIM}  region_block_size={REGION_SIZE}"
    )
    print(f"{'num_tokens':>12} {'ms/iter':>10}")
    print("-" * 26)
    for case in _filter_cases(args.tokens):
        num_tokens = case["num_tokens"]
        fn, desc = _make_bench_case(num_tokens)
        print(f"  {desc}")
        ms = _bench(fn, warmup=args.warmup, iters=args.iters)
        print(f"{num_tokens:>12} {ms:>10.4f}")
        _synchronize()


if __name__ == "__main__":
    main()
