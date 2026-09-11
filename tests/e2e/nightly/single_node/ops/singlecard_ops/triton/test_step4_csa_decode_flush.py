#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
"""NPU test: Step4 CSA decode stage-flush kernel vs torch reference.

Exercises ``_csa_decode_stage_flush_kernel`` through its wrapper
``csa_compact_decode_update`` with a hand-built runtime state -- no layout
builder and no prefill update involved:

* the active slot carries the prefill-left running state for the region's
  first tokens (fp32 numerator/denominator/max, online-softmax form),
* the staging buffer holds raw decode tokens for the middle offsets,
* the completing token (offset ``region_block_size - 1``) arrives as the
  row's index_k/index_z.

The flush must fold all three parts with one shared scalar max, store the
fp8 mean through the fp32 -> bf16 -> e4m3 double rounding, and release the
active slot. The reference recomputes the mean over the full 8-token region
from the raw stream.
"""

import torch

from vllm_ascend.models.step4.summary_kernels import csa_compact_decode_update

DEVICE = "npu"

NUM_PAGES = 8
SUMMARIES_PER_PAGE = 2
PROXY_DIM = 256
ACTIVE_CAPACITY = 16
REGION_BLOCK_SIZE = 8

REGION = 2  # -> page 1, fragment 0 of the block table below
PAGE_OF_REGION = 1
SLOT = 3


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
            (NUM_PAGES * SUMMARIES_PER_PAGE,), -1, device=DEVICE, dtype=torch.int32
        ),
        allocation_success=torch.ones((1,), device=DEVICE, dtype=torch.int32),
        numerator=torch.zeros(
            (ACTIVE_CAPACITY, 1, PROXY_DIM), device=DEVICE, dtype=torch.float32
        ),
        denominator=torch.zeros(
            (ACTIVE_CAPACITY, 1, PROXY_DIM), device=DEVICE, dtype=torch.float32
        ),
        max_logits=torch.full(
            (ACTIVE_CAPACITY, 1), float("-inf"), device=DEVICE, dtype=torch.float32
        ),
        staging_k=torch.zeros(
            (ACTIVE_CAPACITY, REGION_BLOCK_SIZE, 1, PROXY_DIM),
            device=DEVICE,
            dtype=torch.bfloat16,
        ),
        staging_z=torch.zeros(
            (ACTIVE_CAPACITY, REGION_BLOCK_SIZE, 1, PROXY_DIM),
            device=DEVICE,
            dtype=torch.bfloat16,
        ),
        staging_valid=torch.zeros(
            (ACTIVE_CAPACITY, REGION_BLOCK_SIZE), device=DEVICE, dtype=torch.uint8
        ),
    )


def _run_decode(state, region, position, k_t, z_t):
    csa_compact_decode_update(
        state["mean_cache"],
        state["active_region_ids"],
        state["active_slot_by_region"],
        state["allocation_success"],
        state["numerator"],
        state["denominator"],
        state["max_logits"],
        state["staging_k"],
        state["staging_z"],
        state["staging_valid"],
        # int64 to match the layout builder's output dtypes; the kernel's
        # slot redefinition in the allocation path relies on it.
        torch.tensor([region], device=DEVICE, dtype=torch.int64),    # flat_slot
        torch.tensor([-1], device=DEVICE, dtype=torch.int64),        # reset_slots
        torch.ones((1,), device=DEVICE, dtype=torch.int32),          # token_valid
        torch.tensor([position], device=DEVICE, dtype=torch.int64),  # token_positions
        k_t.contiguous().view(1, 1, PROXY_DIM),                      # index_k
        z_t.contiguous().view(1, 1, PROXY_DIM),                      # index_z
        live_rows=1,
        num_pages=NUM_PAGES,
        summaries_per_page=SUMMARIES_PER_PAGE,
        proxy_dim=PROXY_DIM,
        region_block_size=REGION_BLOCK_SIZE,
    )
    assert int(state["allocation_success"].item()) == 1


def test_decode_stage_allocates_and_stages_non_completing_token():
    """A token below the region boundary allocates a slot and stages raw."""
    torch.manual_seed(2)
    state = _alloc_state()
    region, slot_hint = 2, 3
    offset = 5  # < REGION_BLOCK_SIZE - 1: non-completing
    k_t = torch.randn(1, PROXY_DIM, device=DEVICE, dtype=torch.bfloat16)
    z_t = torch.randn(1, PROXY_DIM, device=DEVICE, dtype=torch.bfloat16)

    # Case 1: fresh region -- the CAS probe allocates a slot from empty.
    _run_decode(state, region, offset, k_t, z_t)
    slots = (state["active_region_ids"] == region).nonzero().flatten()
    assert slots.numel() == 1
    slot = int(slots[0])
    assert int(state["active_slot_by_region"][region]) == slot
    assert bool(
        torch.equal(state["staging_k"][slot, offset, 0].to(torch.bfloat16), k_t[0])
    )
    assert bool(
        torch.equal(state["staging_z"][slot, offset, 0].to(torch.bfloat16), z_t[0])
    )
    assert int(state["staging_valid"][slot].sum()) == 1
    assert int(state["staging_valid"][slot, offset]) == 1
    # No fp8 mean written yet and no runtime state touched.
    assert int(state["mean_cache"].view(torch.uint8).abs().sum()) == 0
    assert float(state["numerator"].abs().sum()) == 0.0
    assert bool(state["max_logits"][:, 0].isinf().all())

    # Case 2: continuation into a pre-assigned slot keeps the maps stable
    # and stages the next offset without merging the prefix state.
    state["active_region_ids"][slot] = -1
    state["active_slot_by_region"][region] = slot_hint
    state["active_region_ids"][slot_hint] = region
    prefix_num = torch.randn(PROXY_DIM, device=DEVICE)
    state["numerator"][slot_hint, 0] = prefix_num
    k_t2 = torch.randn(1, PROXY_DIM, device=DEVICE, dtype=torch.bfloat16)
    z_t2 = torch.randn(1, PROXY_DIM, device=DEVICE, dtype=torch.bfloat16)
    _run_decode(state, region, offset + 1, k_t2, z_t2)
    assert int(state["active_region_ids"][slot_hint]) == region
    assert int(state["active_slot_by_region"][region]) == slot_hint
    assert bool(
        torch.equal(state["staging_k"][slot_hint, offset + 1, 0].to(torch.bfloat16), k_t2[0])
    )
    assert int(state["staging_valid"][slot_hint].sum()) == 1  # only the new offset
    assert bool(torch.equal(state["numerator"][slot_hint, 0], prefix_num))
    assert int(state["mean_cache"].view(torch.uint8).abs().sum()) == 0


def test_decode_stage_flush_merges_prefix_and_staged_tokens():
    torch.manual_seed(1)
    state = _alloc_state()
    # One region of 8 tokens: offsets 0..3 prefill prefix, 4..6 staged
    # decode tokens, 7 the completing decode token.
    k = torch.randn(REGION_BLOCK_SIZE, PROXY_DIM, device=DEVICE, dtype=torch.bfloat16)
    z = torch.randn(REGION_BLOCK_SIZE, PROXY_DIM, device=DEVICE, dtype=torch.bfloat16)
    kf, zf = k.float(), z.float()

    # Active slot: prefill-left running state over offsets 0..3.
    m_pre = zf[:4].max()
    w_pre = torch.exp(zf[:4] - m_pre)
    state["active_region_ids"][SLOT] = REGION
    state["active_slot_by_region"][REGION] = SLOT
    state["numerator"][SLOT, 0] = (w_pre * kf[:4]).sum(0)
    state["denominator"][SLOT, 0] = w_pre.sum(0)
    state["max_logits"][SLOT, 0] = m_pre
    # Staged raw decode tokens for offsets 4..6.
    state["staging_k"][SLOT, 4:7, 0] = k[4:7]
    state["staging_z"][SLOT, 4:7, 0] = z[4:7]
    state["staging_valid"][SLOT, 4:7] = 1

    _run_decode(state, REGION, REGION_BLOCK_SIZE - 1, k[7:], z[7:])

    # Reference: mean over the full region with one shared max, stored
    # through the fp32 -> bf16 -> e4m3 double rounding.
    m = zf.max()
    w = torch.exp(zf - m)
    ref = ((w * kf).sum(0) / w.sum(0)).to(torch.bfloat16).to(torch.float8_e4m3fn)
    got = state["mean_cache"][PAGE_OF_REGION, REGION % SUMMARIES_PER_PAGE, 0]
    assert torch.equal(got.view(torch.uint8), ref.view(torch.uint8))

    # The flush released the slot and cleared its runtime state.
    assert int(state["active_region_ids"][SLOT]) == -1
    assert int(state["active_slot_by_region"][REGION]) == -1
    assert int(state["staging_valid"][SLOT].sum()) == 0
    assert float(state["numerator"][SLOT].abs().sum()) == 0.0
    assert bool(state["max_logits"][SLOT, 0].isinf())
