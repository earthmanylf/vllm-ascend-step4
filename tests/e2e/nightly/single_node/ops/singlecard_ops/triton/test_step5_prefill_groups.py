# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

pytest.importorskip("torch_npu", exc_type=ModuleNotFoundError)
if not torch.npu.is_available():
    pytest.skip("Step5 prefill groups require an NPU", allow_module_level=True)

from vllm_ascend.models.step5.summary_kernels import build_prefill_groups

REGION_SIZE = 8
START_POSITION = 23


@pytest.mark.parametrize("length", [9, 1025])
@torch.inference_mode()
def test_single_request_prefill_groups(length):
    # Exactly one request: an empty companion would hide the zero-trip compiler bug.
    positions = torch.arange(START_POSITION, START_POSITION + length, dtype=torch.int64)
    slots = positions.to("npu")
    query_start_loc = torch.tensor([0, length], dtype=torch.int32, device="npu")
    metadata = build_prefill_groups(slots, slots, query_start_loc)
    torch.npu.synchronize()

    group_count = length // REGION_SIZE + 2
    expected = torch.full((group_count, REGION_SIZE + 2), -1, dtype=torch.int64)
    expected[:, 1] = REGION_SIZE
    for row, position in enumerate(positions.tolist()):
        group = (position - START_POSITION + START_POSITION % REGION_SIZE) // REGION_SIZE
        offset = position % REGION_SIZE
        expected[group, 0] = position // REGION_SIZE
        expected[group, 1] = min(expected[group, 1].item(), offset)
        expected[group, 2 + offset] = row

    assert metadata.shape == expected.shape
    assert metadata.dtype == torch.int64
    assert metadata.device == slots.device
    torch.testing.assert_close(metadata.cpu(), expected, rtol=0, atol=0)
