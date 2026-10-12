# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm_ascend.models.step5.generic_block_sparse_attention import (
    _append_current_region,
)


@pytest.mark.parametrize("groups", [1, 2, 4])
@pytest.mark.parametrize("tokens,topk", [(1, 1), (3, 7), (5, 513)])
def test_append_current_region_cpu(groups: int, tokens: int, topk: int) -> None:
    rows = groups * tokens
    ids = torch.full((rows, topk), -1, dtype=torch.int32)
    live_counts = torch.tensor(
        [0 if row % 3 == 0 else topk if row % 3 == 1 else topk // 2 for row in range(rows)],
        dtype=torch.int32,
    )
    for row, live in enumerate(live_counts.tolist()):
        ids[row, :live] = torch.arange(live, dtype=torch.int32)

    positions = torch.arange(tokens, dtype=torch.int64) * 16 + 32
    original_ids = ids.clone()
    original_positions = positions.clone()

    packed, counts = _append_current_region(
        ids, positions, groups=groups, region_size=8
    )

    assert packed.shape == (groups, tokens, topk + 1)
    assert counts.shape == (groups, tokens)
    assert packed.dtype == ids.dtype
    assert counts.dtype == torch.int32
    torch.testing.assert_close(counts.reshape(-1), live_counts + 1)
    packed_rows = packed.reshape(rows, topk + 1)
    for row, live in enumerate(live_counts.tolist()):
        assert packed_rows[row, live].item() == (
            positions[row % tokens] // 8
        ).item()
        assert torch.all(packed_rows[row, live + 1 :] == -1)
    torch.testing.assert_close(ids, original_ids)
    torch.testing.assert_close(positions, original_positions)
