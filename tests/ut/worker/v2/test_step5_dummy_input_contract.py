# SPDX-License-Identifier: Apache-2.0
"""Profile dummy tokens must reach experts; graph padding must remain masked."""

import numpy as np
import pytest
import torch

from vllm_ascend.worker.v2 import input_batch as module


@pytest.mark.parametrize("is_padding", [False, True])
@pytest.mark.parametrize("physical_tokens", [5, 8])
def test_dummy_profile_and_graph_padding_keep_distinct_masks(monkeypatch, is_padding, physical_tokens):
    # Use the real Ascend buffers and the installed SWF make_dummy on CPU.
    # Rotary table setup is a device operation unrelated to this mask contract.
    positions = []
    monkeypatch.setattr(module, "update_cos_sin", lambda value: positions.append(value.clone()))
    buffers = module.AscendInputBuffers(max_num_reqs=2, max_num_tokens=8, device=torch.device("cpu"))
    buffers.dummy_num_tokens = 5
    buffers.is_padding.fill_(not is_padding)
    buffers.input_ids.fill_(99)
    buffers.positions.fill_(99)

    batch = module.AscendInputBatch.make_dummy(
        num_reqs=2,
        num_tokens=physical_tokens,
        input_buffers=buffers,
        max_query_len=3,
        is_padding=is_padding,
    )

    assert batch.num_tokens == 5
    assert batch.num_tokens_after_padding == physical_tokens
    assert batch.is_padding.tolist() == [is_padding] * 5 + [True] * (physical_tokens - 5)
    assert batch.input_ids.tolist() == [0] * physical_tokens
    assert batch.positions.tolist() == [0] * physical_tokens
    np.testing.assert_array_equal(batch.num_scheduled_tokens, [2, 3])
    np.testing.assert_array_equal(batch.seq_lens_np, [2, 3])
    assert len(positions) == 1
    assert batch.is_dummy


def test_dummy_warmup_defaults_to_padding(monkeypatch):
    monkeypatch.setattr(module, "update_cos_sin", lambda value: None)
    buffers = module.AscendInputBuffers(max_num_reqs=2, max_num_tokens=8, device=torch.device("cpu"))
    batch = module.AscendInputBatch.make_dummy(2, 5, buffers, max_query_len=3)
    assert batch.is_padding.tolist() == [True] * 5
