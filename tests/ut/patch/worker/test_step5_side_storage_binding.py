# SPDX-License-Identifier: Apache-2.0
"""Both Ascend binding entry points must preserve SWF sidecar lifecycle."""

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.patch.worker import patch_bind_kv_cache as module


class SideStorage:
    main_layer_name = "layers.0.attn"

    def __init__(self):
        self.calls = 0

    def bind_kv_cache_side_storage(self, context):
        assert all(context[name].kv_cache is value for name, value in self.expected.items())
        self.calls += 1

    def zero_kv_cache_side_storage(self, block_ids):
        pass

    def reset_kv_cache_side_storage_runtime_state(self):
        return 1, 0

    def copy_kv_cache_side_storage(self, block_copies, num_blocks):
        pass


@pytest.mark.parametrize("legacy", [False, True])
def test_bind_all_owners_before_side_storage_once(monkeypatch, legacy):
    caches = {f"layers.{i}.attn": (torch.zeros(2, 8, 1, 16), torch.zeros(2, 8, 1, 16)) for i in range(2)}
    side = SideStorage()
    side.expected = caches
    context = {name: SimpleNamespace(kv_cache=None) for name in caches}
    context.update(summary=side, alias=side)
    ring_calls = []
    monkeypatch.setattr(module.utils, "share_replayssm_ring_trackers", lambda *args: ring_calls.append(args))
    if legacy:
        runner_caches = []
        module.bind_kv_cache(caches, context, runner_caches)
        assert len(runner_caches) == 2
    else:
        module.bind_kv_cache_to_layers(caches, context)
    assert side.calls == 1
    assert len(ring_calls) == 1


@pytest.mark.parametrize("legacy", [False, True])
def test_bind_failure_is_not_hidden(monkeypatch, legacy):
    side = SideStorage()

    def fail(context):
        raise RuntimeError("sidecar failed")

    side.bind_kv_cache_side_storage = fail
    monkeypatch.setattr(module.utils, "share_replayssm_ring_trackers", lambda *args: None)
    with pytest.raises(RuntimeError, match="sidecar failed"):
        if legacy:
            module.bind_kv_cache({}, {"summary": side}, [])
        else:
            module.bind_kv_cache_to_layers({}, {"summary": side})
