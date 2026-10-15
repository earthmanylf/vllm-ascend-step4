# SPDX-License-Identifier: Apache-2.0
"""CPU checks of the real fixed SWF cache and Ascend lifecycle adapter.

Load class definitions without initializing the accelerator runtime. Allocation,
reset, COW and CPU accumulation are the actual upstream implementations. Only
NPU dispatch and logging are mocked.
"""

import ast
import importlib.util
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[4]
MODEL_DIR = ROOT / "vllm_ascend/models/step5"


def _definitions(path, names, namespace, methods=None):
    tree = ast.parse(path.read_text())
    selected = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    if methods is not None:
        for node in selected:
            if isinstance(node, ast.ClassDef):
                node.bases = []
                node.body = [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name in methods]
    tree.body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *selected]
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), namespace)


def _swf_source():
    local = ROOT.parent / "swf-review/vllm/models/step5/nvidia/sparse_summary_cache.py"
    if local.exists():
        return local
    spec = importlib.util.find_spec("vllm")
    return Path(spec.origin).parent / "models/step5/nvidia/sparse_summary_cache.py"


@pytest.fixture
def api(monkeypatch):
    namespace = dict(
        __name__=__name__,
        torch=torch,
        np=np,
        math=math,
        dataclass=dataclass,
        field=field,
        logger=Mock(),
        AttentionLayerBase=type("AttentionLayerBase", (), {}),
        AttentionBackend=object,
        _is_graph_capturing=lambda: False,
    )
    _definitions(_swf_source(), {"Step5SparseSummaryCache", "Step5SparseSummaryCacheLayer"}, namespace)
    namespace["UpstreamCache"] = namespace["Step5SparseSummaryCache"]
    _definitions(
        MODEL_DIR / "_step5_compat.py",
        {"Step5SparseSummaryCacheConfig", "Step5DSAScratchWorkspace", "step5_sparse_summary_cache_bytes_per_block"},
        namespace,
    )
    _definitions(
        MODEL_DIR / "summary_cache.py",
        {"AscendStep5SparseSummaryCache", "AscendStep5SparseSummaryCacheLayer"},
        namespace,
    )
    _definitions(
        MODEL_DIR / "dsa_attention.py",
        {"AscendStep5DSAImpl"},
        namespace,
        methods={
            "__init__",
            "sparse_region_block_size",
            "bind_scratch_workspace",
            "_get_dsa_tensor_buffer_at_least",
            "bind_summary_cache",
            "_round_up",
        },
    )
    namespace["AscendStep5DSAImpl"]._dsa_virtual_engine = lambda self: 0
    transfer = ModuleType("vllm.distributed.kv_transfer")
    transfer.has_kv_transfer_group = lambda: False
    monkeypatch.setitem(sys.modules, transfer.__name__, transfer)
    block_utils = ModuleType("vllm.v1.core.kv_cache_utils")
    block_utils.KVCacheBlockCopy = SimpleNamespace
    monkeypatch.setitem(sys.modules, block_utils.__name__, block_utils)
    kernels = ModuleType("vllm_ascend.models.step5.summary_kernels")
    for name in ("update_prefill", "update_decode", "update_request_journal", "make_prefill_metadata_from_slots"):
        setattr(kernels, name, Mock(name=name))
    monkeypatch.setitem(sys.modules, kernels.__name__, kernels)
    return SimpleNamespace(**namespace, kernels=kernels)


def _layer(api, context, index=0, max_seqs=2, max_tokens=16, pages=4, spec=0):
    impl = api.AscendStep5DSAImpl(
        2,
        128,
        128**-0.5,
        1,
        topk_regions=2,
        region_size=8,
        proxy_dim=256,
        max_num_seqs=max_seqs,
        max_model_len=64,
        block_size=16,
        max_num_batched_tokens=max_tokens,
        decode_out_dtype=torch.bfloat16,
        num_speculative_tokens=spec,
    )
    layer = api.AscendStep5SparseSummaryCacheLayer(
        prefix=f"layers.{index}.summary",
        target_impl=impl,
        sparse_config=SimpleNamespace(proxy_dim=256),
        main_layer_name=f"layers.{index}.attn",
        static_forward_context=context,
    )
    layer._set_summary_cache_budget(16)
    # Owning KV may be virtually split; metadata still addresses scheduler pages.
    owner_k = torch.empty(pages * 2, 8, 1, 128, dtype=torch.bfloat16)
    context[layer.main_layer_name] = SimpleNamespace(kv_cache=(owner_k, torch.empty_like(owner_k)))
    return layer


def _bind(context, layer):
    layer.prepare_kv_cache_side_storage_memory_profile(context, torch.device("cpu"))
    layer.bind_kv_cache_side_storage(context)
    return layer._cache


def _copy(src, dst):
    return SimpleNamespace(src_block_id=src, dst_block_id=dst)


def _bytes(cache):
    names = ["_mean", "numerator", "denominator", "maximum", "last_slot"]
    if cache.request_journal_width:
        names += ["request_journal_values", "request_journal_last", "request_journal_tag"]
    return sum(getattr(cache, n).numel() * getattr(cache, n).element_size() for n in names)


@pytest.mark.parametrize("spec", [0, 1, 2])
def test_bind_uses_swf_storage_and_exact_budget(api, spec):
    context = {}
    layer = _layer(api, context, spec=spec)
    cache = _bind(context, layer)
    assert isinstance(cache, api.UpstreamCache)
    assert cache.num_pages == 4 and cache.page_size == 16
    assert cache.mean_cache.shape == (4, 2, 1, 256)
    assert cache.numerator.shape == cache.denominator.shape == (6, 256)
    assert cache.last_slot.dtype == torch.int64
    assert cache.mean_cache.data_ptr() == cache.mean.data_ptr() == layer.kv_cache.data_ptr()
    assert layer._budget_only_cache is cache is layer._target_impl._summary_cache
    assert layer._set_summary_cache_budget(16) == 512 + 2060
    assert api.step5_sparse_summary_cache_bytes_per_block(
        block_size=16, region_block_size=8, proxy_dim=256
    ) == layer._set_summary_cache_budget(16)
    assert _bytes(cache) == 4 * layer._set_summary_cache_budget(16) + layer._fixed_runtime_state_budget(16)
    assert not any("active_slot" in n or "active_token" in n or "reverse_map" in n for n in vars(cache))
    assert cache.request_journal_width == (spec + 1 if spec else 0)


def test_rejects_unsupported_proxy_contract(api):
    for args in ((1, 15, 8, 1, 256), (1, 16, 4, 1, 256), (1, 16, 8, 2, 256), (1, 16, 8, 1, 128)):
        with pytest.raises(ValueError):
            api.Step5SparseSummaryCacheConfig(*args)


def test_profile_shares_scratch_and_preserves_bound_addresses(api):
    context = {}
    first = _layer(api, context, index=0, max_seqs=1)
    second = _layer(api, context, index=1, max_seqs=3, max_tokens=192)
    first.prepare_kv_cache_side_storage_memory_profile(context, torch.device("cpu"))
    workspace = first._dsa_scratch_workspace
    assert workspace is second._dsa_scratch_workspace
    assert workspace.memory_profiled and workspace.allocations_locked
    assert first._target_impl._summary_cache is None
    ptrs = {n: t.data_ptr() for n, t in workspace.tensor_buffers_by_engine[0].items()}
    second.prepare_kv_cache_side_storage_memory_profile(context, torch.device("cpu"))
    first.bind_kv_cache_side_storage(context)
    second.bind_kv_cache_side_storage(context)
    assert ptrs == {n: t.data_ptr() for n, t in workspace.tensor_buffers_by_engine[0].items()}
    assert first._cache.mean.data_ptr() != second._cache.mean.data_ptr()
    with pytest.raises(RuntimeError, match="capacity was exceeded"):
        second._target_impl._get_dsa_tensor_buffer_at_least(
            "late", (1,), device=torch.device("cpu"), dtype=torch.float32
        )


def test_release_preserves_mean_and_invalidates_all_request_tags(api):
    context = {}
    layer = _layer(api, context, spec=2)
    cache = _bind(context, layer)
    cache.mean[1].fill_(13)
    cache.mean[2].fill_(27)
    cache.last_slot[1:3] = torch.tensor([18, 35])
    # Same physical page tags across two persistent request slots.
    cache.request_journal_tag.copy_(torch.tensor([16, 17, 32, 18, 34, 35]))
    cache._speculative_tails[1] = {18: ()}
    layer.release_kv_cache_side_storage([1])
    assert torch.all(cache.mean[1] == 13) and cache.last_slot[1] == -1
    assert cache.last_slot[2] == 35 and torch.all(cache.mean[2] == 27)
    assert cache.request_journal_tag.tolist() == [-1, -1, 32, -1, 34, 35]
    assert 1 not in cache._speculative_tails
    layer.zero_kv_cache_side_storage(torch.tensor([1]))
    assert not cache.mean[1].any() and torch.all(cache.mean[2] == 27)
    assert cache.request_journal_tag.tolist() == [-1, -1, 32, -1, 34, 35]


def test_cow_snapshots_overlapping_sources_and_rebases_last(api):
    context = {}
    layer = _layer(api, context, spec=2)
    cache = _bind(context, layer)
    for page in range(4):
        cache.mean[page].fill_(page + 10)
        cache.numerator[page].fill_(page + 20)
        cache.denominator[page].fill_(page + 30)
        cache.maximum[page].fill_(page + 40)
        cache.last_slot[page] = page * 16 + 3
    cache.request_journal_tag.copy_(torch.tensor([3, 19, 35, 51, 18, 34]))
    before = {n: getattr(cache, n).clone() for n in ("mean", "numerator", "denominator", "maximum")}
    # C must get old B, not the just-written A. Self-copy must be a no-op.
    layer.copy_kv_cache_side_storage(iter([_copy(0, 1), _copy(1, 2), _copy(3, 3)]), 4)
    for name, old in before.items():
        assert torch.equal(getattr(cache, name)[1], old[0])
        assert torch.equal(getattr(cache, name)[2], old[1])
        assert torch.equal(getattr(cache, name)[0], old[0])
    assert cache.last_slot[:4].tolist() == [3, 19, 35, 51]
    assert cache.request_journal_tag.tolist() == [3, -1, -1, 51, -1, -1]


def test_cow_copies_cpu_tail_snapshots_without_aliasing(api):
    cache = api.AscendStep5SparseSummaryCache(3, 16, torch.device("cpu"))
    saved = (torch.ones(256), torch.ones(256) * 2, torch.ones(1) * 3, torch.tensor(18))
    cache._speculative_tails[1] = {19: saved}
    cache.last_slot[1] = 18
    cache.copy([_copy(1, 2)])
    cloned = cache._speculative_tails[2][35]
    assert cloned[-1] == 34 and cache.last_slot[2] == 34
    cloned[0].fill_(7)
    assert torch.all(saved[0] == 1)


def test_scheduler_page_expansion_applies_to_reset_and_cow(api):
    context = {}
    layer = _layer(api, context, pages=6, spec=1)
    cache = _bind(context, layer)
    cache.blocks_per_scheduler_block = 2
    cache.mean.fill_(5)
    cache.request_journal_tag.copy_(torch.tensor([32, 48, 0, 16]))
    layer.zero_kv_cache_side_storage([1])
    assert not cache.mean[2:4].any() and torch.all(cache.mean[:2] == 5)
    assert cache.request_journal_tag.tolist() == [-1, -1, 0, 16]
    cache.numerator[0].fill_(3)
    cache.numerator[1].fill_(9)
    cache.last_slot[0:2] = torch.tensor([2, 19])
    layer.copy_kv_cache_side_storage([_copy(0, 1)], 3)
    assert torch.all(cache.mean[2:4] == 5)
    assert torch.all(cache.numerator[2] == 3) and torch.all(cache.numerator[3] == 9)
    assert cache.last_slot[2:4].tolist() == [34, 51]


def test_runtime_reset_clears_tags_and_preserves_tensor_addresses(api):
    context = {}
    layer = _layer(api, context, spec=2)
    cache = _bind(context, layer)
    names = (
        "mean",
        "numerator",
        "denominator",
        "maximum",
        "last_slot",
        "request_journal_values",
        "request_journal_tag",
    )
    pointers = {n: getattr(cache, n).data_ptr() for n in names}
    cache.mean.fill_(8)
    cache.last_slot.fill_(3)
    cache.request_journal_tag.fill_(19)
    cache._speculative_tails[1] = {19: ()}
    buffers = layer._dsa_scratch_workspace.tensor_buffers_by_engine[0]
    for t in buffers.values():
        t.fill_(1)
    count, scratch_count = layer.reset_kv_cache_side_storage_runtime_state()
    assert count == 1 and scratch_count == len(buffers)
    assert pointers == {n: getattr(cache, n).data_ptr() for n in names}
    assert not cache.mean.any() and torch.all(cache.last_slot == -1)
    assert torch.all(cache.request_journal_tag == -1) and not cache._speculative_tails
    assert all(not t.any() for t in buffers.values())


def _expected_mean(keys, logits):
    weights = logits.float().softmax(dim=0)
    return (weights * keys.float()).sum(0).bfloat16().to(torch.float8_e4m3fn).view(torch.uint8)


def test_cpu_reference_chunked_prefill_decode_and_partial_cow(api):
    torch.manual_seed(7)
    key = torch.randn(12, 1, 256).bfloat16()
    z = torch.randn(12, 1, 256).bfloat16()
    cache = api.AscendStep5SparseSummaryCache(3, 16, torch.device("cpu"))
    cache.update(key[:3], z[:3], [16, 17, 18])
    cache.copy([_copy(1, 2)])
    cache.update(key[3:8], z[3:8], list(range(35, 40)))
    assert torch.equal(cache.mean[2, 0], _expected_mean(key[:8, 0], z[:8, 0]))
    assert cache.last_slot[2] == -1 and cache.last_slot[1] == 18
    for i in range(8, 12):
        cache.update_decode(key[i : i + 1], z[i : i + 1], torch.tensor([32 + i]))
    assert torch.equal(cache.mean[2, 1], _expected_mean(key[8:12, 0], z[8:12, 0]))
    assert cache.last_slot[2] == 43


def test_cpu_speculative_reference_rejects_across_region_boundary(api):
    torch.manual_seed(11)
    key = torch.randn(15, 1, 256).bfloat16()
    z = torch.randn(15, 1, 256).bfloat16()
    cache = api.AscendStep5SparseSummaryCache(2, 16, torch.device("cpu"))
    cache.update(key[:7], z[:7], list(range(7)))
    cache.update_speculative(key[7:11], z[7:11], [7, 8, 9, 10], [0, 4], [[0, 1]], 3)
    # Accept position7; overwrite positions8..10 and append11 with new tokens.
    cache.update_speculative(key[11:15], z[11:15], [8, 9, 10, 11], [0, 4], [[0, 1]], 3)
    assert torch.equal(cache.mean[0, 0], _expected_mean(key[:8, 0], z[:8, 0]))
    assert torch.equal(cache.mean[0, 1], _expected_mean(key[11:15, 0], z[11:15, 0]))
    assert cache.last_slot[0] == 11


def test_cpu_reference_discards_old_future_snapshots_after_shrinking_window(api):
    cache = api.AscendStep5SparseSummaryCache(2, 16, torch.device("cpu"))
    prefix = torch.zeros(15, 1, 256)
    cache.update(prefix, prefix, list(range(15)))
    old = torch.tensor([1.0, 2.0, 99.0]).view(3, 1, 1).expand(3, 1, 256)
    cache.update_speculative(old, torch.zeros_like(old), [15, 16, 17], [0, 3], [[0, 1]], 2)
    for slot, value in ((16, 10.0), (17, 20.0)):
        key = torch.full((1, 1, 256), value)
        cache.update_speculative(key, torch.zeros_like(key), [slot], [0, 1], [[0, 1]], 2)
    # A stale GPU ring tag for17 would restore old16=2, producing11 here.
    # The independent CPU reference drops the previous window and yields15.
    assert torch.all(cache.mean[1, 0].view(torch.float8_e4m3fn).float() == 15)
    assert cache.last_slot[1] == 17


def test_dispatch_uses_local_port_and_preserves_three_stage_order(api):
    cache = api.AscendStep5SparseSummaryCache(3, 16, torch.device("cpu"), 2, 3)
    fake_npu_key = SimpleNamespace(device=SimpleNamespace(type="npu"))
    z, slots, metadata = object(), np.array([16, 17]), object()
    cache.update(fake_npu_key, z, slots, metadata)
    assert api.kernels.update_prefill.call_args.args[2] is metadata
    cache.update_decode(fake_npu_key, z, slots)
    assert api.kernels.update_decode.call_args.args[3] is cache._mean
    api.kernels.make_prefill_metadata_from_slots.return_value = metadata
    assert cache.prefill_metadata(slots, "npu") is metadata
    metadata_args = api.kernels.make_prefill_metadata_from_slots.call_args.args
    assert metadata_args[0] is slots and metadata_args[1:] == (16, "npu")
    events = []
    api.kernels.update_request_journal.side_effect = lambda *a, **kw: events.append(("journal", a, kw))
    api.kernels.update_prefill.side_effect = lambda *a, **kw: events.append(("prefill", a, kw))
    qsl, lens, state = object(), object(), object()
    cache.update_request_journal(fake_npu_key, z, slots, qsl, lens, state, metadata)
    assert [e[0] for e in events] == ["journal", "prefill", "journal"]
    assert events[0][2] == {"update": False} and events[2][2] == {"restore": False}
    assert events[0][1][5] is state and events[0][1][-1] == 3
    assert events[1][1][2] is metadata
    events.clear()
    cache.update_request_journal(fake_npu_key, z, slots, qsl, lens, state)
    assert len(events) == 1 and events[0][2] == {}


@pytest.mark.parametrize("slots", [[16, 17, -1], torch.tensor([16, 17, -1])])
def test_npu_update_without_metadata_uses_local_host_metadata_helper(api, slots):
    cache = api.AscendStep5SparseSummaryCache(3, 16, torch.device("cpu"))
    fake_npu_key = SimpleNamespace(device=SimpleNamespace(type="npu"))
    metadata = torch.tensor([[2, 0, 17, 1, 0, 1, -1, -1, -1, -1, -1, -1]])
    api.kernels.make_prefill_metadata_from_slots.return_value = metadata
    cache.update(fake_npu_key, object(), slots)
    assert api.kernels.update_prefill.call_args.args[2] is metadata
    uploaded = api.kernels.make_prefill_metadata_from_slots.call_args.args[0]
    assert isinstance(uploaded, np.ndarray) and uploaded.tolist() == [16, 17, -1]
    api.kernels.make_prefill_metadata_from_slots.return_value = None
    api.kernels.update_prefill.reset_mock()
    cache.update(fake_npu_key, object(), [-1, -1])
    api.kernels.update_prefill.assert_not_called()


def test_prefill_and_journal_use_shared_profiled_state_workspace(api):
    cache = api.AscendStep5SparseSummaryCache(3, 16, torch.device("cpu"), 2, 3)
    cache.prefill_state_scratch = torch.empty((8, 513))
    metadata = torch.empty((2, 12), dtype=torch.int64)
    fake_npu_key = SimpleNamespace(device=SimpleNamespace(type="npu"))
    cache.update(fake_npu_key, object(), object(), metadata)
    state = api.kernels.update_prefill.call_args.kwargs["state"]
    assert state.shape == (2, 513) and state.data_ptr() == cache.prefill_state_scratch.data_ptr()
    cache.update_request_journal(fake_npu_key, object(), object(), object(), object(), object(), metadata)
    journal_state = api.kernels.update_prefill.call_args.kwargs["state"]
    assert journal_state.shape == (2, 513) and journal_state.data_ptr() == state.data_ptr()
