# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""CPU lifecycle regressions for the staged Ascend Step5 summary sidecar.

Load the real cache and lifecycle implementations without importing NVIDIA
or NPU kernels. Only registration, block-copy transport and the reset kernel
are replaced; tensor storage, scratch sizing and COW state copying are real.
"""

import ast
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

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


class _RegistrationLayer(torch.nn.Module):
    """SWF registration constructor; its cache-layout methods must be overridden."""

    def __init__(self, *, prefix, target_impl, main_layer_name, static_forward_context):
        super().__init__()
        if prefix in static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        static_forward_context[prefix] = self
        self.layer_name = prefix
        self._target_impl = target_impl
        self.main_layer_name = main_layer_name
        self.kv_cache = torch.tensor([])
        self._cache = None


def _reset_kernel(ids, mean, active_ids, reverse, numerator, denominator, maximum, valid, **kwargs):
    # CPU stand-in for csa_reset_blocks. Hook tests distinguish page zeroing
    # from release, and verify unrelated live pages are never touched.
    pages = ids[(ids >= 0) & (ids < kwargs["num_pages"])].unique()
    if kwargs["clear_mean"]:
        mean[pages] = 0
    regions_per_page = kwargs["summaries_per_page"]
    for page in pages.tolist():
        begin, end = page * regions_per_page, (page + 1) * regions_per_page
        slots = torch.where((active_ids >= begin) & (active_ids < end))[0]
        active_ids[slots] = -1
        reverse[begin:end] = -1
        numerator[slots] = 0
        denominator[slots] = 0
        maximum[slots] = float("-inf")
        valid[slots] = 0


@pytest.fixture
def api(monkeypatch):
    namespace = dict(
        __name__=__name__,
        torch=torch,
        math=math,
        dataclass=dataclass,
        field=field,
        Step5SparseSummaryCacheLayer=_RegistrationLayer,
        csa_reset_blocks=_reset_kernel,
        logger=Mock(),
        _is_graph_capturing=lambda: False,
    )
    _definitions(
        MODEL_DIR / "_step5_compat.py",
        {
            "Step5SparseSummaryCacheConfig",
            "Step5SparseSummaryCache",
            "Step5DSAScratchWorkspace",
            "step5_sparse_summary_cache_bytes_per_block",
        },
        namespace,
    )
    _definitions(
        MODEL_DIR / "summary_cache.py",
        {
            "AscendStep5SparseSummaryCache",
            "AscendStep5SparseSummaryCacheLayer",
        },
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
            "_csa_active_region_capacity",
            "csa_fixed_runtime_state_size_bytes",
            "_initialize_csa_summary_state",
            "bind_summary_cache",
            "_round_up",
        },
    )
    namespace["AscendStep5DSAImpl"]._dsa_virtual_engine = lambda self: 0
    transfer = ModuleType("vllm.distributed.kv_transfer")
    transfer.has_kv_transfer_group = lambda: False
    monkeypatch.setitem(sys.modules, transfer.__name__, transfer)
    utils = ModuleType("vllm.v1.worker.utils")

    def copy_payload(caches, num_blocks, copies):
        assert num_blocks == caches[0].shape[0]
        src = [copy.src_block_id for copy in copies]
        dst = [copy.dst_block_id for copy in copies]
        for tensor in caches:
            tensor[dst] = tensor[src].clone()

    utils.copy_kv_cache_blocks_inplace = copy_payload
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    block_utils = ModuleType("vllm.v1.core.kv_cache_utils")
    block_utils.KVCacheBlockCopy = SimpleNamespace
    monkeypatch.setitem(sys.modules, block_utils.__name__, block_utils)
    return SimpleNamespace(**namespace)


def _layer(api, context, index=0, max_seqs=2, max_tokens=16, pages=4):
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
    )
    layer = api.AscendStep5SparseSummaryCacheLayer(
        prefix=f"layers.{index}.summary",
        target_impl=impl,
        sparse_config=SimpleNamespace(proxy_dim=256),
        main_layer_name=f"layers.{index}.attn",
        static_forward_context=context,
    )
    layer._set_summary_cache_budget(16)
    # Owner KV may be virtually split into eight-token kernel blocks; side
    # storage must still use four scheduler pages of sixteen tokens.
    owner_k = torch.empty(pages * 2, 8, 1, 128, dtype=torch.bfloat16)
    context[layer.main_layer_name] = SimpleNamespace(kv_cache=(owner_k, torch.empty_like(owner_k)))
    return layer


def _bind(api, context, layer):
    layer.prepare_kv_cache_side_storage_memory_profile(context, torch.device("cpu"))
    layer.bind_kv_cache_side_storage(context)
    return layer._budget_only_cache


def _seed(cache, slot, region, value):
    cache._step5_csa_active_region_ids[slot] = region
    cache._step5_csa_active_slot_by_region[region] = slot
    for name in (
        "_step5_csa_numerator_cache",
        "_step5_csa_denominator_cache",
        "_step5_csa_max_cache",
        "_step5_csa_active_token_k",
        "_step5_csa_active_token_z",
        "_step5_csa_active_token_valid",
    ):
        getattr(cache, name)[slot].fill_(value)


def test_bind_keeps_scheduler_geometry_and_page_budget(api):
    context = {}
    layer = _layer(api, context)
    cache = _bind(api, context, layer)
    assert cache.num_pages == 4
    assert cache.page_size == 16
    assert cache.mean_cache.shape == (4, 2, 1, 256)
    assert layer.kv_cache.data_ptr() == cache.mean_cache.data_ptr()
    assert layer._cache is cache is layer._target_impl._summary_cache
    assert layer._set_summary_cache_budget(16) == 512 + 2 * 4
    persistent = sum(
        getattr(cache, name).numel() * getattr(cache, name).element_size()
        for name in (
            "_step5_csa_active_region_ids",
            "_step5_csa_allocation_success",
            "_step5_csa_numerator_cache",
            "_step5_csa_denominator_cache",
            "_step5_csa_max_cache",
            "_step5_csa_active_token_k",
            "_step5_csa_active_token_z",
            "_step5_csa_active_token_valid",
        )
    )
    assert layer._fixed_runtime_state_budget(16) == persistent
    with pytest.raises(ValueError, match="positive"):
        api.Step5SparseSummaryCacheConfig(0, 16, 8, 1, 256)


def test_profile_shares_maximum_scratch_before_bind_and_locks_addresses(api):
    context = {}
    first = _layer(api, context, index=0, max_seqs=1, max_tokens=16)
    second = _layer(api, context, index=1, max_seqs=3, max_tokens=192)
    first.prepare_kv_cache_side_storage_memory_profile(context, torch.device("cpu"))
    workspace = first._dsa_scratch_workspace
    assert workspace is second._dsa_scratch_workspace
    assert workspace.memory_profiled and workspace.allocations_locked
    assert first._target_impl._summary_cache is None
    ptrs = {name: tensor.data_ptr() for name, tensor in workspace.tensor_buffers_by_engine[0].items()}
    second.prepare_kv_cache_side_storage_memory_profile(context, torch.device("cpu"))
    first.bind_kv_cache_side_storage(context)
    second.bind_kv_cache_side_storage(context)
    assert ptrs == {name: tensor.data_ptr() for name, tensor in workspace.tensor_buffers_by_engine[0].items()}
    assert first._budget_only_cache.mean_cache.data_ptr() != second._budget_only_cache.mean_cache.data_ptr()
    with pytest.raises(RuntimeError, match="capacity was exceeded"):
        second._target_impl._get_dsa_tensor_buffer_at_least(
            "late", (1,), device=torch.device("cpu"), dtype=torch.float32
        )


def test_release_preserves_prefix_payload_and_zeroing_clears_reused_page(api):
    context = {}
    layer = _layer(api, context)
    cache = _bind(api, context, layer)
    cache.mean_cache[1].fill_(13)
    cache.mean_cache[2].fill_(27)
    _seed(cache, 0, 2, 1)
    _seed(cache, 1, 4, 2)
    layer.release_kv_cache_side_storage([1])
    assert torch.all(cache.mean_cache[1] == 13)
    assert cache._step5_csa_active_region_ids[0] == -1
    assert cache._step5_csa_active_region_ids[1] == 4
    layer.zero_kv_cache_side_storage(torch.tensor([1]))
    assert not cache.mean_cache[1].any()
    assert torch.all(cache.mean_cache[2] == 27)
    assert cache._step5_csa_active_region_ids[1] == 4


def test_cow_copies_partial_region_and_staging_from_authoritative_slots(api):
    context = {}
    layer = _layer(api, context)
    cache = _bind(api, context, layer)
    cache.mean_cache[1].fill_(42)
    _seed(cache, 0, 2, 3)
    _seed(cache, 1, 4, 9)
    # A stale hint must never discard a live source accumulator.
    cache._step5_csa_active_slot_by_region[2] = -1
    copies = iter([SimpleNamespace(src_block_id=1, dst_block_id=2)])
    layer.copy_kv_cache_side_storage(copies, 4)
    assert torch.equal(cache.mean_cache[1], cache.mean_cache[2])
    src = torch.where(cache._step5_csa_active_region_ids == 2)[0]
    dst = torch.where(cache._step5_csa_active_region_ids == 4)[0]
    assert src.numel() == dst.numel() == 1
    for name in (
        "_step5_csa_numerator_cache",
        "_step5_csa_denominator_cache",
        "_step5_csa_max_cache",
        "_step5_csa_active_token_k",
        "_step5_csa_active_token_z",
        "_step5_csa_active_token_valid",
    ):
        assert torch.equal(getattr(cache, name)[src], getattr(cache, name)[dst])
    cache._step5_csa_numerator_cache[dst].fill_(5)
    assert torch.all(cache._step5_csa_numerator_cache[src] == 3)


def test_cow_overlap_snapshots_sources_and_capacity_failure_is_atomic(api):
    context = {}
    layer = _layer(api, context)
    cache = _bind(api, context, layer)
    _seed(cache, 0, 2, 3)
    _seed(cache, 1, 4, 7)
    layer._copy_csa_runtime_state(
        [
            SimpleNamespace(src_block_id=1, dst_block_id=2),
            SimpleNamespace(src_block_id=2, dst_block_id=1),
        ]
    )
    for region, value in ((2, 7), (4, 3)):
        slot = torch.where(cache._step5_csa_active_region_ids == region)[0]
        assert slot.numel() == 1
        assert torch.all(cache._step5_csa_numerator_cache[slot] == value)
    # More KV pages than active slots: all slots have distinct live regions
    # and the new destination is empty. Failure must leave both payload and
    # live state unchanged.
    context = {}
    layer = _layer(api, context, pages=16)
    cache = _bind(api, context, layer)
    active_ids = cache._step5_csa_active_region_ids
    active_ids.copy_(torch.arange(active_ids.numel(), dtype=active_ids.dtype))
    cache._step5_csa_active_slot_by_region.fill_(-1)
    cache.mean_cache[1].fill_(9)
    before = active_ids.clone()
    payload = cache.mean_cache.clone()
    with pytest.raises(RuntimeError, match="capacity is exhausted"):
        layer.copy_kv_cache_side_storage([SimpleNamespace(src_block_id=1, dst_block_id=15)], 16)
    assert torch.equal(cache._step5_csa_active_region_ids, before)
    assert torch.equal(cache.mean_cache, payload)


def test_capture_reset_restores_sentinels_and_preserves_storage_addresses(api):
    context = {}
    layer = _layer(api, context)
    cache = _bind(api, context, layer)
    _seed(cache, 0, 2, 3)
    cache.mean_cache.fill_(8)
    pointer = cache.mean_cache.data_ptr()
    buffers = layer._dsa_scratch_workspace.tensor_buffers_by_engine[0]
    for tensor in buffers.values():
        tensor.fill_(1)
    count, scratch_count = layer.reset_kv_cache_side_storage_runtime_state()
    assert count == 1 and scratch_count == len(buffers)
    assert cache.mean_cache.data_ptr() == pointer
    assert not cache.mean_cache.any()
    assert torch.all(cache._step5_csa_active_region_ids == -1)
    assert torch.all(cache._step5_csa_active_slot_by_region == -1)
    assert torch.all(cache._step5_csa_max_cache == float("-inf"))
    assert torch.all(buffers["csa_prefill_scratch_region_ids"] == -1)
