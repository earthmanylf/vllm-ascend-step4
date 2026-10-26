# SPDX-License-Identifier: Apache-2.0
"""CPU source-isolated Step5 Mooncake checks; no device/runtime imports.

Compile the production methods unchanged. Only upstream spec/config containers
and the network engine are doubles, never the address or block mapping logic.
"""

import ast
import ctypes
import math
import os
import queue
import threading
from collections import OrderedDict, defaultdict
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Literal, get_args

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[4]
MOONCAKE = ROOT / "vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake"


class Spec:
    def __init__(self, **values):
        self.__dict__.update(
            dict(block_size=16, num_kv_heads=2, head_size=4, head_size_v=4, dtype=torch.bfloat16, tokens_per_state=1)
        )
        self.__dict__.update(values)

    @property
    def num_heads(self):
        return self.num_kv_heads

    @property
    def num_states(self):
        return self.block_size // self.tokens_per_state

    @property
    def state_content_size_bytes(self):
        return (self.head_size + self.head_size_v) * torch.empty((), dtype=self.dtype).element_size()


class FullAttentionSpec(Spec):
    pass


class SlidingWindowSpec(Spec):
    pass


class CircularBufferSpec(Spec):
    def __init__(self, **values):
        super().__init__(**dict(num_kv_heads=1, head_size=32, head_size_v=0) | values)


class MLAAttentionSpec(FullAttentionSpec):
    pass


class SlidingWindowMLASpec(SlidingWindowSpec):
    pass


class MambaSpec(Spec):
    pass


class AscendSFAIndexerCacheSpec(FullAttentionSpec):
    pass


def load_definitions(filename, names, namespace):
    tree = ast.parse((MOONCAKE / filename).read_text())
    nodes = [node for node in tree.body if getattr(node, "name", None) in names]
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[])),
            str(MOONCAKE / filename),
            "exec",
        ),
        namespace,
    )


@pytest.fixture
def dtype_registry():
    source_root = Path(os.environ.get("VLLM_SOURCE_ROOT", ROOT.parent / "vllm-private"))
    source = source_root / "vllm/utils/torch_utils.py"
    tree = ast.parse(source.read_text())
    nodes = [
        node
        for node in tree.body
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "STR_DTYPE_TO_TORCH_DTYPE" for target in node.targets)
        )
        or (isinstance(node, ast.FunctionDef) and node.name == "get_dtype_size")
    ]
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    return namespace


@pytest.fixture
def production(dtype_registry):
    namespace = dict(
        globals(),
        threading=threading,
        Any=object,
        KVCacheSpec=Spec,
        is_circular_kv_cache_spec=lambda spec: isinstance(spec, CircularBufferSpec),
        logger=NS(info_once=lambda *args: None, warning_once=lambda *args: None),
    )
    namespace.update(dtype_registry)
    load_definitions("base_worker.py", {"_is_step5_summary_spec", "_validate_step5_components"}, namespace)
    load_definitions("utils.py", {"group_concurrent_contiguous"}, namespace)
    load_definitions("pull_worker.py", {"MooncakePullRecvingThread"}, namespace)
    return namespace["MooncakePullRecvingThread"]


def metadata(shapes, *, groups=None, dtypes=None, strides=None, lengths=None):
    return NS(
        layer_names=[f"layer.{i}" for i in range(len(shapes))],
        group_indices=groups or list(range(len(shapes))),
        layer_block_sizes=[16] * len(shapes),
        block_shapes=shapes,
        block_lens=lengths
        if lengths is not None
        else [
            [math.prod(shape) * getattr(torch, dtype.removeprefix("torch.")).itemsize for shape, dtype in zip(s, ds)]
            for s, ds in zip(shapes, dtypes or [["torch.bfloat16"] * len(s) for s in shapes])
        ],
        block_strides=strides
        if strides is not None
        else [
            [math.prod(shape) * getattr(torch, dtype.removeprefix("torch.")).itemsize for shape, dtype in zip(s, ds)]
            for s, ds in zip(shapes, dtypes or [["torch.bfloat16"] * len(s) for s in shapes])
        ],
        block_size_scales=[[1] * len(s) for s in shapes],
        block_dtypes=dtypes or [["torch.bfloat16"] * len(s) for s in shapes],
        is_step5=True,
        block_size=16,
        num_blocks=8,
    )


def thread_from(production, specs, local, *, rank=1, tp=4):
    thread = production.__new__(production)
    thread.local_metadata = local
    for name in (
        "layer_names",
        "group_indices",
        "layer_block_sizes",
        "block_shapes",
        "block_lens",
        "block_strides",
        "block_size_scales",
        "block_dtypes",
        "is_step5",
    ):
        setattr(thread, name, getattr(local, name))
    thread.kv_cache_specs = specs
    thread.spec_indices = list(range(len(specs)))
    thread.tp_rank, thread.tp_size = rank, tp
    thread.pp_rank, thread.pp_size = 0, 1
    thread.pcp_rank = thread.dcp_rank = 0
    thread.pcp_size = thread.dcp_size = 1
    thread.num_speculative_tokens = 0
    thread.kv_caches_base_addr = [[1000] * len(s) for s in local.block_shapes]
    return thread


def grouped(remote, tp=4):
    remote.metadata_by_pcp_rank = {
        0: NS(
            metadata_by_tp_rank={
                rank: NS(kv_caches_base_addr=[[5000] * len(s) for s in remote.block_shapes]) for rank in range(tp)
            }
        )
    }
    return NS(
        is_step5=True, pp_size=1, pcp_size=1, dcp_size=1, tp_size=tp, use_kv_pp=False, metadata_by_pp_rank={0: remote}
    )


def test_step5_layout_preserves_nhd():
    namespace = dict(
        KVConnectorBase_V1=type("Base", (), {}),
        SupportsHMA=type("SupportsHMA", (), {}),
        TYPE_CHECKING=False,
        vllm_version_is=lambda version: False,
        logger=NS(info_once=lambda *args: None, warning_once=lambda *args: None),
    )
    load_definitions("connector.py", {"MooncakeBaseConnector"}, namespace)
    config = NS(model_config=NS(use_mla=False, hf_text_config=NS(model_type="step5")))
    assert namespace["MooncakeBaseConnector"].get_required_kvcache_layout(config) in ("LBNHC", "NHD")


def test_step5_main_matching_rank_summary_and_ring_replicas(production):
    shapes = [[(16, 2, 4)] * 2, [(1, 2, 256)], [(1, 16, 32)]]
    specs = [
        FullAttentionSpec(),
        FullAttentionSpec(num_kv_heads=1, head_size=256, head_size_v=0, dtype=torch.uint8, tokens_per_state=8),
        CircularBufferSpec(),
    ]
    local = metadata(shapes, dtypes=[["torch.bfloat16"] * 2, ["torch.uint8"], ["torch.bfloat16"]])
    thread = thread_from(production, specs, local)
    groups, _ = thread._build_remote_transfer_layout(grouped(metadata(shapes, dtypes=local.block_dtypes)))
    assert groups[0][(0, 0)] == [[1]]
    assert groups[0][(1, 1)] == [[0, 1, 2, 3]]
    assert groups[0][(2, 2)] == [[0, 1, 2, 3]]


@pytest.mark.parametrize(
    "field,value", [("tp_size", 8), ("pcp_size", 2), ("dcp_size", 2), ("pp_size", 2), ("is_step5", False)]
)
def test_step5_rejects_topology_before_transfer(production, field, value):
    thread = thread_from(production, [FullAttentionSpec()], metadata([[(16, 2, 4)] * 2]))
    remote = grouped(metadata([[(16, 2, 4)] * 2]))
    setattr(remote, field, value)
    with pytest.raises(ValueError, match="Step5"):
        thread._build_remote_transfer_layout(remote)


@pytest.mark.parametrize(
    "field,value",
    [
        ("block_shapes", [[(2, 16, 4)]]),
        ("block_lens", [[64]]),
        ("block_dtypes", [["torch.float16"]]),
        ("group_indices", [1]),
        ("layer_block_sizes", [32]),
        ("block_size_scales", [[2]]),
    ],
)
def test_step5_rejects_layer_mismatch(production, field, value):
    thread = thread_from(production, [FullAttentionSpec()], metadata([[(16, 2, 4)] * 2]))
    remote = metadata([[(16, 2, 4)] * 2])
    setattr(remote, field, value)
    with pytest.raises(ValueError, match="Step5"):
        thread._build_remote_transfer_layout(grouped(remote))


def test_mapping_cache_includes_independent_group_identity(production):
    thread = thread_from(production, [FullAttentionSpec()] * 2, metadata([[(16, 2, 4)]] * 2), rank=0, tp=1)
    thread.is_step5 = False
    thread.spec_indices = [0, 0]  # Equivalent specs must not collapse block tables.
    remote = metadata([[(16, 2, 4)]] * 2)
    request = NS(
        local_block_ids=([1, 2], [3, 4]),
        local_full_block_ids=([1, 2], [3, 4]),
        remote_block_ids=([5, 6], [7, 8]),
        local_num_prompt_tokens=32,
        remote_num_prompt_tokens=32,
        num_computed_tokens=0,
    )
    buckets, _ = thread._build_transfer_block_buckets(
        remote, [(0, 0), (1, 1)], {(0, 0): [[0]], (1, 1): [[0]]}, 1, 1, {"req": request}, {}
    )
    assert buckets[(0, 0)][0][(0, 0)] == [("req", [1, 2], [5, 6])]
    assert buckets[(0, 0)][0][(1, 1)] == [("req", [3, 4], [7, 8])]


@pytest.mark.parametrize("padded", [False, True])
def test_step5_nhd_tuple_whole_blocks_cpu_roundtrip(production, padded):
    shape = (16, 2, 4)
    payload = 256
    local_stride, remote_stride = (320, 384) if padded else (payload, payload)
    local_storage = [torch.full((8 * local_stride + 16,), 0xA5, dtype=torch.uint8) for _ in range(2)]
    remote_storage = [torch.arange(8 * remote_stride + 16).to(torch.uint8) for _ in range(2)]
    local = metadata([[shape] * 2], lengths=[[payload] * 2], strides=[[local_stride] * 2])
    remote = metadata([[shape] * 2], lengths=[[payload] * 2], strides=[[remote_stride] * 2])
    thread = thread_from(production, [FullAttentionSpec()], local)
    thread.kv_caches_base_addr = [[storage.data_ptr() + 8 for storage in local_storage]]
    grouped(remote)
    remote.metadata_by_pcp_rank[0].metadata_by_tp_rank[1].kv_caches_base_addr = [
        [storage.data_ptr() + 8 for storage in remote_storage]
    ]
    src, dst, lengths = [], [], []
    # Despite equal TP, wrong-rank main transfer must not use whole-block copy.
    with pytest.raises(ValueError, match="Step5"):
        thread._append_spec_transfer_addresses(
            0, 0, 0, 1, 4, 1, {(0, 0): [("req", [1, 2], [3, 4])]}, remote, [], [], []
        )
    thread._append_spec_transfer_addresses(
        0, 0, 1, 1, 4, 1, {(0, 0): [("req", [1, 2], [3, 4])]}, remote, src, dst, lengths
    )
    for target, source, length in zip(src, dst, lengths):
        ctypes.memmove(target, source, length)
    for local_raw, remote_raw in zip(local_storage, remote_storage):
        expected = torch.full_like(local_raw, 0xA5)
        for local_id, remote_id in [(1, 3), (2, 4)]:
            start = 8 + local_id * local_stride
            remote_start = 8 + remote_id * remote_stride
            expected[start : start + payload] = remote_raw[remote_start : remote_start + payload]
        assert torch.equal(local_raw, expected)


def test_step5_rejects_hnd_on_both_peers(production):
    thread = thread_from(production, [FullAttentionSpec()], metadata([[(2, 16, 4)]]))
    with pytest.raises(ValueError, match="Step5"):
        thread._build_remote_transfer_layout(grouped(metadata([[(2, 16, 4)]])))


def test_step5_rejects_truncated_block_lists(production):
    thread = thread_from(production, [FullAttentionSpec()], metadata([[(16, 2, 4)] * 2]))
    request = NS(
        local_block_ids=([1, 2],),
        local_full_block_ids=([1, 2],),
        remote_block_ids=([3],),
        local_num_prompt_tokens=32,
        remote_num_prompt_tokens=32,
        num_computed_tokens=0,
    )
    with pytest.raises(ValueError, match="Step5"):
        thread._build_transfer_block_buckets(
            metadata([[(16, 2, 4)] * 2]), [(0, 0)], {(0, 0): [[1]]}, 1, 1, {"req": request}, {}
        )


def test_step5_equal_spec_independent_groups_address_generation(production):
    shapes = [[(16, 2, 4)] * 2] * 2
    thread = thread_from(production, [FullAttentionSpec()] * 2, metadata(shapes))
    thread.spec_indices = [0, 0]
    remote = metadata(shapes)
    grouped(remote)
    src, dst, lengths = [], [], []
    thread._append_spec_transfer_addresses(
        0, 0, 1, 1, 4, 1, {(0, 0): [("req", [1], [3])], (1, 1): [("req", [2], [4])]}, remote, src, dst, lengths
    )
    assert src == [1256, 1256, 1512, 1512]
    assert dst == [5768, 5768, 6024, 6024]


@pytest.fixture
def registration(dtype_registry):
    namespace = dict(
        globals(),
        dataclass=dataclass,
        KVConnectorHandshakeMetadata=type("Handshake", (), {}),
        KVConnectorMetadata=type("Metadata", (), {}),
        math=math,
        is_circular_kv_cache_spec=lambda spec: isinstance(spec, CircularBufferSpec),
        UniformTypeKVCacheSpecs=type("UniformTypeKVCacheSpecs", (), {}),
        get_kv_cache_tensor_layers=lambda descriptor: descriptor.layers,
        global_te=NS(register_buffer=lambda *args: None),
        logger=NS(info=lambda *args: None, debug=lambda *args: None),
        Any=object,
    )
    load_definitions(
        "metadata.py",
        {
            "MooncakeTransferMetadata",
            "MooncakeTPTransferMetadata",
            "MooncakePCPTransferMetadata",
            "MooncakePPTransferMetadata",
            "MooncakeTransferMetadataGroups",
        },
        namespace,
    )
    utils_source = ROOT / "vllm_ascend/distributed/kv_transfer/utils/utils.py"
    tree = ast.parse(utils_source.read_text())
    nodes = [
        n
        for n in tree.body
        if getattr(n, "name", None) in {"RegisterRegions", "tensor_storage_key", "validate_register_region_count"}
    ]
    namespace["MAX_HCCL_REGISTER_REGIONS"] = 256
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(utils_source), "exec"), namespace)
    namespace["KV_CACHE_BUFFER_ALIGNMENT"] = 2 * 1024 * 1024
    load_definitions(
        "utils.py",
        {
            "as_kv_cache_tensors",
            "_get_storage_nbytes",
            "_get_tensor_span_nbytes",
            "collect_configured_register_regions",
        },
        namespace,
    )
    namespace.update(dtype_registry)
    load_definitions(
        "base_worker.py",
        {
            "MooncakeBaseConnectorWorker",
            "_is_component_major_mla_binding",
            "_is_step5_summary_spec",
            "_validate_step5_components",
        },
        namespace,
    )
    load_definitions("pull_scheduler.py", {"MooncakeSchedulerSendingThread"}, namespace)
    return namespace


def register_cpu(registration, caches, specs):
    worker = registration["MooncakeBaseConnectorWorker"].__new__(registration["MooncakeBaseConnectorWorker"])
    names = list(caches)
    worker.vllm_config = NS(model_config=NS(hf_text_config=NS(model_type="step5")))
    worker.kv_cache_config = NS(
        num_blocks=8,
        kv_cache_groups=[NS(layer_names=[name], kv_cache_spec=spec) for name, spec in zip(names, specs)],
        kv_cache_tensors=[NS(layers=[name], size=0) for name in names],
    )
    worker.ascend_config = NS(kvpp_config=NS(size=1))
    worker.pcp_rank = worker.tp_rank = 0
    worker.tp_size = 1
    worker.block_size = 16
    worker.engine_id, worker.te_rpc_port = "cpu-fixture", 9000
    worker.side_channel_host, worker.handshake_port = "localhost", 5000
    worker.register_kv_caches(caches)
    return worker.transfer_metadata


def test_step5_registration_requires_dense_inner_blocks(registration):
    cache = torch.zeros((8, 16, 2, 8), dtype=torch.bfloat16)[..., ::2]
    with pytest.raises(ValueError, match="Step5"):
        register_cpu(registration, {"main": (cache, cache.clone())}, [FullAttentionSpec()])


def test_step5_metadata_requires_component_dtypes(registration):
    fields = vars(metadata([[(16, 2, 4)] * 2])).copy()
    fields.update(engine_id="p", te_rpc_port=9000, kv_caches_base_addr=[[1000, 2000]])
    fields["block_dtypes"] = None
    with pytest.raises(ValueError, match="Step5"):
        registration["MooncakeTransferMetadata"](**fields)


def test_step5_registration_and_scheduler_preserve_marker_dtypes(registration):
    caches = {
        "main": tuple(torch.zeros((8, 16, 2, 4), dtype=torch.bfloat16) for _ in range(2)),
        "summary": torch.zeros((8, 1, 2, 256), dtype=torch.uint8),
        "ring": torch.zeros((8, 1, 16, 32), dtype=torch.bfloat16),
    }
    specs = [
        FullAttentionSpec(),
        FullAttentionSpec(num_kv_heads=1, head_size=256, head_size_v=0, dtype=torch.uint8, tokens_per_state=8),
        CircularBufferSpec(),
    ]
    metadata_p = register_cpu(registration, caches, specs)
    assert metadata_p.is_step5
    assert metadata_p.group_indices == [0, 1, 2]
    assert metadata_p.block_dtypes == [["torch.bfloat16"] * 2, ["torch.uint8"], ["torch.bfloat16"]]
    scheduler = registration["MooncakeSchedulerSendingThread"].__new__(registration["MooncakeSchedulerSendingThread"])
    scheduler.pp_size = scheduler.pcp_size = scheduler.tp_size = 1
    scheduler.engine_id = metadata_p.engine_id
    scheduler.use_kv_pp = False
    merged = scheduler._merge_metadata_by_pp_rank({0: metadata_p})[0]
    assert merged.is_step5
    assert dict(zip(merged.layer_names, merged.block_dtypes)) == dict(
        zip(metadata_p.layer_names, metadata_p.block_dtypes)
    )


@pytest.mark.parametrize("prompt_tokens", [*range(120, 130)])
def test_step5_all_groups_registered_cpu_roundtrip(production, registration, prompt_tokens):
    specs = [
        FullAttentionSpec(block_size=128),
        FullAttentionSpec(
            block_size=128, num_kv_heads=1, head_size=256, head_size_v=0, dtype=torch.uint8, tokens_per_state=8
        ),
        CircularBufferSpec(block_size=128),
    ]
    shapes = [(128, 2, 4), (1, 16, 256), (1, 128, 32)]
    local_caches, remote_caches = {}, {}
    local_raws, remote_raws = [], []
    for index, (shape, spec) in enumerate(zip(shapes, specs)):
        views_d, views_p = [], []
        for _ in range(2 if index == 0 else 1):
            payload_elements = int(np.prod(shape))
            stride_d, stride_p = payload_elements + 16, payload_elements + 32
            raw_d = torch.full((8 * stride_d + 16,), 5, dtype=spec.dtype)
            raw_p = torch.arange(8 * stride_p + 16).to(spec.dtype)
            inner_strides = (shape[1] * shape[2], shape[2], 1)
            views_d.append(torch.as_strided(raw_d, (8, *shape), (stride_d, *inner_strides), 8))
            views_p.append(torch.as_strided(raw_p, (8, *shape), (stride_p, *inner_strides), 8))
            local_raws.append(raw_d)
            remote_raws.append(raw_p)
        local_caches[f"group.{index}"] = tuple(views_d)
        remote_caches[f"group.{index}"] = tuple(views_p)
    local = register_cpu(registration, local_caches, specs)
    remote = register_cpu(registration, remote_caches, specs)
    scheduler = registration["MooncakeSchedulerSendingThread"].__new__(registration["MooncakeSchedulerSendingThread"])
    scheduler.engine_id = remote.engine_id
    scheduler.pp_size = scheduler.pcp_size = scheduler.tp_size = 1
    scheduler.use_kv_pp = False
    pp_remote = scheduler._merge_metadata_by_pp_rank({0: remote})[0]
    thread = thread_from(production, specs, local, rank=0, tp=1)
    thread.kv_caches_base_addr = local.kv_caches_base_addr
    topology = NS(
        is_step5=True, pp_size=1, pcp_size=1, dcp_size=1, tp_size=1, use_kv_pp=False, metadata_by_pp_rank={0: pp_remote}
    )
    groups, pairs = thread._build_remote_transfer_layout(topology)
    blocks = 1 if prompt_tokens <= 128 else 2
    local_ids = ([1, 2][:blocks], [3, 4][:blocks], [5])
    remote_ids = ([4, 5][:blocks], [1, 2][:blocks], [6])
    request = NS(
        local_block_ids=local_ids,
        local_full_block_ids=local_ids,
        remote_block_ids=remote_ids,
        local_num_prompt_tokens=prompt_tokens,
        remote_num_prompt_tokens=prompt_tokens,
        num_computed_tokens=0,
    )
    buckets, _ = thread._build_transfer_block_buckets(pp_remote, pairs[0], groups[0], 1, 1, {"req": request}, {})
    src, dst, lengths = [], [], []
    for spec_index, entries in buckets[(0, 0)].items():
        thread._append_spec_transfer_addresses(spec_index, 0, 0, 1, 1, 1, entries, pp_remote, src, dst, lengths)
    for target, source, length in zip(src, dst, lengths):
        ctypes.memmove(target, source, length)
    raw_index = 0
    for group_index, name in enumerate(local_caches):
        for view_d, view_p in zip(local_caches[name], remote_caches[name]):
            expected = torch.full_like(local_raws[raw_index], 5)
            for local_id, remote_id in zip(local_ids[group_index], remote_ids[group_index]):
                start = view_d.storage_offset() + local_id * view_d.stride(0)
                source = view_p.storage_offset() + remote_id * view_p.stride(0)
                count = view_d[0].numel()
                expected[start : start + count] = remote_raws[raw_index][source : source + count]
            assert torch.equal(local_raws[raw_index], expected)
            raw_index += 1


def test_step5_rejects_different_group_membership(production):
    local = metadata([[(16, 2, 4)] * 2])
    thread = thread_from(production, [FullAttentionSpec()], local)
    remote = metadata([[(16, 2, 4)] * 2] * 2, groups=[0, 0])
    with pytest.raises(ValueError, match="Step5"):
        thread._build_remote_transfer_layout(grouped(remote))


def test_step5_rejects_same_peer_dtype_but_wrong_local_spec(production):
    local = metadata([[(16, 2, 4)] * 2], dtypes=[["torch.float16"] * 2])
    remote = metadata([[(16, 2, 4)] * 2], dtypes=[["torch.float16"] * 2])
    thread = thread_from(production, [FullAttentionSpec()], local)
    with pytest.raises(ValueError, match="Step5"):
        thread._build_remote_transfer_layout(grouped(remote))


@pytest.mark.parametrize("stage", ["layout", "mapping", "address"])
@pytest.mark.parametrize("defect", ["unequal_scales", "one_byte", "summary", "ring"])
def test_step5_rejects_identical_invalid_descriptors(production, registration, stage, defect):
    spec = FullAttentionSpec()
    caches = {"main": tuple(torch.zeros((8, 16, 2, 4), dtype=spec.dtype) for _ in range(2))}
    if defect in ("summary", "ring"):
        spec = (
            FullAttentionSpec(num_kv_heads=1, head_size=256, head_size_v=0, dtype=torch.uint8, tokens_per_state=8)
            if defect == "summary"
            else CircularBufferSpec()
        )
        shape = (1, spec.num_states, spec.head_size)
        caches = {"main": torch.zeros((8, *shape), dtype=spec.dtype)}
    local = NS(**vars(register_cpu(registration, caches, [spec])))
    remote = NS(**vars(register_cpu(registration, caches, [spec])))
    for descriptor in (local, remote):
        if defect == "unequal_scales":
            descriptor.block_size_scales = [[1, 2]]
        elif defect == "one_byte":
            descriptor.block_lens = [[1, 1]]
        else:
            descriptor.block_shapes = [[(1, 999, spec.head_size)]]
            payload = 999 * spec.head_size * spec.dtype.itemsize
            descriptor.block_lens = descriptor.block_strides = [[payload]]
    grouped(remote, tp=1)
    thread = thread_from(production, [spec], local, rank=0, tp=1)
    request = NS(
        local_block_ids=([1],),
        local_full_block_ids=([1],),
        remote_block_ids=([3],),
        local_num_prompt_tokens=16,
        remote_num_prompt_tokens=16,
        num_computed_tokens=0,
    )
    src, dst, lengths = [], [], []
    with pytest.raises(ValueError, match="Step5"):
        if stage == "layout":
            thread._build_remote_transfer_layout(grouped(remote, tp=1))
        elif stage == "mapping":
            thread._build_transfer_block_buckets(remote, [(0, 0)], {(0, 0): [[0]]}, 1, 1, {"req": request}, {})
        else:
            thread._append_spec_transfer_addresses(
                0, 0, 0, 1, 1, 1, {(0, 0): [("req", [1], [3])]}, remote, src, dst, lengths
            )
    assert src == dst == lengths == []


@pytest.mark.parametrize("defect", ["unequal_scales", "summary", "ring"])
def test_step5_registration_rejects_invalid_geometry(registration, defect):
    if defect == "unequal_scales":
        spec = FullAttentionSpec()
        caches = tuple(torch.zeros((blocks, 16, 2, 4), dtype=spec.dtype) for blocks in (8, 16))
    else:
        spec = (
            FullAttentionSpec(num_kv_heads=1, head_size=256, head_size_v=0, dtype=torch.uint8, tokens_per_state=8)
            if defect == "summary"
            else CircularBufferSpec()
        )
        caches = torch.zeros((8, 1, 999, spec.head_size), dtype=spec.dtype)
    with pytest.raises(ValueError, match="Step5"):
        register_cpu(registration, {"main": caches}, [spec])


@pytest.mark.parametrize(
    "field,value",
    [
        ("block_size_scales", [[0, 0]]),
        ("block_size_scales", [[-1, -1]]),
        ("block_size_scales", [[3, 3]]),
        ("block_strides", [[256]]),
        ("block_dtypes", [["torch.unknown", "torch.unknown"]]),
        ("block_shapes", [[(16, 2, 4)]]),
    ],
)
def test_step5_rejects_invalid_component_metadata(production, field, value):
    local, remote = metadata([[(16, 2, 4)] * 2]), metadata([[(16, 2, 4)] * 2])
    setattr(local, field, value)
    setattr(remote, field, value)
    thread = thread_from(production, [FullAttentionSpec()], local)
    with pytest.raises(ValueError, match="Step5"):
        thread._build_remote_transfer_layout(grouped(remote))


@pytest.mark.parametrize("sidecar", ["summary", "ring"])
@pytest.mark.parametrize("defect", ["heads", "content", "components"])
def test_step5_sidecar_spec_axes_and_count(production, sidecar, defect):
    spec = (
        FullAttentionSpec(num_kv_heads=1, head_size=256, head_size_v=0, dtype=torch.uint8, tokens_per_state=8)
        if sidecar == "summary"
        else CircularBufferSpec()
    )
    shape = (1, spec.num_states, spec.head_size)
    if defect == "heads":
        shape = (2, *shape[1:])
    elif defect == "content":
        shape = (*shape[:2], spec.head_size + 1)
    shapes = [[shape] * (2 if defect == "components" else 1)]
    local, remote = [metadata(shapes, dtypes=[[str(spec.dtype)] * len(shapes[0])]) for _ in range(2)]
    thread = thread_from(production, [spec], local)
    with pytest.raises(ValueError, match="Step5"):
        thread._build_remote_transfer_layout(grouped(remote))


def test_step5_equal_physical_scales_registration_mapping_addresses(production, registration):
    caches = {"main": tuple(torch.zeros((16, 8, 2, 4), dtype=torch.bfloat16) for _ in range(2))}
    local, remote = [register_cpu(registration, caches, [FullAttentionSpec()]) for _ in range(2)]
    assert local.block_size_scales == [[2, 2]]
    thread = thread_from(production, [FullAttentionSpec()], local, rank=0, tp=1)
    thread.kv_caches_base_addr = local.kv_caches_base_addr
    remote = NS(**vars(remote))
    groups, pairs = thread._build_remote_transfer_layout(grouped(remote, tp=1))
    request = NS(
        local_block_ids=([1],),
        local_full_block_ids=([1],),
        remote_block_ids=([3],),
        local_num_prompt_tokens=16,
        remote_num_prompt_tokens=16,
        num_computed_tokens=0,
    )
    buckets, _ = thread._build_transfer_block_buckets(remote, pairs[0], groups[0], 1, 1, {"req": request}, {})
    assert buckets[(0, 0)][0][(0, 0)] == [("req", [2, 3], [6, 7])]
    src, dst, lengths = [], [], []
    thread._append_spec_transfer_addresses(0, 0, 0, 1, 1, 1, buckets[(0, 0)][0], remote, src, dst, lengths)
    assert src == [base + 2 * 128 for base in local.kv_caches_base_addr[0]]
    assert dst == [5000 + 6 * 128] * 2
    assert lengths == [256, 256]


def test_other_model_registration_keeps_backward_defaults(registration):
    worker_type = registration["MooncakeBaseConnectorWorker"]
    worker = worker_type.__new__(worker_type)
    worker.vllm_config = NS(model_config=NS(hf_text_config=NS(model_type="other")))
    worker.kv_cache_config = NS(
        num_blocks=8,
        kv_cache_groups=[NS(layer_names=["main"], kv_cache_spec=FullAttentionSpec())],
        kv_cache_tensors=[NS(layers=["main"], size=0)],
    )
    worker.ascend_config = NS(kvpp_config=NS(size=1))
    worker.pcp_rank = worker.tp_rank = 0
    worker.tp_size, worker.block_size = 1, 16
    worker.engine_id, worker.te_rpc_port = "cpu", 9000
    worker.side_channel_host, worker.handshake_port = "localhost", 5000
    caches = tuple(torch.zeros((blocks, 16, 2, 4), dtype=torch.bfloat16) for blocks in (8, 16))
    worker.register_kv_caches({"main": caches})
    assert worker.transfer_metadata.block_size_scales == [[1, 2]]
    assert not worker.transfer_metadata.is_step5
    assert worker.transfer_metadata.block_dtypes is None


def test_step5_unknown_spec_dtype_is_rejected(production):
    local, remote = [metadata([[(16, 2, 4)] * 2], dtypes=[["torch.float64"] * 2]) for _ in range(2)]
    thread = thread_from(production, [FullAttentionSpec(dtype=torch.float64)], local)
    with pytest.raises(ValueError, match="Step5 unknown component dtype"):
        thread._build_remote_transfer_layout(grouped(remote))


def test_cpu_existing_pull_regressions(production, registration):
    namespace = dict(
        registration,
        MooncakePullRecvingThread=production,
        replace=replace,
        SimpleNamespace=NS,
        KVCacheSpec=Spec,
        queue=queue,
    )
    load_definitions("metadata.py", {"ReqMeta"}, namespace)
    helpers = ROOT / "tests/ut/kv_offload/mooncake_v2/helpers.py"
    tests = ROOT / "tests/ut/kv_offload/mooncake_v2/test_pull_worker.py"
    load_definitions(helpers, {"make_full_spec", "make_transfer_metadata", "make_pp_metadata"}, namespace)
    load_definitions(
        tests,
        {
            "make_thread",
            "make_req_meta",
            "test_whole_block_mla_address_generation_uses_independent_stride",
            "test_step5_nhd_tuple_uses_matching_rank_whole_blocks",
        },
        namespace,
    )
    namespace["test_whole_block_mla_address_generation_uses_independent_stride"]()
    namespace["test_step5_nhd_tuple_uses_matching_rank_whole_blocks"]()


def transfer_config(connector="MooncakeConnectorV2", role="kv_producer"):
    # Compile the real config properties, avoiding its platform/config imports.
    source_root = Path(os.environ.get("VLLM_SOURCE_ROOT", ROOT.parent / "vllm-private"))
    tree = ast.parse((source_root / "vllm/config/kv_transfer.py").read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "KVTransferConfig")
    cls.decorator_list = []
    cls.body = [
        node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "is_kv_transfer_instance"
    ]
    aliases = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id in {"KVProducer", "KVConsumer", "KVRole"}
            for target in node.targets
        )
    ]
    namespace = {"get_args": get_args, "Literal": Literal}
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[*aliases, cls], type_ignores=[])), "kv_transfer.py", "exec"),
        namespace,
    )
    config = namespace["KVTransferConfig"]()
    config.kv_connector, config.kv_role = connector, role
    config.kv_connector_module_path = None
    return config


def step5_config(transfer):
    return NS(
        kv_transfer_config=transfer,
        parallel_config=NS(pipeline_parallel_size=1, prefill_context_parallel_size=1, decode_context_parallel_size=1),
        speculative_config=None,
        cache_config=NS(enable_prefix_caching=False),
        model_config=NS(is_multimodal_model=False, hf_config=NS(fp32_residual_connection=False)),
    )


def step5_model_namespace():
    source = ROOT / "vllm_ascend/models/step5/model.py"
    tree = ast.parse(source.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AscendStep5ForCausalLM")
    constructor = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    helper = [node for node in tree.body if getattr(node, "name", None) == "validate_step5_kv_transfer_config"]
    core_source = ROOT / "vllm_ascend/core/step5_kv_cache.py"
    core_tree = ast.parse(core_source.read_text())
    validator = next(n for n in core_tree.body if getattr(n, "name", None) == "validate_step5_kv_transfer_config")
    core_namespace = {}
    exec(compile(ast.Module(body=[validator], type_ignores=[]), str(core_source), "exec"), core_namespace)
    namespace = dict(
        nn=NS(Module=NS(__init__=lambda self: None)),
        maybe_prefix=lambda *args: "model",
        _validate_step5_kv_transfer_config=core_namespace["validate_step5_kv_transfer_config"],
    )

    def reached_model(**kwargs):
        raise RuntimeError("reached model allocation")

    namespace["AscendStep5Model"] = reached_model
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[future, *helper, constructor], type_ignores=[])),
            str(source),
            "exec",
        ),
        namespace,
    )
    return namespace


def call_step5_guard(config):
    namespace = step5_model_namespace()
    assert "validate_step5_kv_transfer_config" in namespace, "Step5-local transfer guard is missing"
    return namespace["validate_step5_kv_transfer_config"](config)


@pytest.mark.parametrize("connector", ["MooncakeConnectorV2", "MooncakePullConnector"])
@pytest.mark.parametrize("role", ["kv_producer", "kv_consumer"])
def test_step5_guard_accepts_registered_pull_protocol(connector, role):
    call_step5_guard(step5_config(transfer_config(connector, role)))


@pytest.mark.parametrize("transfer", [None, transfer_config(None, "kv_both")])
def test_step5_guard_inactive_transfer_preserves_basic_behavior(transfer):
    # No other configuration is inspected when this is not a transfer instance.
    call_step5_guard(NS(kv_transfer_config=transfer))


@pytest.mark.parametrize(
    "connector",
    [
        "MooncakeConnector",
        "MooncakeConnectorV1",
        "MooncakePushConnector",
        "MooncakeHybridConnector",
        "AscendStoreConnector",
        "MooncakeStoreConnector",
        "NixlConnector",
        "MultiConnector",
    ],
)
@pytest.mark.parametrize("role", ["kv_producer", "kv_both"])
def test_step5_guard_rejects_unsupported_transfer_names(connector, role):
    with pytest.raises(ValueError, match="Step5 KV transfer.*MooncakeConnectorV2.*MooncakePullConnector"):
        call_step5_guard(step5_config(transfer_config(connector, role)))


@pytest.mark.parametrize(
    "section,field,value,message",
    [
        ("parallel_config", "pipeline_parallel_size", 2, "pipeline_parallel_size=1"),
        ("parallel_config", "prefill_context_parallel_size", 2, "prefill_context_parallel_size=1"),
        ("parallel_config", "decode_context_parallel_size", 2, "decode_context_parallel_size=1"),
        ("cache_config", "enable_prefix_caching", True, "prefix caching disabled"),
        ("model_config", "is_multimodal_model", True, "text-only"),
        ("kv_transfer_config", "kv_connector_module_path", "custom.connector", "registered Ascend"),
    ],
)
def test_step5_guard_rejects_out_of_slice_config(section, field, value, message):
    config = step5_config(transfer_config())
    setattr(getattr(config, section), field, value)
    with pytest.raises(ValueError, match=message):
        call_step5_guard(config)


@pytest.mark.parametrize(
    "speculation", [NS(method="mtp", num_speculative_tokens=3), NS(method="eagle", num_speculative_tokens=1)]
)
def test_step5_guard_rejects_speculative_transfer(speculation):
    config = step5_config(transfer_config())
    config.speculative_config = speculation
    with pytest.raises(ValueError, match="speculative decoding disabled"):
        call_step5_guard(config)


def test_step5_constructor_rejects_transfer_before_model_allocation():
    namespace = step5_model_namespace()
    with pytest.raises(ValueError, match="Step5 KV transfer"):
        namespace["__init__"](NS(), vllm_config=step5_config(transfer_config("NixlConnector")))


@pytest.mark.parametrize("connector", ["MooncakeConnectorV2", "MooncakePullConnector"])
def test_step5_guard_and_constructor_reject_kv_both_before_allocation(connector):
    config = step5_config(transfer_config(connector, "kv_both"))
    with pytest.raises(ValueError, match="kv_producer or kv_consumer.*kv_both"):
        call_step5_guard(config)
    with pytest.raises(ValueError, match="kv_producer or kv_consumer.*kv_both"):
        step5_model_namespace()["__init__"](NS(), vllm_config=config)


@pytest.mark.parametrize("transfer", [None, transfer_config(), transfer_config("MooncakePullConnector", "kv_consumer")])
def test_step5_constructor_basic_and_allowed_transfer_reach_model(transfer):
    with pytest.raises(RuntimeError, match="reached model allocation"):
        step5_model_namespace()["__init__"](NS(), vllm_config=step5_config(transfer))


def upstream_method(relative_path, class_name, method_name, namespace=None):
    source_root = Path(os.environ.get("VLLM_SOURCE_ROOT", ROOT.parent / "vllm-private"))
    source = source_root / relative_path
    tree = ast.parse(source.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == method_name)
    namespace = dict(namespace or {})
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[future, method], type_ignores=[])), str(source), "exec"),
        namespace,
    )
    return namespace[method_name]


def test_step5_loaded_groups_skip_zeroing_including_actual_ring_manager():
    namespace = dict(
        KVConnectorBase_V1=type("Base", (), {}), SupportsHMA=type("HMA", (), {}), vllm_version_is=lambda version: False
    )
    load_definitions("connector.py", {"MooncakeBaseConnector"}, namespace)
    connector = namespace["MooncakeBaseConnector"].__new__(namespace["MooncakeBaseConnector"])
    connector._kv_cache_config = NS(transfer_group_ids=(0, 1, 2))
    loaded = connector.get_loaded_kv_cache_group_ids(NS())
    assert loaded == (0, 1, 2)  # main, summary, request-private ring
    manager_source = "vllm/v1/core/single_type_kv_cache_manager.py"
    external = upstream_method(
        manager_source,
        "SingleTypeKVCacheManager",
        "allocate_external_computed_blocks",
        {"cdiv": lambda n, d: (n + d - 1) // d},
    )
    ring_external = upstream_method(manager_source, "CircularBufferManager", "allocate_external_computed_blocks")
    claim_ring = upstream_method(manager_source, "CircularBufferManager", "_claim_ring_block")
    managers = []
    for i in range(3):
        manager = NS(
            num_cached_block={},
            req_to_blocks=defaultdict(list),
            new_block_ids=[],
            _record_new_block_ids=True,
            block_size=16,
            get_num_skipped_tokens=lambda tokens: 0,
        )
        manager.block_pool = NS(get_new_blocks=lambda n: [NS(block_id=10 + j) for j in range(n)])
        manager.add_local_computed_blocks = lambda *args: None
        manager._claim_ring_block = lambda *args, m=manager: claim_ring(m, *args)
        method = ring_external if i == 2 else external
        manager.allocate_external_computed_blocks = lambda *args, m=manager, fn=method, **kwargs: fn(m, *args, **kwargs)
        managers.append(manager)
    allocate = upstream_method(
        "vllm/v1/core/kv_cache_coordinator.py", "KVCacheCoordinator", "allocate_new_computed_blocks"
    )
    allocate(NS(single_type_managers=managers), "req", ([], [], []), 0, 129, skip_zeroing_group_ids=loaded)
    assert [len(manager.req_to_blocks["req"]) for manager in managers] == [9, 9, 1]
    assert all(manager.new_block_ids == [] for manager in managers)
    assert managers[2]._claim_ring_block("req") == []
    assert managers[2].new_block_ids == []  # later allocation cannot zero the loaded ring
    managers[2]._claim_ring_block("ordinary")
    assert managers[2].new_block_ids == [10]  # control: a new non-loaded ring needs zeroing


@pytest.mark.parametrize("failed_group", [None, "main", "summary", "ring"])
def test_step5_actual_pull_waits_all_groups_and_propagates_failure(production, failed_group):
    specs = [
        FullAttentionSpec(),
        FullAttentionSpec(num_kv_heads=1, head_size=256, head_size_v=0, dtype=torch.uint8, tokens_per_state=8),
        CircularBufferSpec(),
    ]
    shapes = [[(16, 2, 4)] * 2, [(1, 2, 256)], [(1, 16, 32)]]
    local = metadata(shapes, dtypes=[["torch.bfloat16"] * 2, ["torch.uint8"], ["torch.bfloat16"]])
    remote = metadata(shapes, dtypes=local.block_dtypes)
    topology = grouped(remote, tp=1)
    peer = remote.metadata_by_pcp_rank[0].metadata_by_tp_rank[0]
    peer.local_ip, peer.te_rpc_port = "cpu", 9000
    thread = thread_from(production, specs, local, rank=0, tp=1)
    thread.local_metadata.local_ip = "cpu"
    groups, pairs = thread._build_remote_transfer_layout(topology)
    thread.remote_tp_rank_groups = {"p": groups}
    thread.remote_layer_index_pairs = {"p": pairs}
    thread._get_remote_metadata = lambda *args: topology
    thread._handle_requests.__globals__["as_completed"] = as_completed
    thread._handle_requests.__globals__["time"] = NS(perf_counter=lambda: 0)
    thread._handle_requests.__globals__["logger"] = NS(info=lambda *args: None, exception=lambda *args: None)
    request = NS(
        local_block_ids=([1], [3], [5]),
        local_full_block_ids=([1], [3], [5]),
        remote_block_ids=([2], [4], [6]),
        local_num_prompt_tokens=16,
        remote_num_prompt_tokens=16,
        num_computed_tokens=0,
    )
    expected_addresses = {1000 + 256: "main", 1000 + 3 * 512: "summary", 1000 + 5 * 1024: "ring"}
    transferred = []

    def network_read(session, src, dst, lengths):
        transferred.extend(expected_addresses[address] for address in src)
        assert set(transferred) == {"main", "summary", "ring"}
        assert len(src) == len(dst) == len(lengths) == 4
        return -1 if failed_group in transferred else 0

    thread.engine = NS(batch_transfer_sync_read=network_read)
    with ThreadPoolExecutor(max_workers=1) as executor:
        thread.executor = executor
        failed = thread._handle_requests("p", "cpu", 9000, {"req": request})
    assert failed == ({"req"} if failed_group else set())
    assert set(transferred) == {"main", "summary", "ring"}


@pytest.mark.parametrize("failed_group", [None, "main", "summary", "ring"])
def test_step5_staggered_remote_futures_do_not_publish_before_all_groups(production, lifecycle, failed_group):
    specs = [
        FullAttentionSpec(),
        FullAttentionSpec(num_kv_heads=1, head_size=256, head_size_v=0, dtype=torch.uint8, tokens_per_state=8),
        CircularBufferSpec(),
    ]
    shapes = [[(16, 2, 4)] * 2, [(1, 2, 256)], [(1, 16, 32)]]
    local = metadata(shapes, dtypes=[["torch.bfloat16"] * 2, ["torch.uint8"], ["torch.bfloat16"]])
    remote = metadata(shapes, dtypes=local.block_dtypes)
    topology = grouped(remote, tp=2)
    for rank, peer in remote.metadata_by_pcp_rank[0].metadata_by_tp_rank.items():
        peer.local_ip, peer.te_rpc_port = "cpu", 9000 + rank
    receiver = thread_from(production, specs, local, rank=1, tp=2)
    receiver.local_metadata.local_ip = "cpu"
    groups, pairs = receiver._build_remote_transfer_layout(topology)
    assert groups[0] == {(0, 0): [[1]], (1, 1): [[0, 1]], (2, 2): [[0, 1]]}
    receiver.remote_tp_rank_groups = {"p": groups}
    receiver.remote_layer_index_pairs = {"p": pairs}
    receiver._get_remote_metadata = lambda *args: topology
    receiver.finished_requests = queue.SimpleQueue()
    receiver.can_report_invalid_block_ids = False
    receiver.ready_event = threading.Event()
    receiver.device = "cpu"
    request = NS(
        local_block_ids=([1], [3], [5]),
        local_full_block_ids=([1], [3], [5]),
        remote_block_ids=([2], [4], [6]),
        local_num_prompt_tokens=16,
        remote_num_prompt_tokens=16,
        num_computed_tokens=0,
        remote_host="cpu",
        remote_port=9000,
    )
    pending = queue.Queue()
    pending.put(("p", {"req": request}))

    def get_once():
        if pending.unfinished_tasks == 0:
            raise StopIteration  # terminate the production infinite loop after one batch
        return pending.get()

    receiver.request_queue = NS(get=get_once, task_done=pending.task_done)
    namespace = receiver.run.__globals__
    namespace.update(
        torch=NS(**(vars(torch) | {"npu": NS(set_device=lambda device: None)})),
        time=NS(perf_counter=lambda: 0),
    )
    namespace["logger"] = NS(info=lambda *args: None, exception=lambda *args: None)
    entered = {group: threading.Event() for group in ("main", "summary", "ring")}
    gates = {group: threading.Event() for group in entered}
    completed_ranks = queue.Queue()
    sessions = []

    def tracked_completed(futures):
        for future in as_completed(futures):
            yield future
            completed_ranks.put(futures[future][2])  # result/failure already consumed

    namespace["as_completed"] = tracked_completed
    addresses = {1256: "main", 2536: "summary", 6120: "ring"}

    def network_read(session, src, dst, lengths):
        sessions.append(session)
        batch_groups = list(dict.fromkeys(addresses[address] for address in src))
        for group in batch_groups:
            entered[group].set()
            assert gates[group].wait(5), f"test did not release {group}"
        return -1 if failed_group in batch_groups else 0

    receiver.engine = NS(batch_transfer_sync_read=network_read)
    with ThreadPoolExecutor(max_workers=2) as transfers, ThreadPoolExecutor(max_workers=1) as runner:
        receiver.executor = transfers
        task = runner.submit(receiver.run)
        try:
            assert entered["main"].wait(5) and entered["summary"].wait(5)
            assert receiver.get_and_clear_finished_requests().finished_recving == set()
            gates["main"].set()
            assert completed_ranks.get(timeout=5) == 1
            assert not task.done()
            assert receiver.get_and_clear_finished_requests().finished_recving == set()
            gates["summary"].set()
            assert entered["ring"].wait(5)
            assert not task.done()
            assert receiver.get_and_clear_finished_requests().finished_recving == set()
            gates["ring"].set()
            with pytest.raises(StopIteration):
                task.result(timeout=5)
            assert completed_ranks.get(timeout=5) == 0
        finally:
            for gate in gates.values():
                gate.set()
    results = receiver.get_and_clear_finished_requests()
    assert results.finished_recving == {"req"}
    assert results.failed_recving == ({"req"} if failed_group else set())
    assert len(sessions) == 2  # main shard and independently selected sidecar replica


@pytest.fixture
def lifecycle(registration, production):
    namespace = dict(
        registration,
        field=field,
        dataclass=dataclass,
        MooncakeBaseConnectorScheduler=type("BaseScheduler", (), {}),
        OrderedDict=OrderedDict,
        envs=NS(VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT=10),
        time=NS(time=lambda: 100),
        logger=NS(
            info=lambda *args: None, debug=lambda *args: None, warning=lambda *args: None, error=lambda *args: None
        ),
    )
    source_root = Path(os.environ.get("VLLM_SOURCE_ROOT", ROOT.parent / "vllm-private"))
    load_definitions(
        source_root / "vllm/distributed/kv_transfer/kv_connector/v1/base.py", {"KVConnectorTransferResults"}, namespace
    )
    load_definitions("base_scheduler.py", {"MooncakeBaseConnectorScheduler"}, namespace)
    load_definitions(
        "pull_scheduler.py", {"MooncakePullConnectorScheduler", "MooncakeSchedulerSendingThread"}, namespace
    )
    load_definitions("pull_worker.py", {"MooncakePullConnectorWorker"}, namespace)
    production.get_and_clear_finished_requests.__globals__["KVConnectorTransferResults"] = namespace[
        "KVConnectorTransferResults"
    ]
    return namespace


def test_step5_worker_failed_receive_is_terminal_not_success(lifecycle, production):
    receiver = production.__new__(production)
    receiver.finished_requests = queue.SimpleQueue()
    receiver.finished_requests.put(("ok", False))
    receiver.finished_requests.put(("failed", True))
    receiver.can_report_invalid_block_ids = False  # independent groups cannot use flat IDs
    receiver.invalid_block_ids = set()
    receiver.invalid_block_ids_lock = threading.Lock()
    receiver._mark_request_failed(NS(local_block_ids=([1], [1], [1])))
    worker = lifecycle["MooncakePullConnectorWorker"].__new__(lifecycle["MooncakePullConnectorWorker"])
    worker._recving_thread = receiver
    results = worker.get_transfer_results()
    assert results.finished_recving == {"ok", "failed"}
    assert results.failed_recving == {"failed"}
    assert results.finished_recving - results.failed_recving == {"ok"}
    assert worker.get_transfer_results().finished_recving == set()
    assert worker.get_block_ids_with_load_errors() == set()
    handle_failed = upstream_method(
        "vllm/v1/core/sched/scheduler.py",
        "Scheduler",
        "_handle_failed_recving",
        {"RequestStatus": NS(WAITING_FOR_REMOTE_KVS="loading")},
    )
    request = NS(status="loading", num_computed_tokens=129)
    scheduler = NS(requests={"failed": request}, recompute_kv_load_failures=False, failed_recving_kv_req_ids=set())
    assert handle_failed(scheduler, results.failed_recving) == {"failed"}  # fail policy, not continuation
    scheduler.recompute_kv_load_failures = True
    assert handle_failed(scheduler, results.failed_recving) == set()
    assert scheduler.failed_recving_kv_req_ids == {"failed"}
    assert request.num_computed_tokens == 0


@pytest.mark.parametrize("failed_worker", [None, 0, 1, 2])
def test_step5_producer_release_waits_for_staggered_worker_aggregation(lifecycle, production, failed_worker):
    sender = lifecycle["MooncakeSchedulerSendingThread"].__new__(lifecycle["MooncakeSchedulerSendingThread"])
    sender.state_lock = threading.Lock()
    sender.is_step5 = True
    sender.timed_out_requests = set()
    sender.delayed_free_requests = OrderedDict()
    sender.finished_requests = queue.SimpleQueue()
    sender.add_delayed_request("producer", 100)
    cls = lifecycle["MooncakePullConnectorScheduler"]
    producer = cls.__new__(cls)
    producer._reqs_recv_info = {}
    producer._reqs_need_send = {"producer": 100}
    producer._sending_thread = sender
    consumer = cls.__new__(cls)
    consumer._reqs_recv_info = {"consumer": ("cpu", 9000, "producer")}
    consumer._sending_thread = None
    consumer.kv_transfer_config = NS(kv_load_failure_policy="fail")
    notifications = []
    consumer._recving_thread = NS(add_request=lambda *args: notifications.append(args))

    def connector_output(**kwargs):
        return NS(
            **(
                dict(
                    finished_recving=set(),
                    failed_recving=set(),
                    finished_sending=set(),
                    expected_finished_count=0,
                    invalid_block_ids=set(),
                    kv_connector_stats=None,
                    kv_connector_worker_meta=None,
                    kv_cache_events=None,
                )
                | kwargs
            )
        )

    namespace = {"KVConnectorOutput": connector_output}
    source_root = Path(os.environ.get("VLLM_SOURCE_ROOT", ROOT.parent / "vllm-private"))
    load_definitions(
        source_root / "vllm/distributed/kv_transfer/kv_connector/utils.py", {"KVOutputAggregator"}, namespace
    )
    aggregator = namespace["KVOutputAggregator"](expected_finished_count=3)
    receivers = []
    workers = []
    arrivals = [Future() for _ in range(3)]
    for _ in arrivals:
        receiver = production.__new__(production)
        receiver.finished_requests = queue.SimpleQueue()
        receivers.append(receiver)
        worker_cls = lifecycle["MooncakePullConnectorWorker"]
        worker = worker_cls.__new__(worker_cls)
        worker._recving_thread = receiver
        workers.append(worker)
    for worker_index in (1, 0, 2):
        # Controlled arrivals across scheduling iterations, not a pre-aggregated output.
        assert not arrivals[worker_index].done()
        arrivals[worker_index].set_result(worker_index == failed_worker)
        receivers[worker_index].finished_requests.put(("consumer", arrivals[worker_index].result()))
        outputs = []
        for worker in workers:
            result = worker.get_transfer_results()
            outputs.append(
                NS(
                    kv_connector_output=connector_output(
                        finished_recving=result.finished_recving,
                        failed_recving=result.failed_recving,
                    )
                )
            )
        output = aggregator.aggregate(outputs).kv_connector_output
        consumer.update_connector_output(output)
        producer.update_connector_output(output)
        if worker_index != 2:
            assert not output.finished_recving
            assert output.failed_recving == set()  # failure held until the other workers quiesce
            assert not output.finished_sending
            assert notifications == []
            assert producer._reqs_need_send == {"producer": 100}
            assert sender.delayed_free_requests == {"producer": 100}
    assert output.finished_recving == {"consumer"}
    assert output.failed_recving == ({"consumer"} if failed_worker is not None else set())
    assert notifications == [("cpu", 9000, "producer")]
    assert producer._reqs_need_send == {"producer": 100}  # queuing D notification is not P completion
    sender._handle_finished_request("producer")  # network delivery boundary
    output.finished_recving = set()
    producer.update_connector_output(output)
    assert output.finished_sending == {"producer"}
    assert producer._reqs_need_send == {}
    sender._handle_finished_request("producer")  # duplicate DONE cannot release twice
    assert sender.get_and_clear_finished_requests() == set()


@pytest.mark.parametrize("is_step5", [False, True])
def test_step5_producer_timeout_retains_until_done_nonstep5_still_frees(lifecycle, is_step5):
    sender = lifecycle["MooncakeSchedulerSendingThread"].__new__(lifecycle["MooncakeSchedulerSendingThread"])
    sender.state_lock = threading.Lock()
    sender.is_step5 = is_step5
    sender.timed_out_requests = set()
    sender.delayed_free_requests = OrderedDict()
    sender.finished_requests = queue.SimpleQueue()
    errors = []
    lifecycle["logger"].error = lambda *args: errors.append(args)
    sender.add_delayed_request("boundary", 90)
    assert sender.get_and_clear_finished_requests() == set()  # exactly timeout: retained
    sender.add_delayed_request("boundary", 89)
    sender.add_delayed_request("second", 88)
    cls = lifecycle["MooncakePullConnectorScheduler"]
    producer = cls.__new__(cls)
    producer._reqs_recv_info = {}
    producer._reqs_need_send = {"boundary": 89, "second": 88}
    producer._sending_thread = sender
    output = NS(finished_recving=set(), failed_recving=set(), finished_sending=set())
    producer.update_connector_output(output)
    if not is_step5:
        assert output.finished_sending == {"boundary", "second"}
        assert producer._reqs_need_send == {}
        return
    assert output.finished_sending == set()
    assert set(sender.delayed_free_requests) == {"boundary", "second"}
    assert set(producer._reqs_need_send) == {"boundary", "second"}
    assert len(errors) == 2
    assert all("retaining blocks" in error[0] and "DONE" in error[0] for error in errors)
    producer.update_connector_output(output)
    assert len(errors) == 2  # no repeated timeout logging or early release
    sender._handle_finished_request("second")
    producer.update_connector_output(output)
    assert output.finished_sending == {"second"}
    assert set(producer._reqs_need_send) == {"boundary"}
    sender._handle_finished_request("boundary")
    producer.update_connector_output(output)
    assert output.finished_sending == {"boundary", "second"}
    assert sender.delayed_free_requests == {}
    assert sender.timed_out_requests == set()
    sender._handle_finished_request("boundary")
    assert sender.get_and_clear_finished_requests() == set()


@pytest.mark.parametrize("already_received", [False, True])
def test_step5_actual_scheduler_cancellation_defers_blocks_until_receive(already_received):
    status = NS(
        WAITING_FOR_REMOTE_KVS="loading",
        RUNNING="running",
        WAITING_FOR_STREAMING_REQ="streaming",
        is_finished=lambda state: state == "aborted",
    )
    source = "vllm/v1/core/sched/scheduler.py"
    finish = upstream_method(source, "Scheduler", "finish_requests", {"RequestStatus": status})
    free = upstream_method(source, "Scheduler", "_free_request")
    update = upstream_method(
        source,
        "Scheduler",
        "_update_from_kv_xfer_finished",
        {"RequestStatus": status, "logger": NS(debug=lambda *args: None)},
    )
    request = NS(request_id="req", status="loading", is_finished=lambda: request.status == "aborted")
    freed = []
    scheduler = NS(
        requests={"req": request},
        finished_recving_kv_req_ids={"req"} if already_received else set(),
        failed_recving_kv_req_ids=set(),
        waiting=NS(remove_requests=lambda requests: None),
        kv_holding_waiting=NS(remove_requests=lambda requests: None),
        deferred_waiting=NS(difference_update=lambda requests: None),
        aux_output_connector=None,
        ec_connector=None,
        _inflight_prefills=NS(discard=lambda req: None),
        _set_kv_fetch_stage=lambda *args: None,
        _connector_finished=lambda req: (False, None),
        encoder_cache_manager=NS(free=lambda req: None),
        finished_req_ids=set(),
        finished_req_ids_dict=None,
        _free_blocks=lambda req: freed.append(req.request_id),
        connector=None,
    )
    scheduler._free_request = lambda req, **kwargs: free(scheduler, req, **kwargs)
    assert finish(scheduler, "req", "aborted") == [request]
    assert freed == (["req"] if already_received else [])
    if not already_received:
        update(scheduler, NS(finished_recving={"req"}, finished_sending=set()))
        assert freed == ["req"]


@pytest.mark.parametrize(
    "prompt_len,remote_len,computed,expected",
    [
        (9, 8, 0, 8),
        (9, 8, 4, 4),
        (9, 8, 8, 0),
        (8, 8, 0, 8),
        (9, None, 0, 9),
        (2, 1, 0, 1),
        (130, 129, 0, 129),
        (129, 128, 0, 128),
        (9, 0, 0, 0),
    ],
)
def test_step5_remote_prefix_credit_does_not_include_producer_sample(
    lifecycle, prompt_len, remote_len, computed, expected
):
    cls = lifecycle["MooncakePullConnectorScheduler"]
    scheduler = cls.__new__(cls)
    scheduler.need_truncate = False
    scheduler.vllm_config = NS(model_config=NS(hf_text_config=NS(model_type="step5")))
    params = {"do_remote_prefill": True}
    if remote_len is not None:
        params["remote_num_prompt_tokens"] = remote_len
    request = NS(prompt_token_ids=list(range(prompt_len)), kv_transfer_params=params)
    assert scheduler.get_num_new_matched_tokens(request, computed) == (expected, expected > 0)


@pytest.mark.parametrize("remote_len", [-1, 1.5, True, "8", None])
def test_step5_remote_prefix_credit_rejects_invalid_source_count(lifecycle, remote_len):
    cls = lifecycle["MooncakePullConnectorScheduler"]
    scheduler = cls.__new__(cls)
    scheduler.need_truncate = False
    scheduler.vllm_config = NS(model_config=NS(hf_text_config=NS(model_type="step5")))
    request = NS(
        prompt_token_ids=list(range(9)),
        kv_transfer_params={
            "do_remote_prefill": True,
            "remote_num_prompt_tokens": remote_len,
        },
    )
    with pytest.raises(ValueError, match="remote_num_prompt_tokens"):
        scheduler.get_num_new_matched_tokens(request, 0)


def test_non_step5_remote_credit_keeps_legacy_count(lifecycle):
    cls = lifecycle["MooncakePullConnectorScheduler"]
    scheduler = cls.__new__(cls)
    scheduler.need_truncate = False
    scheduler.vllm_config = NS(model_config=NS(hf_text_config=NS(model_type="other")))
    request = NS(
        prompt_token_ids=list(range(9)),
        kv_transfer_params={
            "do_remote_prefill": True,
            "remote_num_prompt_tokens": 8,
        },
    )
    assert scheduler.get_num_new_matched_tokens(request, 0) == (9, True)
