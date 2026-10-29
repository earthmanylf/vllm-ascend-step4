# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contracts for the Step4 FA3 -> MRV2 port, with the NPU ABI mocked.

The real source classes/functions are compiled in isolation, following the
Step5 contract tests. These tests are not device-accuracy evidence.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
FA3 = ROOT / "vllm_ascend/attention/flash_attention_v3.py"
ATTN_UTILS = ROOT / "vllm_ascend/worker/v2/attn_utils.py"


def load_definitions(path, namespace, names=None):
    namespace.setdefault("__name__", __name__)
    tree = ast.parse(path.read_text())
    body = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and (names is None or node.name in names)
    ]
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *body], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)  # noqa: S102
    return namespace


def load_real_registry(namespace):
    root = os.environ.get("VLLM_SOURCE_ROOT")
    if root:
        path = Path(root) / "vllm/v1/attention/backends/registry.py"
    else:
        spec = importlib.util.find_spec("vllm")
        if spec is None or spec.origin is None:
            raise RuntimeError("Set VLLM_SOURCE_ROOT to test the actual backend registry")
        path = Path(spec.origin).parent / "v1/attention/backends/registry.py"
    logger = ModuleType("vllm.logger")
    logger.init_logger = lambda _: Mock()
    resolver = ModuleType("vllm.utils.import_utils")
    resolver.resolve_obj_by_qualname = lambda name: namespace[name.rsplit(".", 1)[-1]]
    spec = importlib.util.spec_from_file_location("fa3_registry_contract", path)
    registry = importlib.util.module_from_spec(spec)
    with patch.dict("sys.modules", {"vllm.logger": logger, "vllm.utils.import_utils": resolver}):
        spec.loader.exec_module(registry)
    return registry


@dataclass
class Metadata:
    num_actual_tokens: int = 0
    query_start_loc: object = None
    query_start_loc_gpu: object = None
    seq_lens: object = None
    seq_lens_gpu: object = None
    block_tables: object = None
    slot_mapping: object = None
    max_query_len: int = 0
    attn_state: object = None
    causal: bool = True
    model_runner_type: str = "generate"


class BuilderBase:
    def __class_getitem__(cls, _):
        return cls

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        self.layer_names = layer_names
        self.vllm_config = vllm_config
        self.device = device


class ImplBase:
    def __init__(self, num_heads, head_size, scale, num_kv_heads, sliding_window=None, logits_soft_cap=None, **kwargs):
        self.num_heads, self.num_kv_heads = num_heads, num_kv_heads
        self.head_size, self.scale = head_size, scale
        self.sliding_window = sliding_window
        self.kv_cache_dtype = torch.bfloat16
        self.enable_c8_quant = False
        self.attn_type = "decoder"
        self.alibi_slopes = self.sinks = None
        self.vllm_config = kwargs["vllm_config"]


@pytest.fixture
def fa3():
    calls = []

    def scheduler(*, num_blocks, max_num_blocks_per_seq, softmax_scale, **kwargs):
        calls.append(
            dict(
                num_blocks=num_blocks,
                max_num_blocks_per_seq=max_num_blocks_per_seq,
                softmax_scale=softmax_scale,
                **kwargs,
            )
        )
        return torch.full((16,), len(calls), dtype=torch.uint8)

    namespace = dict(
        inspect=inspect,
        dataclass=dataclass,
        field=field,
        torch=torch,
        AscendMetadata=Metadata,
        AscendAttentionBackend=type("Backend", (), {}),
        AscendAttentionBackendImpl=ImplBase,
        AttentionMetadataBuilder=BuilderBase,
        AttentionCGSupport=SimpleNamespace(ALWAYS="always", NEVER="never"),
        AttentionType=SimpleNamespace(DECODER="decoder"),
        AscendCommonAttentionMetadata=object,
        VllmConfig=object,
        AttentionSpec=object,
        AscendAttentionState=object,
        get_scheduler_metadata=scheduler,
        flash_attn_with_kvcache=Mock(side_effect=lambda query, *args, **kwargs: query + 1),
        record_attention_compute_start=Mock(),
    )
    namespace["flash_attn_with_kvcache"].block_stride_abi_version = 1
    registry = load_real_registry(namespace)
    namespace["register_backend"] = registry.register_backend
    load_definitions(FA3, namespace)
    namespace["calls"] = calls
    namespace["registry"] = registry
    return SimpleNamespace(**namespace)


def config():
    return SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=4),
        compilation_config=SimpleNamespace(cudagraph_capture_sizes=[]),
        model_config=SimpleNamespace(runner_type="generate"),
        use_v2_model_runner=True,
        parallel_config=SimpleNamespace(prefill_context_parallel_size=1, decode_context_parallel_size=1),
        kv_transfer_config=None,
    )


def make_builder(fa3, num_blocks=19, sliding_window=128, value_dim=128):
    cfg = config()
    impl = fa3.AscendFlashAttentionImpl(
        8, 128, 0.125, 2, sliding_window=sliding_window, logits_soft_cap=0.0, vllm_config=cfg
    )
    builder = fa3.AscendFlashAttentionMetadataBuilder(
        SimpleNamespace(block_size=128), ["layer"], cfg, torch.device("cpu")
    )
    cache = (
        torch.zeros(num_blocks, 128, 2, 128, dtype=torch.bfloat16),
        torch.zeros(num_blocks, 128, 2, value_dim, dtype=torch.bfloat16),
    )
    context = {"layer": SimpleNamespace(impl=impl, kv_cache=cache)}
    builder.bind_fa3_kv_cache(context)
    return builder, impl, context


def common():
    return SimpleNamespace(
        num_reqs=3,
        query_start_loc=torch.tensor([0, 1, 4, 8], dtype=torch.int32),
        seq_lens=torch.tensor([129, 259, 77], dtype=torch.int32),
        block_table_tensor=torch.tensor([[17, 18, 0], [2, 3, 4], [5, 6, 7]], dtype=torch.int32),
        slot_mapping=torch.arange(8),
        num_input_tokens=8,
        num_actual_tokens=8,
        max_query_len=4,
        attn_state="spec",
        causal=True,
    )


def test_explicit_pool_capacity_and_swa_parameters(fa3):
    builder, _, _ = make_builder(fa3, value_dim=64)
    metadata = builder.build(0, common())
    call = fa3.calls[-1]
    assert call["num_blocks"] == 19
    assert call["max_num_blocks_per_seq"] == 3
    assert call["max_seqlen_k"] == 384
    assert call["window_size"] == (127, 0)
    assert call["softmax_scale"] == 0.125
    assert call["headdim_v"] == 64
    assert metadata.seq_lens.device.type == "cpu"
    assert metadata.seq_lens.data_ptr() == call["cache_seqlens"].data_ptr()


def test_backend_name_roundtrips_through_real_vllm_registry(fa3):
    # Attention.__init__ performs exactly this lookup after constructing impl.
    backend = fa3.registry.AttentionBackendEnum[fa3.AscendFlashAttentionBackend.get_name()]
    assert backend.name == "ASCEND_DEVICE_FA3"
    assert backend.get_class() is fa3.AscendFlashAttentionBackend
    assert backend.get_path().endswith(".AscendFlashAttentionBackend")


def test_padding_uses_device_boundary_not_padded_token_count(fa3):
    builder, _, _ = make_builder(fa3)
    metadata = builder.build_for_cudagraph_capture(common(), num_actual_reqs=2)
    assert metadata.query_start_loc.tolist() == [0, 1, 4, 4, 4, 4]
    assert metadata.seq_lens.tolist() == [129, 259, 0, 0, 0]
    assert torch.count_nonzero(metadata.block_tables[2:]) == 0


def test_graph_addresses_stable_when_request_count_changes(fa3):
    builder, _, _ = make_builder(fa3)
    source = common()
    first = builder.build_for_cudagraph_capture(source, num_actual_reqs=2)
    first_tiling = next(iter(first.scheduler_metadata.values()))
    source.query_start_loc[1] = 2
    second = builder.build(0, source, num_actual_reqs=1, full_graph_mode=True)
    assert first.query_start_loc.data_ptr() == second.query_start_loc.data_ptr()
    assert second.query_start_loc.tolist() == [0, 2, 2, 2, 2, 2]
    second_tiling = next(iter(second.scheduler_metadata.values()))
    assert first_tiling.data_ptr() == second_tiling.data_ptr()
    assert torch.all(second_tiling == 2)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_full_writer_slots_refresh_at_the_captured_address_and_mask_pad(fa3, dtype):
    builder, _, context = make_builder(fa3)
    source = common()
    source.slot_mapping = torch.arange(128, 136, dtype=dtype)
    captured = builder.build_for_cudagraph_capture(source, num_actual_reqs=2)
    address = captured.slot_mapping.data_ptr()
    assert address != source.slot_mapping.data_ptr()
    for real in (1, 0, 2):
        actual = common()
        actual.slot_mapping = torch.arange(256, 264, dtype=dtype)
        replay = builder.build(0, actual, num_actual_reqs=real, full_graph_mode=True)
        boundary = actual.query_start_loc[real].item()
        assert replay.slot_mapping.data_ptr() == address
        assert captured.slot_mapping[:boundary].tolist() == actual.slot_mapping[:boundary].tolist()
        assert captured.slot_mapping[boundary:].tolist() == [-1] * (8 - boundary)
    builder.bind_fa3_kv_cache(context)
    assert not builder.graph_slot_buffers


def test_full_writer_rejects_changed_dtype_and_short_source(fa3):
    builder, _, _ = make_builder(fa3)
    builder.build_for_cudagraph_capture(common())
    actual = common()
    actual.slot_mapping = actual.slot_mapping.int()
    with pytest.raises(ValueError, match="dtype changed"):
        builder.build(0, actual, full_graph_mode=True)
    actual = common()
    actual.slot_mapping = actual.slot_mapping[:7]
    with pytest.raises(ValueError, match="captured token extent"):
        builder.build(0, actual, full_graph_mode=True)


def test_capture_sizes_resolved_after_builder_construction(fa3):
    builder, _, _ = make_builder(fa3)
    builder.vllm_config.compilation_config.cudagraph_capture_sizes = [8]
    metadata = builder.build_for_cudagraph_capture(common())
    assert metadata.max_query_len == 8
    assert builder.scheduler_buffers


def test_mtp_preserves_full_graph_capability(fa3):
    cfg = config()
    assert fa3.AscendFlashAttentionMetadataBuilder.get_cudagraph_support(cfg, None) == "always"
    cfg.speculative_config = object()
    assert fa3.AscendFlashAttentionMetadataBuilder.get_cudagraph_support(cfg, None) == "always"


def test_replay_refreshes_captured_buffers_when_source_pointers_change(fa3):
    builder, _, _ = make_builder(fa3)
    captured = builder.build_for_cudagraph_capture(common())
    actual = common()
    actual.query_start_loc[1] = 2
    actual.seq_lens[0] = 197
    actual.block_table_tensor[0, 0] = 16
    replay = builder.build(0, actual, full_graph_mode=True)
    for name in ("query_start_loc", "seq_lens", "block_tables"):
        assert getattr(captured, name).data_ptr() == getattr(replay, name).data_ptr()
    assert captured.query_start_loc[1] == 2
    assert captured.seq_lens[0] == 197
    assert captured.block_tables[0, 0] == 16
    assert (
        next(iter(captured.scheduler_metadata.values())).data_ptr()
        == next(iter(replay.scheduler_metadata.values())).data_ptr()
    )


def test_device_fa3_does_not_enter_fia_updatable_task_graph_path(fa3):
    path = ROOT / "vllm_ascend/utils.py"
    tree = ast.parse(path.read_text())
    method = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "use_updatable_graph"
    )
    # Import isolation only: preserve the actual decision expression.
    method.body = [node for node in method.body if not isinstance(node, ast.ImportFrom)]
    base = fa3.AscendFlashAttentionBackend.__bases__[0]
    namespace = {"AscendAttentionBackend": base}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)  # noqa: S102
    assert namespace["use_updatable_graph"](base)
    assert not namespace["use_updatable_graph"](fa3.AscendFlashAttentionBackend)
    assert not namespace["use_updatable_graph"](None)


def test_mtp_runtime_propagates_full_descriptor_after_capture_size_rounding(fa3):
    """Exercise the actual MTP method/factory; never manually set builder FULL."""
    builder, _, _ = make_builder(fa3)
    builder.vllm_config.compilation_config.cudagraph_capture_sizes = [8]
    capture_common = common()
    capture_common.num_input_tokens = capture_common.num_actual_tokens = 9
    capture_common.slot_mapping = torch.arange(9)
    capture_common.query_start_loc = torch.tensor([0, 3, 6, 9], dtype=torch.int32)
    captured = builder.build_for_cudagraph_capture(capture_common)
    actual = common()
    actual.num_input_tokens = actual.num_actual_tokens = 9
    actual.slot_mapping = torch.arange(9)
    actual.query_start_loc = torch.tensor([0, 3, 6, 6], dtype=torch.int32)
    actual.seq_lens[0] = 197
    actual.block_table_tensor[0, 0] = 16
    received = {}

    def generic_metadata(**kwargs):
        received.update(kwargs)
        return {
            "layer": builder.build(
                0,
                actual,
                full_graph_mode=kwargs.get("full_graph_mode", False),
                num_actual_reqs=kwargs.get("num_actual_reqs"),
            )
        }

    module = SimpleNamespace(build_attn_metadata=generic_metadata)
    namespace = {
        "_BUILD_ATTN_METADATA_MODULE": module,
        "contextmanager": contextmanager,
    }
    load_definitions(ATTN_UTILS, namespace, {"build_attn_metadata_factory"})

    class Upstream:
        def _build_attn_metadata(self, *args, **kwargs):
            return module.build_attn_metadata()

    path = ROOT / "vllm_ascend/worker/v2/spec_decode/multi_module_mtp/speculator.py"
    tree = ast.parse(path.read_text())
    cls = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AscendMultiModuleMTPSpeculator"
    )
    method = next(
        node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_build_attn_metadata"
    )
    isolated = ast.ClassDef(
        name=cls.name,
        bases=[ast.Name(id="Upstream", ctx=ast.Load())],
        keywords=[],
        body=[method],
        decorator_list=[],
    )
    base = fa3.AscendFlashAttentionBackend.__bases__[0]
    namespace.update(
        Upstream=Upstream,
        torch=torch,
        CUDAGraphMode=SimpleNamespace(NONE="none", FULL="full"),
        AscendAttentionBackend=base,
    )
    exec(compile(ast.fix_missing_locations(ast.Module(body=[isolated], type_ignores=[])), str(path), "exec"), namespace)  # noqa: S102
    runner = namespace[cls.name]()
    runner.attn_backend = fa3.AscendFlashAttentionBackend
    runner.attn_groups = [[SimpleNamespace(backend=runner.attn_backend)]]
    runner.input_batch = SimpleNamespace(is_prefilling_np=np.zeros(2, dtype=np.bool_))
    runner.input_buffers = SimpleNamespace(positions=torch.arange(9), seq_lens_cpu=actual.seq_lens)
    runner.draft_attn_state = "spec"
    runner.draft_vllm_config = builder.vllm_config
    desc = SimpleNamespace(cg_mode="full", num_tokens=9, num_reqs=3, uniform_token_count=3)
    replay = runner._build_attn_metadata(2, desc, np.array([0, 3, 6]), None, 0)["layer"]
    assert received["full_graph_mode"] is True
    assert received["num_actual_reqs"] == 2
    for name in ("query_start_loc", "seq_lens", "block_tables"):
        assert getattr(captured, name).data_ptr() == getattr(replay, name).data_ptr()
    assert captured.query_start_loc.tolist() == [0, 3, 6, 6, 6, 6]
    assert captured.seq_lens.tolist() == [197, 259, 0, 0, 0]
    assert captured.block_tables[0, 0] == 16
    assert (
        next(iter(captured.scheduler_metadata.values())).data_ptr()
        == next(iter(replay.scheduler_metadata.values())).data_ptr()
    )


def test_target_and_draft_pools_do_not_share_capacity_or_buffers(fa3):
    target, _, _ = make_builder(fa3, num_blocks=19)
    draft, _, _ = make_builder(fa3, num_blocks=7)
    source = common()
    first = target.build_for_cudagraph_capture(source)
    second = draft.build_for_cudagraph_capture(source)
    assert fa3.calls[0]["num_blocks"] == 19
    assert fa3.calls[1]["num_blocks"] == 7
    assert first.query_start_loc.data_ptr() != second.query_start_loc.data_ptr()


def test_rebind_discards_profiling_cache_state(fa3):
    builder, _, context = make_builder(fa3)
    builder.build_for_cudagraph_capture(common())
    context["layer"].kv_cache = tuple(t[:7].clone() for t in context["layer"].kv_cache)
    builder.bind_fa3_kv_cache(context)
    assert not builder.scheduler_buffers
    assert not builder.graph_buffers
    builder.build(0, common())
    assert fa3.calls[-1]["num_blocks"] == 7


def test_no_hardcoded_window_for_full_attention(fa3):
    builder, _, _ = make_builder(fa3, sliding_window=None)
    builder.build(0, common())
    assert fa3.calls[-1]["window_size"] == (-1, -1)


def test_old_scheduler_abi_is_rejected(fa3):
    fa3.validate_fa3_scheduler_api.__globals__["get_scheduler_metadata"] = lambda **kwargs: None
    with pytest.raises(RuntimeError, match="accuracy-fix"):
        fa3.validate_fa3_scheduler_api()


def test_old_native_stride_abi_is_rejected(fa3):
    fa3.flash_attn_with_kvcache.block_stride_abi_version = 0
    with pytest.raises(RuntimeError, match="block-stride ABI"):
        fa3.validate_fa3_scheduler_api()


def test_unbound_pool_and_missing_real_request_rows_are_rejected(fa3):
    builder, _, _ = make_builder(fa3)
    source = common()
    source.seq_lens = source.seq_lens[:1]
    with pytest.raises(ValueError, match="every real request"):
        builder.build(0, source)
    builder.scheduler_specs.clear()
    with pytest.raises(RuntimeError, match="bound"):
        builder.build(0, common())


def test_strided_kv_pool_is_rejected_not_copied_each_forward(fa3):
    builder, _, context = make_builder(fa3)
    k, v = context["layer"].kv_cache
    context["layer"].kv_cache = (k.transpose(1, 2), v)
    with pytest.raises(ValueError, match="dense BNHD"):
        builder.bind_fa3_kv_cache(context)


def test_forward_calls_operator_with_bound_metadata(fa3):
    builder, impl, _ = make_builder(fa3)
    metadata = builder.build(0, common())
    query = torch.ones((8, 8, 128), dtype=torch.bfloat16)
    output = torch.empty_like(query)
    assert impl.forward_impl(query, None, None, None, metadata, output) is output
    torch.testing.assert_close(output, query + 1)
    kwargs = fa3.flash_attn_with_kvcache.call_args.kwargs
    assert kwargs["window_size"] == (127, 0)
    assert kwargs["scheduler_metadata"] is next(iter(metadata.scheduler_metadata.values()))
    fa3.record_attention_compute_start.assert_called_once()


def test_mrv2_binding_visits_every_target_and_draft_builder():
    namespace = load_definitions(ATTN_UTILS, {}, {"bind_fa3_kv_cache"})
    target, draft = Mock(), Mock()
    context = {"cache": object()}
    namespace["bind_fa3_kv_cache"](
        [
            [SimpleNamespace(metadata_builders=[target, SimpleNamespace()])],
            [SimpleNamespace(metadata_builders=[draft])],
        ],
        context,
    )
    target.bind_fa3_kv_cache.assert_called_once_with(context)
    draft.bind_fa3_kv_cache.assert_called_once_with(context)


@dataclass
class FullSpec:
    block_size: int = 128
    num_kv_heads: int = 2
    head_size: int = 128
    head_size_v: int = 128
    dtype: object = torch.bfloat16
    page_size_bytes: int = 262144


class WindowSpec(FullSpec):
    pass


class RingSpec(FullSpec):
    pass


def test_fa3_mtp_does_not_read_or_rewrite_exact_cpu_lengths():
    path = ROOT / "vllm_ascend/worker/v2/spec_decode/autoregressive/speculator.py"
    tree = ast.parse(path.read_text())
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and any(
            isinstance(child, ast.FunctionDef) and child.name == "_update_decode_attn_metadata" for child in node.body
        )
    )
    method = next(
        node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_update_decode_attn_metadata"
    )
    namespace = {"Any": object}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)  # noqa: S102
    runner = SimpleNamespace(attn_architecture="GQA")
    metadata = {"layer": SimpleNamespace(uses_device_fa3=True, seq_lens_cpu=None)}
    namespace["_update_decode_attn_metadata"](runner, metadata, 1)


@pytest.mark.parametrize(
    "runner_v2,pcp,dcp,transfer", [(False, 1, 1, None), (True, 2, 1, None), (True, 1, 2, None), (True, 1, 1, object())]
)
def test_unsupported_runner_and_parallel_paths_fail_explicitly(fa3, runner_v2, pcp, dcp, transfer):
    cfg = config()
    cfg.use_v2_model_runner = runner_v2
    cfg.parallel_config.prefill_context_parallel_size = pcp
    cfg.parallel_config.decode_context_parallel_size = dcp
    cfg.kv_transfer_config = transfer
    with pytest.raises(ValueError):
        fa3.AscendFlashAttentionImpl(8, 128, 0.125, 2, vllm_config=cfg)
