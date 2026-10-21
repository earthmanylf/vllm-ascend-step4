"""Step5 DSA contracts: native-op adapters, model state, aclgraph step control.

Merged from 3 source files per design spec section 5.2. AST
isolation is preserved: class definitions are compiled from source so these
run on CPU-only hosts as well as in the NPU unit-test suite.

Path note: the source files used ``Path(__file__).resolve().parents[N]`` to
reach the repo root, and N differs with directory depth. Every such
expression is rewritten to REPO_ROOT below, so this file is depth
independent.
"""

from __future__ import annotations

from pathlib import Path

# Repo root of vllm-ascend-private.
REPO_ROOT = Path(__file__).resolve().parents[4]

import ast
import types
from pathlib import Path
import pytest
import torch
import unittest
from dataclasses import dataclass
from unittest.mock import patch
import importlib.util
from contextlib import nullcontext
from dataclasses import dataclass, replace
from types import SimpleNamespace

MODEL_DIR = REPO_ROOT / "vllm_ascend/models/step5"
ROOT = REPO_ROOT
SPEC = ROOT / "vllm_ascend/worker/v2/spec_decode/autoregressive/speculator.py"
GRAPH = ROOT / "vllm_ascend/worker/v2/spec_decode/autoregressive/aclgraph.py"


# ========================================================================
# from tests/ut/models/step5/test_native_dsa_contract.py
# ========================================================================

def load_functions(filename, names, namespace):
    tree = ast.parse((MODEL_DIR / filename).read_text())
    body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            body.append(node)
        elif isinstance(node, ast.ClassDef):
            body.extend(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name in names)
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])),
            filename,
            "exec",
        ),
        namespace,
    )
    return types.SimpleNamespace(**{name: namespace[name] for name in names})


@pytest.mark.parametrize("region_len", [0, 2])
def test_indexer_uses_current_tuple_api_and_preserves_fp8_summary_storage(region_len):
    calls = []

    def native(query, weights, **kwargs):
        calls.append((query, weights, kwargs))
        ids = kwargs["selected_region_ids"]
        ids.fill_(-1)
        if kwargs["max_region_len"]:
            ids[:, :, 0] = 0
        return torch.zeros(1, 2, kwargs["max_region_len"]), ids

    adapter = load_functions(
        "indexer_region_score.py",
        {"indexer_score_topk"},
        {"torch": torch, "indexer_region_score": native},
    )
    query = torch.ones(2, 1, 4, 256, dtype=torch.bfloat16)
    weights = torch.full((2, 1, 4), -0.5)
    storage = torch.zeros(3, 16, 1, 256, dtype=torch.uint8)
    mean = storage.view(torch.float8_e4m3fn)
    qsl, seq = (
        torch.tensor([0, 2], dtype=torch.int32),
        torch.tensor([18], dtype=torch.int32),
    )
    scores, ids = adapter.indexer_score_topk(
        query,
        weights,
        mean,
        torch.tensor([[2]], dtype=torch.int32),
        qsl,
        seq,
        max_region_len=region_len,
        topk=512,
    )
    assert scores.shape == (2, region_len) and ids.shape == (2, 512)
    assert calls[0][0].dtype == torch.float8_e4m3fn
    kwargs = calls[0][2]
    assert kwargs["paged_key_summary"].data_ptr() == storage.data_ptr()
    assert kwargs["cu_seq_lens"] is qsl and kwargs["actual_kv_lengths"] is seq
    assert (
        kwargs["mask_mode"],
        kwargs["head_scale"],
        kwargs["cache_mode"],
        kwargs["sort_mode"],
    ) == (3, 1.0, 1, 1)
    assert ids.data_ptr() == kwargs["selected_region_ids"].data_ptr()
    with pytest.raises(ValueError, match="requires G"):
        adapter.indexer_score_topk(
            torch.ones(2, 2, 2, 256),
            weights,
            mean,
            None,
            qsl,
            seq,
            max_region_len=region_len,
            topk=512,
        )


def test_native_metadata_counts_cover_first_region_and_topk_boundary():
    calls = []

    def native(*args, **kwargs):
        calls.append((args, kwargs))
        return torch.tensor([17], dtype=torch.int32)

    ns = {
        "torch": torch,
        "generic_block_sparse_attention_metadata": native,
        "_SPARSE_PATTERN_PER_GROUP_PER_TOKEN": 4,
        "_MASK_CAUSAL": 1,
        "_QUANT_NONE": 0,
        "_SOFTMAX_PRECISION_FP32": 1,
    }
    adapter = load_functions(
        "generic_block_sparse_attention.py",
        {"gbsa_expected_counts", "gbsa_sparse_attention_metadata"},
        ns,
    )
    positions = torch.tensor([0, 7, 8, 4095, 4096, 4104])
    counts = adapter.gbsa_expected_counts(positions, groups=1, region_size=8, topk=512)
    assert counts.tolist() == [[1, 1, 2, 512, 513, 513]]
    qsl = torch.tensor([0, 6], dtype=torch.int64)
    seq = torch.tensor([4105], dtype=torch.int32)
    adapter.gbsa_sparse_attention_metadata(
        counts,
        num_heads=8,
        num_kv_groups=1,
        head_dim=128,
        region_size=8,
        topk=512,
        cu_seqlens_q=qsl,
        seqused_kv=seq,
        max_query_len=6,
        max_pages=33,
        page_size=128,
    )
    args, kwargs = calls[0]
    assert args[0].shape == (1, 6, 513) and args[1] is counts
    assert args[2:] == (8, 1, 128, [1, 8])
    assert kwargs["cu_seqlens_q"].data_ptr() == qsl.data_ptr()
    assert kwargs["seqused_kv"].data_ptr() == seq.data_ptr()
    assert (
        kwargs["layout_q"],
        kwargs["layout_kv"],
        kwargs["mask_mode"],
        kwargs["quant_mode"],
    ) == ("TND", "PA_BBND", 1, 0)
    assert kwargs["is_consistent_topk"] is False


@pytest.mark.parametrize("query_len,num_reqs", [(4, 2), (3, 2), (1, 5)])
def test_captured_speculative_shape_refreshes_schedule_at_the_captured_address(query_len, num_reqs):
    """Actual capture descriptors can differ from raw graph token buckets."""
    functions = load_functions(
        "dsa_attention.py",
        {"build_for_cudagraph_capture", "_prepare_gbsa_metadata", "_build_gbsa_metadata"},
        {
            "torch": torch,
            "gbsa_expected_counts": lambda positions, **kwargs: positions.unsqueeze(0),
            "gbsa_sparse_attention_metadata": lambda counts, **kwargs: kwargs["seqused_kv"].clone(),
        },
    )
    raw_capture_sizes = {1, 5, 16}
    actual_tokens = query_len * num_reqs
    if query_len > 1:
        assert actual_tokens not in raw_capture_sizes
    builder = types.SimpleNamespace(
        capture_sizes=set(),
        gbsa_meta_buffers={},
        gbsa_num_heads=8,
        gbsa_num_kv_groups=1,
        gbsa_head_size=128,
        gbsa_region_size=8,
        gbsa_topk=512,
        gbsa_block_size=128,
        query_start_loc_int64_buffer=torch.empty(num_reqs + 1, dtype=torch.int64),
    )
    for name in ("_prepare_gbsa_metadata", "_build_gbsa_metadata"):
        setattr(builder, name, types.MethodType(getattr(functions, name), builder))

    def build(*, common_prefix_len, common_attn_metadata):
        assert common_prefix_len == 0
        builder._prepare_gbsa_metadata(common_attn_metadata)
        return common_attn_metadata

    builder.build = build
    decode = types.SimpleNamespace(
        seq_lens=torch.arange(20, 20 + num_reqs, dtype=torch.int32),
        query_len=query_len,
        block_table=torch.zeros(num_reqs, 2, dtype=torch.int32),
    )
    metadata = types.SimpleNamespace(
        query_start_loc=torch.arange(0, actual_tokens + 1, query_len, dtype=torch.int32),
        num_actual_tokens=actual_tokens,
        num_decodes=num_reqs,
        num_prefills=0,
        decode=decode,
        prefill=None,
    )
    # An eager/profile shape must not accumulate persistent graph schedules.
    builder._prepare_gbsa_metadata(metadata)
    assert not builder.gbsa_meta_buffers
    captured = functions.build_for_cudagraph_capture(builder, metadata)
    assert captured is metadata and builder.capture_sizes == {actual_tokens}
    schedule = decode.gbsa_metadata
    schedule_address = schedule.data_ptr()
    query_address = decode.cu_seqlens_q.data_ptr()
    before = schedule.clone()

    # Warmup/capture can rebuild metadata before runtime starts.
    functions.build_for_cudagraph_capture(builder, metadata)
    assert decode.gbsa_metadata.data_ptr() == schedule_address
    decode.seq_lens.add_(11)
    builder._prepare_gbsa_metadata(metadata)
    assert decode.gbsa_metadata.data_ptr() == schedule_address
    assert decode.cu_seqlens_q.data_ptr() == query_address
    assert torch.equal(schedule, before + 11)
    assert len(builder.gbsa_meta_buffers) == 1


def test_mixed_metadata_preserves_request_boundaries():
    fn = load_functions("dsa_attention.py", {"_prepare_gbsa_metadata"}, {"torch": torch})._prepare_gbsa_metadata
    records = []
    builder = types.SimpleNamespace(
        query_start_loc_int64_buffer=torch.empty(9, dtype=torch.int64),
        capture_sizes=set(),
        _build_gbsa_metadata=lambda *args, **kwargs: records.append((args, kwargs)) or torch.tensor([7]),
    )
    d = types.SimpleNamespace(
        seq_lens=torch.tensor([18, 35], dtype=torch.int32),
        query_len=2,
        block_table=torch.tensor([[3, 8], [6, 1]], dtype=torch.int32),
    )
    p = types.SimpleNamespace(
        seq_lens=torch.tensor([12], dtype=torch.int32),
        context_lens=torch.tensor([9]),
        query_start_loc=torch.tensor([0, 3], dtype=torch.int32),
        num_query_tokens=3,
        max_query_len=3,
        block_table=torch.tensor([[5]], dtype=torch.int32),
    )
    md = types.SimpleNamespace(
        query_start_loc=torch.tensor([0, 2, 4, 7, 7], dtype=torch.int32),
        num_actual_tokens=7,
        decode=d,
        prefill=p,
        num_decodes=2,
        num_prefills=1,
    )
    fn(builder, md)
    assert records[0][0][0].tolist() == [16, 17, 33, 34]
    assert records[0][0][1].tolist() == [0, 2, 4]
    assert records[0][0][2] is d.seq_lens and records[0][0][4] == 2
    assert records[1][0][0].tolist() == [9, 10, 11]
    assert records[1][0][1].tolist() == [0, 3]
    assert d.cu_seqlens_q.dtype == p.cu_seqlens_q.dtype == torch.int64
    assert d.cu_seqlens_q.data_ptr() == builder.query_start_loc_int64_buffer.data_ptr()
    assert md.query_start_loc.tolist() == [0, 2, 4, 7, 7]


@pytest.mark.parametrize("path", ["decode", "verify", "prefill"])
def test_consumer_uses_paged_kv_and_request_varlen_without_flattening(path):
    records = []

    def indexer(*args, **kwargs):
        records.append(("indexer", args, kwargs))
        tokens, groups = args[0].shape[:2]
        return torch.zeros(groups * tokens, kwargs["max_region_len"]), torch.full(
            (groups * tokens, 512), -1, dtype=torch.int32
        )

    def attention(*args, **kwargs):
        records.append(("attention", args, kwargs))
        return torch.full_like(args[0], 2)

    name = "_forward_sparse_gqa_" + path
    fn = getattr(
        load_functions(
            "dsa_attention.py",
            {name},
            {
                "torch": torch,
                "indexer_score_topk": indexer,
                "gbsa_sparse_attention": attention,
            },
        ),
        name,
    )
    impl = types.SimpleNamespace(
        num_speculative_tokens=1,
        num_kv_groups=1,
        _max_regions=16,
        topk_regions=512,
        region_size=8,
        scale=0.125,
    )
    qlen = 2 if path == "verify" else 1
    d = types.SimpleNamespace(
        query_len=qlen,
        seq_lens=torch.tensor([18], dtype=torch.int32),
        block_table=torch.tensor([[2]], dtype=torch.int32),
        cu_seqlens_q=torch.tensor([0, qlen]),
        gbsa_metadata=torch.tensor([23]),
    )
    p = types.SimpleNamespace(
        num_query_tokens=3,
        seq_lens_host=[12],
        context_lens_host=[9],
        block_table=d.block_table,
        query_start_loc=torch.tensor([0, 3], dtype=torch.int32),
        cu_seqlens_q=torch.tensor([0, 3]),
        seq_lens=torch.tensor([12], dtype=torch.int32),
        gbsa_metadata=torch.tensor([29]),
    )
    md = types.SimpleNamespace(
        num_decodes=1,
        num_prefills=1,
        decode=d,
        prefill=p,
        query_start_loc=torch.tensor([0, qlen], dtype=torch.int32),
    )
    total = 3 if path == "prefill" else qlen
    cache = types.SimpleNamespace(mean=torch.zeros(3, 16, 256, dtype=torch.uint8))
    key = torch.empty(3, 128, 1, 128)
    value = torch.empty_like(key)
    output = torch.zeros(5, 8, 128)
    kwargs = dict(
        summary_cache=cache,
        md=md,
        query=torch.empty(total, 8, 128),
        key_cache=key,
        value_cache=value,
        index_q=torch.ones(total, 1, 4, 256),
        weights=torch.ones(total, 1, 4),
        output=output,
    )
    if path == "verify":
        kwargs["positions"] = torch.tensor([16, 17])
    elif path == "prefill":
        kwargs.update(positions=torch.tensor([7, 8, 9, 10, 11]), token_offset=2)
    fn(impl, **kwargs)
    _, inputs, _ = records[0]
    _, attn_inputs, _ = records[1]
    assert attn_inputs[1] is key and attn_inputs[2] is value
    assert inputs[3].shape[0] == 1 and inputs[4].tolist() == [0, total]
    assert inputs[5].tolist() == ([12] if path == "prefill" else [18])
    assert attn_inputs[6].tolist() == [0, total]
    assert attn_inputs[6].dtype == torch.int64
    assert attn_inputs[4].tolist() == ({"decode": [17], "verify": [16, 17], "prefill": [9, 10, 11]}[path])
    if path == "prefill":
        assert not output[:2].any() and (output[2:] == 2).all()
    else:
        assert (output[:total] == 2).all() and not output[total:].any()


def test_native_attention_passes_matching_metadata_and_paged_kv():
    calls = []
    idx = torch.tensor([[[0, 2]]], dtype=torch.int32)
    counts = torch.tensor([[2]], dtype=torch.int32)

    def native(*args, **kwargs):
        calls.append((args, kwargs))
        return torch.full_like(args[0], 3), None

    ns = {
        "torch": torch,
        "generic_block_sparse_attention": native,
        "_append_current_region": lambda *args, **kwargs: (idx, counts),
        "_SPARSE_PATTERN_PER_GROUP_PER_TOKEN": 4,
        "_MASK_CAUSAL": 1,
        "_QUANT_NONE": 0,
        "_SOFTMAX_PRECISION_FP32": 1,
    }
    fn = load_functions("generic_block_sparse_attention.py", {"gbsa_sparse_attention"}, ns).gbsa_sparse_attention
    q = torch.empty(1, 8, 128, dtype=torch.bfloat16)
    key = torch.empty(3, 128, 1, 128, dtype=torch.bfloat16)
    value = torch.empty_like(key)
    qsl = torch.tensor([0, 1], dtype=torch.int64)
    seq = torch.tensor([18], dtype=torch.int32)
    schedule = torch.tensor([7], dtype=torch.int32)
    table = torch.tensor([[2]], dtype=torch.int32)
    out = fn(
        q,
        key,
        value,
        torch.tensor([[0]], dtype=torch.int32),
        torch.tensor([17]),
        table,
        qsl,
        seq,
        schedule,
        num_kv_groups=1,
        region_size=8,
        softmax_scale=0.125,
    )
    args, kwargs = calls[0]
    assert args[1] is key and args[2] is value and args[3] is idx and args[4] is counts
    assert args[5] == [1, 8] and kwargs["metadata"] is schedule and kwargs["block_table"] is table
    assert kwargs["cu_seqlens_q"] is qsl and kwargs["seqused_kv"] is seq
    assert kwargs["layout_sparse_pattern"] == 4 and kwargs["mask_mode"] == 1
    assert kwargs["quant_mode"] == 0 and kwargs["softmax_precision"] == 1
    assert kwargs["return_softmax_lse"] is False and kwargs["is_consistent_topk"] is False
    assert out.dtype == q.dtype and (out == 3).all()


# ========================================================================
# from tests/ut/models/step5/test_step5_model_state_contract.py
# ========================================================================

class Scalar(int):
    def item(self):
        return int(self)


class Buffer:
    """A storage-sharing vector: slices retain the same allocation."""

    device = "cpu"

    def __init__(self, values, *, storage=None, indices=None):
        self.storage = list(values) if storage is None else storage
        self.indices = list(range(len(self.storage))) if indices is None else indices

    def __getitem__(self, index):
        if isinstance(index, slice):
            return Buffer([], storage=self.storage, indices=self.indices[index])
        return Scalar(self.storage[self.indices[index]])

    def __sub__(self, other):
        values = other.tolist() if isinstance(other, Buffer) else [other] * self.numel()
        return Buffer([left - right for left, right in zip(self.tolist(), values)])

    def numel(self):
        return len(self.indices)

    def fill_(self, value):
        for index in self.indices:
            self.storage[index] = value
        return self

    def copy_(self, source, **kwargs):
        values = source.tolist()
        assert len(values) == self.numel()
        for index, value in zip(self.indices, values):
            self.storage[index] = value
        return self

    def tolist(self):
        return [self.storage[index] for index in self.indices]

    def max(self):
        return Scalar(max(self.tolist()))

    def sum(self):
        return Scalar(sum(self.tolist()))

    def detach(self):
        return self

    def cpu(self):
        return self

    def to(self, **kwargs):
        return self


class TestStep5RequestMetadata(unittest.TestCase):
    """Metadata carries no per-request journal state any more.

    The TailRing design addresses summary updates through
    ``ring.metadata().base / .slot_mapping``, computed per step, so neither
    the builder nor the metadata dataclasses carry a persistent request
    index. These tests pin that contract: the request-state fields are gone,
    and the decode/prefill views still split by request rather than by token.
    """

    def setUp(self):
        tree = ast.parse((MODEL_DIR / "dsa_attention.py").read_text())
        wanted = {"AscendStep5DSAPrefillMetadata", "AscendStep5DSADecodeMetadata", "AscendStep5DSAMetadata"}
        body = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in wanted]
        builder = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "AscendStep5DSAMetadataBuilder"
        )
        builder.bases = []
        builder.body = [node for node in builder.body if isinstance(node, ast.FunctionDef) and node.name == "build"]
        namespace = {
            "__name__": __name__,
            "dataclass": dataclass,
            "AttentionMetadata": object,
            "torch": types.SimpleNamespace(int32="int32"),
            "split_decodes_and_prefills": lambda *args, **kwargs: (2, 2, 8, 3),
            "_active_prefill_num_reqs": lambda *args: 1,
            "_active_decode_num_reqs": lambda count, tokens, width: min(count, tokens // width),
        }
        body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *body, builder]
        exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), "metadata", "exec"), namespace)
        self.builder = namespace[builder.name]()
        self.builder.reorder_batch_threshold = 4
        self.builder._prepare_gbsa_metadata = lambda metadata: None
        self.builder.context_len_buffer = Buffer([0] * 5)
        self.common = types.SimpleNamespace(
            query_start_loc=Buffer([0, 4, 8, 11, 11]),
            query_start_loc_cpu=Buffer([0, 4, 8, 11, 11]),
            seq_lens=Buffer([16, 24, 7, 0]),
            block_table_tensor=Buffer([10, 11, 12, -1]),
            num_reqs=4,
            num_actual_tokens=11,
            slot_mapping=Buffer(range(11)),
        )

    def test_no_request_state_field_survives(self):
        # The journal is gone, so none of the metadata dataclasses may still
        # expose a step5_request_state field.
        import dataclasses as _dc

        md = self.builder.build(0, self.common)
        for holder, label in ((md, "metadata"), (md.decode, "decode"), (md.prefill, "prefill")):
            names = {f.name for f in _dc.fields(holder)}
            self.assertNotIn("step5_request_state", names, f"{label} must not carry journal state")
        self.assertFalse(hasattr(self.builder, "step5_request_state"), "builder must not carry journal state")

    def test_decode_and_prefill_take_request_views_not_token_views(self):
        md = self.builder.build(0, self.common)
        # Views are still split by request, not by token row.
        self.assertEqual(md.decode.query_len, 4)
        self.assertEqual(md.prefill.query_start_loc.tolist(), [0, 3])
        self.assertEqual(md.prefill.context_lens.tolist(), [4])
        # The full-batch views keep their own shapes.
        self.assertEqual(md.query_start_loc.tolist(), [0, 4, 8, 11, 11])
        self.assertEqual(md.slot_mapping.tolist(), list(range(11)))


if __name__ == "__main__":
    unittest.main()


# ========================================================================
# from tests/ut/worker/v2/test_ar_graph_step_contract.py
# ========================================================================

def methods(path, class_name, names, base, namespace):
    tree = ast.parse(path.read_text())
    original = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    body = [n for n in original.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in body} == set(names)
    cls = ast.ClassDef(
        name=class_name,
        bases=[ast.Name(id="TestBase", ctx=ast.Load())],
        keywords=[],
        body=body,
        decorator_list=[],
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            cls,
        ],
        type_ignores=[],
    )
    namespace = dict(namespace, TestBase=base)
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)  # noqa: S102
    return namespace[class_name]


class Mode:
    NONE = 0
    FULL = 1
    PIECEWISE = 2


@dataclass(frozen=True)
class Desc:
    cg_mode: int
    num_tokens: int
    num_reqs: int
    max_query_len: int | None = None
    uniform_token_count: int | None = 1


class Vector:
    def __init__(self):
        self.value = 0
        self.zeroed = False

    def __getitem__(self, key):
        return self

    def zero_(self):
        self.zeroed = True

    def fill_(self, value):
        self.value = value


class PerStepContract(unittest.TestCase):
    def build_spec(self):
        # Execute the fixed upstream loop that the new Ascend wrapper delegates
        # to, instead of mocking away the K-1 replay behavior being tested.
        current = ROOT.parents[1] / "current/vllm/vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py"
        if not current.exists():
            spec = importlib.util.find_spec("vllm")
            if spec is not None and spec.submodule_search_locations:
                current = (
                    Path(next(iter(spec.submodule_search_locations)))
                    / "v1/worker/gpu/spec_decode/autoregressive/speculator.py"
                )
        if not current.exists():
            self.skipTest("A matching vLLM source installation is required")
        upstream = methods(
            current,
            "AutoRegressiveSpeculator",
            ["_multi_step_decode"],
            object,
            {
                "CUDAGraphMode": Mode,
                "build_slot_mappings_by_layer": lambda slots, config: slots,
            },
        )
        parent = type(
            "Parent",
            (upstream,),
            {"_configure_fused_multi_step_decode": lambda self: None},
        )
        cls = methods(
            SPEC,
            "AscendAutoRegressiveSpeculator",
            [
                "_multi_step_decode",
                "build_draft_attn_metadatas",
                "_configure_fused_multi_step_decode",
            ],
            parent,
            {
                "CUDAGraphMode": Mode,
                "replace": replace,
                "logger": SimpleNamespace(info=lambda *args: None),
            },
        )
        s = cls()
        s.input_buffers = SimpleNamespace(positions=Vector(), query_start_loc=Vector())
        s.idx_mapping = Vector()
        s.current_draft_step = Vector()
        s.num_speculative_steps = 3
        s.advance_draft_positions = True
        s.kv_cache_config = object()
        s._draft_decode_attn_metadata = None
        s.built, s.replayed, s.slot_calls = [], [], []

        def slots(*args):
            s.slot_calls.append(args)
            return object()

        s.block_tables = SimpleNamespace(compute_slot_mappings=slots)

        def metadata(**kwargs):
            step = kwargs["step"]
            md = {"draft": SimpleNamespace(length=10 + step, step=step)}
            s._draft_decode_attn_metadata = md
            s.built.append(md)
            return md

        s._build_uniform_attn_metadata = metadata

        def replay(desc):
            md = s.build_draft_attn_metadatas(desc.num_reqs, desc.num_tokens, False)[0]
            s.replayed.append((s.current_draft_step.value, md))

        s.decode_cudagraph_manager = SimpleNamespace(run_fullgraph=replay)
        s._generate_draft = lambda *args, **kwargs: None
        return s

    def test_two_steps_replay_distinct_current_metadata(self):
        s = self.build_spec()
        s._multi_step_decode(1, False, Desc(Mode.FULL, 1, 1), None, object())
        self.assertEqual([step for step, _ in s.replayed], [1, 2])
        self.assertEqual([md["draft"].length for _, md in s.replayed], [11, 12])
        self.assertIs(s.replayed[0][1], s.built[0])
        self.assertIs(s.replayed[1][1], s.built[1])
        self.assertEqual(len(s.slot_calls), 2)
        self.assertIsNone(s._draft_decode_attn_metadata)

    def test_one_additional_draft(self):
        s = self.build_spec()
        s.num_speculative_steps = 2
        s._multi_step_decode(1, False, Desc(Mode.FULL, 1, 1), None, object())
        self.assertEqual([step for step, _ in s.replayed], [1])

    def test_failure_clears_metadata(self):
        s = self.build_spec()

        def fail(desc):
            raise ValueError("device failure")

        s.decode_cudagraph_manager.run_fullgraph = fail
        with self.assertRaisesRegex(ValueError, "device failure"):
            s._multi_step_decode(1, False, Desc(Mode.FULL, 1, 1), None, object())
        self.assertIsNone(s._draft_decode_attn_metadata)

    def test_skip_attention_never_replays_a_live_graph(self):
        s = self.build_spec()
        s._multi_step_decode(1, True, Desc(Mode.FULL, 1, 1), None, object())
        self.assertEqual(s.replayed, [])
        self.assertEqual(s.slot_calls, [])

    def test_no_metadata_cannot_fall_back_to_target_state(self):
        s = self.build_spec()
        s.model_state = SimpleNamespace(attn_metadata={"target": object()})
        with self.assertRaisesRegex(AssertionError, "not prepared"):
            s.build_draft_attn_metadatas(1, 1, False)


class CaptureContract(unittest.TestCase):
    def test_actual_capture_signature_and_query_bounds(self):
        for phase in (False, True):
            for maximum, width in ((None, 1), (8, None)):
                with self.subTest(phase=phase, maximum=maximum):
                    prepared, calls = [], []
                    desc = Desc(Mode.FULL, 4, 2, maximum, width)

                    def prepare(*args, prepared=prepared, **kwargs):
                        result = (object(), object())
                        prepared.append((args, kwargs, result))
                        return result

                    class Base:
                        @staticmethod
                        def capture(manager, create, progress_bar_desc, desc=desc):
                            create(desc, True)(Mode.NONE)
                            create(desc, False)(Mode.NONE)

                    cls = methods(
                        GRAPH,
                        "AutoRegressiveAclGraphManager",
                        ["capture"],
                        object,
                        {
                            "communicator_switch": nullcontext,
                            "model_capture_wrapper": lambda *a: nullcontext(),
                            "CudaGraphManager": Base,
                            "CUDAGraphMode": Mode,
                            "prepare_inputs_to_capture": prepare,
                            "torch": SimpleNamespace(full=lambda *a, **k: (4, 4), int32=object()),
                        },
                    )
                    g = cls()
                    g.speculator = object()
                    g.is_draft_model_prefill = phase
                    g.max_num_reqs = 3
                    g.dp_size = 2
                    inputs = [object() for _ in range(5)]
                    g.capture(lambda *a, calls=calls: calls.append(a), *inputs)
                    self.assertEqual(len(prepared), 2)
                    self.assertEqual(len(calls), 2)
                    for (_, kw, result), call in zip(prepared, calls):
                        self.assertEqual(
                            kw,
                            {"full_cudagraph": True, "max_query_len": maximum or width},
                        )
                        self.assertEqual(call, (2, 4, *result, (4, 4), Mode.NONE))
                    self.assertIsNot(calls[0][2], calls[1][2])

    def test_replay_resolves_one_current_context_before_launch(self):
        order = []

        class Base:
            def run_fullgraph(self, desc):
                order.append("replay")
                return "output"

        class FakeGraph:
            def resolve_tasks(self, source):
                order.append("resolve")
                self.source = source
                return "resolved"

            def update(self, stream, tasks):
                self.tasks = tasks
                order.append("update")

        cls = methods(
            GRAPH,
            "AutoRegressiveAclGraphManager",
            ["_updatable_graph_replay"],
            Base,
            {
                "UpdatableGraph": FakeGraph,
                "ContextSource": lambda context: SimpleNamespace(context=context),
                "torch": SimpleNamespace(npu=SimpleNamespace(current_stream=lambda: "stream")),
            },
        )
        g = cls()
        desc = object()
        graph = FakeGraph()
        g.graphs = {desc: graph}
        g.update_stream = SimpleNamespace(wait_stream=lambda stream: order.append("wait"))
        metadata = {"draft": object()}
        self.assertEqual(g._updatable_graph_replay(desc, [metadata]), "output")
        self.assertIs(graph.source.context, metadata)
        self.assertEqual(order, ["resolve", "wait", "replay", "update"])
        with self.assertRaises(AssertionError):
            g._updatable_graph_replay(desc, [metadata, metadata])


class ProviderContract(unittest.TestCase):
    def source(self, relative):
        path = ROOT / "vllm_ascend" / relative
        if not path.exists():
            path = ROOT.parents[1] / "current/vllm-ascend/vllm_ascend" / relative
        return path

    def test_fia_sliding_draft_and_nonpaged_branches(self):
        cls = methods(
            self.source("attention/attention_v1.py"),
            "FIAParamProvider",
            ["resolve"],
            object,
            {},
        )
        block = object()
        md = SimpleNamespace(actual_seq_lengths_q=[1, 2], seq_lens_list=[11, 21], block_tables=block)
        for draft, sliding, nonpaged in (
            (True, 128, False),
            (True, 128, True),
            (False, None, False),
            (False, 128, False),
        ):
            p = cls()
            p.layer_name = "draft"
            p.is_draft_model = draft
            p.sliding_window = sliding
            p.is_non_pa = nonpaged
            result = p.resolve({"draft": md})
            self.assertEqual(result["actual_seq_lengths"], [1, 2])
            self.assertEqual(result["actual_seq_lengths_kv"], [11, 21])
            if draft or not sliding:
                self.assertIs(result["block_table"], None if nonpaged else block)
            else:
                self.assertNotIn("block_table", result)

    def test_fiav2_and_pa_use_backend_fields(self):
        md = SimpleNamespace(
            actual_seq_lengths_q=[1],
            seq_lens_list=[11],
            seq_lens=object(),
            block_tables=object(),
        )
        for name in ("FIAV2ParamProvider", "PAParamProvider"):
            cls = methods(self.source("attention/attention_v1.py"), name, ["resolve"], object, {})
            p = cls()
            p.layer_name = "draft"
            result = p.resolve({"draft": md})
            if name == "FIAV2ParamProvider":
                self.assertEqual(result, {"actual_seq_qlen": [1], "actual_seq_kvlen": [11]})
            else:
                self.assertIs(result["context_lens"], md.seq_lens)
                self.assertIs(result["block_table"], md.block_tables)

    def test_dcp_current_history_and_full_scopes(self):
        scope = SimpleNamespace(CURRENT=1, HISTORY=2, FULL=3)
        cls = methods(
            self.source("attention/context_parallel/attention_cp.py"),
            "DCPFIAParamProvider",
            ["resolve"],
            object,
            {"CPKVScope": scope},
        )

        class Matrix:
            def __getitem__(self, index):
                return SimpleNamespace(tolist=lambda: [7, 8])

        decode = SimpleNamespace(
            actual_seq_lengths_q=[1, 2],
            cp_history_seq_len=[4, 5],
            block_tables=object(),
            num_computed_tokens_of_dcp=Matrix(),
        )
        md = SimpleNamespace(decode=decode)
        for kind, lengths in (
            (scope.CURRENT, [1, 2]),
            (scope.HISTORY, [4, 5]),
            (scope.FULL, [7, 8]),
        ):
            p = cls()
            p.metadata_layer_name = "draft"
            p.attention_kind = kind
            p.dcp_rank = 0
            result = p.resolve({"draft": md})
            self.assertEqual(result["actual_seq_lengths_kv"], lengths)
            self.assertIs(
                result["block_table"],
                None if kind == scope.CURRENT else decode.block_tables,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
