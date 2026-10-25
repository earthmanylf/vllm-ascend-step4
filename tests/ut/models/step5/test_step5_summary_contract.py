"""Step5 summary-cache contracts under main's TailRing architecture.

AST isolation: the class definitions are compiled straight from the source
files, so importing ``vllm_ascend`` (and therefore its build-time
``_build_info``) is never triggered. That keeps these runnable on CPU-only
hosts as well as in the NPU unit-test suite.

These replace the SWF-journal sources (test_summary_cache_contract.py,
test_summary_update_dispatch_contract.py, test_summary_scratch_contract.py),
which tested the per-request undo log that upstream vLLM removed. The cache
now holds exactly one member, ``mean``, and the three update entry points
address the ring instead of page accumulators.
"""

import ast
import os
import types
import weakref
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
SRC = REPO_ROOT / "vllm_ascend/models/step5/summary_cache.py"


class FakeRing:
    """Minimal stand-in for Step5CSATailRing.

    The real object exposes ``rows`` (the flattened ring storage),
    ``capacity`` (rows per page) and ``metadata()`` returning
    ``Step5CSATailRingMetadata(base, slot_mapping)``. Only those three are
    touched by the update entry points.
    """

    def __init__(self, n: int):
        self.base = list(range(n))
        self.slot_mapping = list(range(n))
        self.rows = object()
        self.capacity = 16

    def metadata(self):
        return types.SimpleNamespace(base=self.base, slot_mapping=self.slot_mapping)


class Slice:
    """Stand-in for a device tensor sliced by ``[:n]``.

    The cache reads ``slots.shape[0]`` (and ``metadata.shape[0]``) only to
    size the ring views, so a real tensor is unnecessary -- but the object
    must carry a ``shape``.
    """

    def __init__(self, n: int):
        self.shape = (n,)
        self.n = n

    def __getitem__(self, item):
        if isinstance(item, slice):
            return Slice(self.n)
        raise TypeError(item)

    def __eq__(self, other):
        return isinstance(other, Slice) and other.n == self.n

    def __repr__(self):
        return f"Slice({self.n})"


class FakeKernels:
    """Records the arguments each update entry point forwards."""

    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def record(*args, **kwargs):
            self.calls.append((name, args, kwargs))

        return record


def _load_cache_cls(kernels):
    """Compile AscendStep5SparseSummaryCache out of the source file.

    The module-level ``from vllm_ascend...summary_kernels import ...`` inside
    each method is redirected to the fake, and the ``from __future__`` import
    is dropped because the class is compiled standalone.
    """
    tree = ast.parse(SRC.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AscendStep5SparseSummaryCache")
    for fn in node.body:
        if isinstance(fn, ast.FunctionDef):
            fn.decorator_list = []
            fn.body = [
                stmt
                for stmt in fn.body
                if not (isinstance(stmt, ast.ImportFrom) and stmt.module and "summary_kernels" in stmt.module)
            ]
    namespace = {
        "torch": __import__("torch"),
        "vllm_ascend": types.SimpleNamespace(
            models=types.SimpleNamespace(
                step5=types.SimpleNamespace(summary_kernels=kernels),
            ),
        ),
        # Each method does ``from ...summary_kernels import update_*``; the
        # import statement is stripped above, so bind the names directly.
        "update_decode": kernels.update_decode,
        "update_verify": kernels.update_verify,
        "update_prefill": kernels.update_prefill,
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SRC), "exec"), namespace)  # noqa: S102
    return namespace["AscendStep5SparseSummaryCache"]


@pytest.fixture
def kernels():
    return FakeKernels()


@pytest.fixture
def cache(kernels):
    return _load_cache_cls(kernels)(mean=object())


def test_cache_holds_only_the_mean_tensor(cache):
    # The TailRing design keeps no per-request state: no accumulators, no
    # open-region cursor, no undo journal, no host snapshots.
    public = [name for name in vars(cache) if not name.startswith("_")]
    assert public == ["mean"]


def test_update_decode_addresses_the_ring(cache, kernels):
    ring = FakeRing(3)
    cache.update_decode("key", "z", Slice(3), "positions", ring)
    name, args, _ = kernels.calls[0]
    assert name == "update_decode"
    # (key, z, slots, positions, ring_base, mean, ring_rows, capacity)
    assert args[0] == "key" and args[1] == "z" and args[3] == "positions"
    assert args[2] == Slice(3)
    assert args[4] == ring.base[:3]
    assert args[5] is cache.mean
    assert args[6] is ring.rows
    assert args[7] == ring.capacity


def test_update_verify_addresses_the_ring(cache, kernels):
    ring = FakeRing(4)
    cache.update_verify("key", "z", Slice(4), "qsl", "seq_lens", ring)
    name, args, _ = kernels.calls[0]
    assert name == "update_verify"
    assert args[0] == "key" and args[1] == "z" and args[3] == "qsl" and args[4] == "seq_lens"
    assert args[2] == Slice(4)
    assert args[5] == ring.base[:4]
    assert args[6] is cache.mean
    assert args[7] is ring.rows
    assert args[8] == ring.capacity


def test_update_prefill_passes_both_ring_views(cache, kernels):
    # Prefill is the only path that also consumes slot_mapping, because a
    # long chunk may store open-region rows after summarising closed ones.
    ring = FakeRing(5)
    key = torch.zeros((5, 1, 256))
    cache.update_prefill(key, "z", Slice(5), "positions", ring)
    name, args, _ = kernels.calls[0]
    assert name == "update_prefill"
    assert args[0] is key and args[1] == "z" and args[3] == "positions"
    assert args[2] == Slice(5)
    assert args[4] == ring.base[:5]
    assert args[5] == ring.slot_mapping[:5]
    assert args[6] is cache.mean
    assert args[7] is ring.rows
    assert args[8] == ring.capacity


def test_prefill_ring_views_follow_token_rows_not_region_group_count(cache, kernels):
    ring = FakeRing(12)
    key = torch.zeros((9, 1, 256))
    metadata = torch.zeros((2, 10), dtype=torch.int64)
    cache.update_prefill(key, key, metadata, torch.arange(9), ring)
    _, args, _ = kernels.calls[0]
    assert args[4] == ring.base[:9]
    assert args[5] == ring.slot_mapping[:9]


def test_prefill_ring_views_skip_the_decode_prefix(cache, kernels):
    ring = FakeRing(12)
    key = torch.zeros((9, 1, 256))
    metadata = torch.zeros((2, 10), dtype=torch.int64)
    cache.update_prefill(key, key, metadata, torch.arange(9), ring, token_offset=2)
    _, args, _ = kernels.calls[0]
    assert args[4] == ring.base[2:11]
    assert args[5] == ring.slot_mapping[2:11]


def _load_dsa_method(name, namespace):
    source = REPO_ROOT / "vllm_ascend/models/step5/dsa_attention.py"
    tree = ast.parse(source.read_text())
    impl = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AscendStep5DSAImpl")
    method = next(n for n in impl.body if isinstance(n, ast.FunctionDef) and n.name == name)
    method.decorator_list = []
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    code = ast.fix_missing_locations(ast.Module(body=[future, method], type_ignores=[]))
    exec(compile(code, str(source), "exec"), namespace)  # noqa: S102
    return namespace[name]


@pytest.mark.parametrize("spec_tokens", [0, 3])
def test_mixed_batch_uses_decode_then_prefill_with_matching_ring_offset(spec_tokens):
    def groups(slots, positions, owners, qsl):
        assert slots.tolist() == [40, 41, 42]
        assert positions.tolist() == [5, 6, 7]
        return torch.zeros((1, 10), dtype=torch.int64)

    update = _load_dsa_method("_maybe_update_summary_cache", dict(torch=torch, build_prefill_groups=groups))
    calls = []
    cache = types.SimpleNamespace(
        update_decode=lambda *args, **kw: calls.append(("decode", args, kw)),
        update_verify=lambda *args, **kw: calls.append(("verify", args, kw)),
        update_prefill=lambda *args, **kw: calls.append(("prefill", args, kw)),
    )
    impl = types.SimpleNamespace(
        num_speculative_tokens=spec_tokens,
        csa_ring=FakeRing(4),
        _summary_owners=lambda qsl, rows: torch.zeros(rows, dtype=torch.int32),
    )
    md = types.SimpleNamespace(
        graph_decode=None,
        num_actual_tokens=4,
        num_decode_tokens=1,
        num_decodes=1,
        num_prefills=1,
        decode=types.SimpleNamespace(query_len=1),
        prefill=types.SimpleNamespace(query_start_loc=torch.tensor([0, 3])),
        slot_mapping=torch.tensor([120, 140, 141, 142]),
        query_start_loc=torch.tensor([0, 1, 4]),
        seq_lens=torch.tensor([21, 8]),
    )
    key = torch.zeros((4, 1, 256))
    update(
        impl,
        summary_cache=cache,
        summary_metadata=types.SimpleNamespace(slot_mapping=torch.tensor([20, 40, 41, 42])),
        md=md,
        index_k=key,
        index_z=key,
        positions=torch.tensor([20, 5, 6, 7]),
    )
    assert [call[0] for call in calls] == ["decode", "prefill"]
    assert calls[1][2]["token_offset"] == 1


@pytest.mark.parametrize("query_len", [1, 4])
def test_uniform_decode_only_uses_verify_for_actual_multitoken_queries(query_len):
    update = _load_dsa_method("_maybe_update_summary_cache", dict(torch=torch))
    calls = []
    cache = types.SimpleNamespace(
        update_decode=lambda *args: calls.append(("decode", args)),
        update_verify=lambda *args: calls.append(("verify", args)),
    )
    impl = types.SimpleNamespace(csa_ring=FakeRing(8), num_speculative_tokens=3)
    rows = 2 * query_len
    md = types.SimpleNamespace(
        graph_decode=None,
        num_actual_tokens=rows,
        num_decode_tokens=rows,
        num_decodes=2,
        num_prefills=0,
        decode=types.SimpleNamespace(query_len=query_len),
        query_start_loc=torch.tensor([0, query_len, rows, rows]),
        seq_lens=torch.tensor([20, 40, 0]),
        slot_mapping=torch.arange(rows),
    )
    key = torch.zeros((rows, 1, 256))
    update(
        impl, summary_cache=cache, summary_metadata=md, md=md, index_k=key, index_z=key, positions=torch.arange(rows)
    )
    assert calls[0][0] == ("decode" if query_len == 1 else "verify")
    if query_len > 1:
        assert calls[0][1][3].tolist() == [0, query_len, rows]
        assert calls[0][1][4].tolist() == [20, 40]


@pytest.mark.parametrize("tokens", [0, 1])
def test_forward_clears_padding_without_touching_actual_output(tokens):
    md = types.SimpleNamespace(
        graph_decode=None,
        decode=None,
        num_actual_tokens=tokens,
        num_decodes=0,
        num_prefills=tokens,
        num_decode_tokens=0,
    )
    namespace = dict(
        torch=torch,
        AscendStep5DSAMetadata=types.SimpleNamespace,
        AscendStep5SparseSummaryMetadata=types.SimpleNamespace,
        get_forward_context=lambda: types.SimpleNamespace(attn_metadata={"dsa": md, "summary": md}),
    )
    forward = _load_dsa_method("forward", namespace)
    impl = types.SimpleNamespace(
        summary_cache=object(),
        summary_layer_name="summary",
        _maybe_update_summary_cache=lambda **_: None,
        _forward_sparse_gqa_prefill=lambda **kw: kw["output"][:tokens].fill_(7),
    )
    output = torch.full((4, 1), 99.0)
    forward(
        impl,
        types.SimpleNamespace(layer_name="dsa"),
        output,
        (output, output),
        output,
        output,
        output,
        output,
        output,
        output,
    )
    assert torch.all(output[:tokens] == 7)
    assert torch.all(output[tokens:] == 0)


def test_no_journal_entry_points_remain(cache):
    # Every mechanism the SWF port needed for rollback is gone.
    for name in (
        "update_request_journal",
        "update_speculative",
        "reset_scheduler_blocks",
        "reset_blocks",
        "prefill_metadata",
    ):
        assert not hasattr(cache, name), f"{name} must not survive the TailRing rewrite"


def _load_cpu_builder():
    """Use the real upstream splitter and Ascend builder without runtime imports."""
    upstream = Path(os.environ.get("VLLM_SOURCE_ROOT", REPO_ROOT.parent / "vllm"))
    source = upstream / "vllm/v1/attention/backends/utils.py"
    tree = ast.parse(source.read_text())
    splitter = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "split_decodes_and_prefills")
    namespace = dict(torch=torch, dataclass=dataclass, AttentionMetadata=object)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[future, splitter], type_ignores=[])), str(source), "exec"),
        namespace,
    )  # noqa: S102
    source = REPO_ROOT / "vllm_ascend/models/step5/dsa_attention.py"
    tree = ast.parse(source.read_text())
    names = {"AscendStep5DSAPrefillMetadata", "AscendStep5DSADecodeMetadata", "AscendStep5DSAMetadata"}
    body = [future]
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in names:
            body.append(node)
        elif isinstance(node, ast.ClassDef) and node.name == "AscendStep5DSAMetadataBuilder":
            body.extend(
                n for n in node.body if isinstance(n, ast.FunctionDef) and n.name in {"build", "_prepare_gbsa_metadata"}
            )
        elif isinstance(node, ast.FunctionDef) and node.name.startswith("_active_"):
            body.append(node)
    exec(compile(ast.Module(body=body, type_ignores=[]), str(source), "exec"), namespace)  # noqa: S102
    builder = types.SimpleNamespace(
        reorder_batch_threshold=4,
        context_len_buffer=torch.empty(512, dtype=torch.int32),
        query_start_loc_int64_buffer=torch.empty(513, dtype=torch.int64),
        capture_sizes=set(),
        # Substitute only the device schedule, not its coordinate preparation.
        _build_gbsa_metadata=lambda positions, *args, **kw: positions.clone(),
    )
    for name in ("build", "_prepare_gbsa_metadata"):
        setattr(builder, name, types.MethodType(namespace[name], builder))
    return builder


def _cpu_common(query_lens, contexts=None, *, dummy_lens=()):
    contexts = contexts if contexts is not None else [5] * len(query_lens)
    all_lens = [*query_lens, *dummy_lens]
    qsl = torch.tensor([0, *torch.tensor(all_lens, dtype=torch.int32).cumsum(0).tolist()], dtype=torch.int32)
    tokens = sum(query_lens)
    return types.SimpleNamespace(
        query_start_loc=qsl,
        query_start_loc_cpu=qsl,
        seq_lens=torch.tensor([c + q for c, q in zip(contexts, query_lens)] + [0] * len(dummy_lens), dtype=torch.int32),
        block_table_tensor=torch.arange(len(all_lens), dtype=torch.int32).reshape(-1, 1),
        slot_mapping=torch.cat((torch.arange(tokens), torch.full((sum(dummy_lens),), -1))),
        num_reqs=len(all_lens),
        num_actual_tokens=tokens,
        max_query_len=max(all_lens, default=0),
    )


def test_real_splitter_builder_accepts_empty_mtp_batch():
    md = _load_cpu_builder().build(0, _cpu_common([]))
    assert (md.num_actual_tokens, md.num_decodes, md.num_prefills) == (0, 0, 0)
    assert (md.num_decode_tokens, md.num_prefill_tokens) == (0, 0)
    assert md.decode is md.prefill is None
    assert md.query_start_loc.tolist() == [0]


class CpuGroupKernel:
    """CPU replacement for the device grouping kernel, retaining its wrapper."""

    def __getitem__(self, grid):
        return self.run

    @staticmethod
    def run(slots, positions, owners, qsl, metadata, position_stride):
        metadata.fill_(-1)
        for row, position in enumerate(positions.tolist()):
            owner = int(owners[row])
            start, end = int(qsl[owner]), int(qsl[owner + 1])
            first = position % 8
            if row != start and first != 0:
                continue
            indices = list(range(row, min(end, row + 8 - first)))
            valid = [i for i in indices if slots[i] >= 0]
            if valid:
                metadata[row, 0] = int(slots[valid[-1]]) // 8
                metadata[row, 1] = first
                metadata[row, 2 + first : 2 + first + len(valid)] = torch.tensor(valid)


def _load_cpu_groups():
    source = REPO_ROOT / "vllm_ascend/models/step5/summary_kernels.py"
    tree = ast.parse(source.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "build_prefill_groups")
    namespace = dict(torch=torch, _build_prefill_groups=CpuGroupKernel())
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)  # noqa: S102
    return namespace[node.name]


class CpuRingKernels:
    """Coordinate oracle: fingerprints raw key/z rows, not summary math."""

    def __init__(self):
        self.paths = []

    @staticmethod
    def store(key, z, row, ring, slot):
        if slot >= 0:
            ring[slot] = torch.stack((key[row, 0, 0], z[row, 0, 0]))

    def update_decode(self, key, z, slots, positions, bases, mean, ring, capacity):
        self.paths.append("decode")
        for row, position in enumerate(positions.tolist()):
            base = int(bases[row])
            self.store(key, z, row, ring, base + position % capacity)
            if position % 8 == 7:
                indices = base + torch.arange(position - 7, position + 1) % capacity
                mean[int(slots[row]) // 8] = ring[indices].sum(0)

    def update_verify(self, key, z, slots, qsl, seq_lens, bases, mean, ring, capacity):
        self.paths.append("verify")
        positions = torch.cat(
            [
                torch.arange(int(seq) - int(end - start), int(seq))
                for start, end, seq in zip(qsl[:-1], qsl[1:], seq_lens)
            ]
        )
        self.update_decode(key, z, slots, positions, bases, mean, ring, capacity)
        self.paths.pop()

    def update_prefill(self, key, z, metadata, positions, bases, ring_slots, mean, ring, capacity):
        self.paths.append("prefill")
        # All prefix reads precede writes, including chunks that wrap the ring.
        for group in metadata:
            region, first = int(group[0]), int(group[1])
            if region < 0 or group[9] < 0:
                continue
            head = int(group[2 + first])
            position, base = int(positions[head]), int(bases[head])
            indices = base + torch.arange(position - first, position) % capacity
            fingerprint = ring[indices].sum(0)
            for row in group[2 + first :].tolist():
                fingerprint += torch.stack((key[row, 0, 0], z[row, 0, 0]))
            mean[region] = fingerprint
        for row, slot in enumerate(ring_slots.tolist()):
            self.store(key, z, row, ring, slot)


def _cpu_summary_impl(ring):
    namespace = dict(torch=torch, build_prefill_groups=_load_cpu_groups())
    update = _load_dsa_method("_maybe_update_summary_cache", namespace)
    owners = _load_dsa_method("_summary_owners", dict(torch=torch))
    impl = types.SimpleNamespace(csa_ring=ring, num_speculative_tokens=3, _summary_owners=owners)
    impl._maybe_update_summary_cache = types.MethodType(update, impl)
    return impl


def _run_cpu_summary_batch(requests, *, dummy_lens=()):
    """Run real splitter/build/update/slicing on shared CPU cache storage."""
    capacity = 16
    ring_rows = torch.arange(5 * capacity * 2, dtype=torch.float32).reshape(-1, 2) + 1000
    mean = torch.full((5 * 16, 2), -999.0)
    initial_ring, initial_mean = ring_rows.clone(), mean.clone()
    common = _cpu_common([r[2] for r in requests], [r[1] for r in requests], dummy_lens=dummy_lens)
    positions = torch.cat([torch.arange(context, context + length) for _, context, length in requests])
    slots = torch.cat([owner * 128 + torch.arange(context, context + length) for owner, context, length in requests])
    bases = torch.cat([torch.full((length,), owner * capacity) for owner, _, length in requests])
    keys = torch.cat(
        [owner * 100 + torch.arange(context, context + length) for owner, context, length in requests]
    ).float()
    key = keys.reshape(-1, 1, 1)
    z = (keys * 3 + 1).reshape(-1, 1, 1)
    padding = sum(dummy_lens)
    common.slot_mapping = torch.cat((slots, torch.full((padding,), -1)))
    ring = types.SimpleNamespace(
        rows=ring_rows,
        capacity=capacity,
        metadata=lambda: types.SimpleNamespace(base=bases, slot_mapping=bases + positions % capacity),
    )
    kernels = CpuRingKernels()
    cache = _load_cache_cls(kernels)(mean)
    impl = _cpu_summary_impl(ring)
    md = _load_cpu_builder().build(0, common)
    impl._maybe_update_summary_cache(
        summary_cache=cache, summary_metadata=md, md=md, index_k=key, index_z=z, positions=positions
    )
    # Request 4 is absent: both its summary and raw ring must stay untouched.
    assert torch.equal(ring_rows[4 * capacity :], initial_ring[4 * capacity :])
    assert torch.equal(mean[4 * 16 :], initial_mean[4 * 16 :])
    return md, kernels.paths, ring_rows, mean


@pytest.mark.parametrize("decode_len", [1, 4])
@pytest.mark.parametrize("dummy_lens", [(), (0,), (4,)])
def test_real_mixed_summary_cache_matches_independent_and_reordered_requests(decode_len, dummy_lens):
    requests = [(0, 7, decode_len), (1, 15, decode_len), (2, 5, 9), (3, 13, 11)]
    md, paths, ring, mean = _run_cpu_summary_batch(requests, dummy_lens=dummy_lens)
    assert (md.num_decodes, md.num_prefills) == (2, 2)
    assert md.decode.query_len == decode_len
    assert paths == ["decode" if decode_len == 1 else "verify", "prefill"]
    assert md.prefill.context_lens_host == [5, 13]
    assert md.prefill.query_start_loc.tolist() == [0, 9, 20]
    assert md.prefill.gbsa_metadata.tolist() == [*range(5, 14), *range(13, 24)]
    for request in requests:
        owner = request[0]
        _, _, independent_ring, independent_mean = _run_cpu_summary_batch([request])
        assert torch.equal(ring[owner * 16 : (owner + 1) * 16], independent_ring[owner * 16 : (owner + 1) * 16])
        assert torch.equal(mean[owner * 16 : (owner + 1) * 16], independent_mean[owner * 16 : (owner + 1) * 16])
    reordered = [requests[1], requests[0], requests[3], requests[2]]
    _, _, reordered_ring, reordered_mean = _run_cpu_summary_batch(reordered, dummy_lens=dummy_lens)
    assert torch.equal(ring, reordered_ring)
    assert torch.equal(mean, reordered_mean)
    # The first partial-region summary consumes five old rows plus three new.
    initial = torch.arange(160, dtype=torch.float32).reshape(-1, 2) + 1000
    expected = initial[32:37].sum(0) + torch.tensor([[205, 616], [206, 619], [207, 622]]).sum(0)
    assert torch.equal(mean[32], expected)


@pytest.mark.parametrize("query_lens,expected", [([1, 2, 3], (1, 2, 1)), ([129], (0, 1, 0))])
@pytest.mark.parametrize("dummy_lens", [(), (0,), (4,)])
def test_real_splitter_builder_keeps_ragged_and_long_mtp_queries_on_prefill(query_lens, expected, dummy_lens):
    md = _load_cpu_builder().build(0, _cpu_common(query_lens, dummy_lens=dummy_lens))
    assert (md.num_decodes, md.num_prefills, md.num_decode_tokens) == expected
    assert md.num_prefill_tokens == sum(query_lens) - expected[2]
    assert md.prefill.num_query_tokens == md.num_prefill_tokens
    assert md.prefill.query_lens.tolist() == query_lens[expected[0] :]
    assert md.prefill.context_lens_host == [5] * expected[1]
    assert md.prefill.max_query_len == max(query_lens)


@pytest.mark.parametrize("requests", [[(0, 7, 1), (1, 6, 2), (2, 5, 3)], [(2, 5, 129)]])
def test_real_ragged_and_long_prefill_summary_matches_independent_requests(requests):
    _, _, ring, mean = _run_cpu_summary_batch(requests)
    for request in requests:
        owner = request[0]
        _, _, independent_ring, independent_mean = _run_cpu_summary_batch([request])
        assert torch.equal(ring[owner * 16 : (owner + 1) * 16], independent_ring[owner * 16 : (owner + 1) * 16])
        assert torch.equal(mean[owner * 16 : (owner + 1) * 16], independent_mean[owner * 16 : (owner + 1) * 16])


@pytest.mark.parametrize(
    "query_lens,dummy_lens", [([], ()), ([4, 9], (0,)), ([4, 9], (4,)), ([4, 4], (0,)), ([4, 4], (4,))]
)
def test_real_builder_forward_dispatch_and_padding(query_lens, dummy_lens):
    md = _load_cpu_builder().build(0, _cpu_common(query_lens, dummy_lens=dummy_lens))
    paths = []

    def attention(path, value, **kwargs):
        paths.append(path)
        start = kwargs.get("token_offset", 0)
        kwargs["output"][start : start + kwargs["query"].shape[0]].fill_(value)

    impl = types.SimpleNamespace(
        summary_cache=object(),
        summary_layer_name="summary",
        num_speculative_tokens=3,
        _maybe_update_summary_cache=lambda **kw: None,
        _forward_sparse_gqa_decode=lambda **kw: attention("decode", 1, **kw),
        _forward_sparse_gqa_verify=lambda **kw: attention("verify", 4, **kw),
        _forward_sparse_gqa_prefill=lambda **kw: attention("prefill", 9, **kw),
    )
    namespace = dict(
        torch=torch,
        AscendStep5DSAMetadata=type(md),
        AscendStep5SparseSummaryMetadata=type(md),
        get_forward_context=lambda: types.SimpleNamespace(attn_metadata={"dsa": md, "summary": md}),
    )
    forward = _load_dsa_method("forward", namespace)
    output = torch.full((max(1, sum(query_lens) + sum(dummy_lens)), 1), 99.0)
    forward(impl, types.SimpleNamespace(layer_name="dsa"), output, (output, output), *([output] * 6))
    assert paths == (["verify", "prefill"] if md.num_prefills else ["verify"] if md.num_decodes else [])
    assert torch.all(output[: md.num_decode_tokens] == 4)
    assert torch.all(output[md.num_decode_tokens : md.num_actual_tokens] == 9)
    assert torch.all(output[md.num_actual_tokens :] == 0)


def _load_binding(namespace):
    source = REPO_ROOT / "vllm_ascend/patch/worker/patch_bind_kv_cache.py"
    tree = ast.parse(source.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "bind_kv_cache_to_layers")
    namespace.update(
        torch=torch,
        Sequence=list,
        Attention=object,
        KVCacheGroupSpec=object,
        extract_layer_index=lambda *_: 0,
        utils=types.SimpleNamespace(share_replayssm_ring_trackers=lambda *_: None),
    )
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), "exec"), namespace)  # noqa: S102
    return namespace[fn.name]


def test_binding_initializes_step5_sidecars_without_changing_attention_tuples():
    class Sidecar:
        def bind_kv_cache(self, cache):
            self.kv_cache = cache
            self.initialized = True

    class Summary(Sidecar):
        pass

    class Ring(Sidecar):
        pass

    class Deepseek(Sidecar):
        pass

    class Ordinary:
        def bind_kv_cache(self, _):
            raise AssertionError("Ascend K/V tuples must not use upstream tensor binding")

    namespace = dict(
        Step5SparseSummaryCacheLayer=Summary,
        Step5CSATailRing=Ring,
        DeepseekV41CacheLayer=Deepseek,
    )
    bind = _load_binding(namespace)
    layers = dict(summary=Summary(), ring=Ring(), deepseek=Deepseek(), ordinary=Ordinary())
    pair = (torch.zeros(1), torch.zeros(1))
    caches = dict(summary=torch.zeros(1), ring=torch.zeros(1), deepseek=torch.zeros(1), ordinary=pair)
    bind(caches, layers)
    assert layers["summary"].initialized
    assert layers["ring"].initialized
    assert layers["deepseek"].initialized
    assert layers["ordinary"].kv_cache is pair


def test_dsa_backend_accepts_upstream_optional_cache_spec():
    source = REPO_ROOT / "vllm_ascend/models/step5/dsa_attention.py"
    tree = ast.parse(source.read_text())
    backend = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AscendStep5DSABackend")
    method = next(
        n for n in backend.body if isinstance(n, ast.FunctionDef) and n.name == "get_supported_kernel_block_sizes"
    )
    method.decorator_list = []
    namespace = dict(MultipleOf=object, AttentionSpec=object)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)  # noqa: S102
    assert namespace[method.name](None) == [128]
    assert namespace[method.name]() == [128]


def test_summary_spec_preserves_logical_block_size_with_physical_page_padding():
    from dataclasses import replace

    @dataclass(frozen=True)
    class Spec:
        block_size: int = 128
        page_size_padded: int | None = None
        tokens_per_state: int = 8
        head_size: int = 256

        @property
        def page_size_bytes(self):
            return self.page_size_padded or self.block_size // self.tokens_per_state * self.head_size

    class Base:
        def get_kv_cache_spec(self, _):
            return Spec()

    tree = ast.parse(SRC.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AscendStep5SparseSummaryCacheLayer")
    ns = dict(
        torch=torch,
        Any=object,
        Step5SparseSummaryCacheLayer=Base,
        uses_step5_blnhc=lambda _: False,
        replace=replace,
        AscendStep5SparseSummaryBackend=object,
    )
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SRC), "exec"), ns)  # noqa: S102
    layer = ns[node.name].__new__(ns[node.name])
    main_layer = types.SimpleNamespace(
        get_kv_cache_spec=lambda _: types.SimpleNamespace(block_size=128, page_size_bytes=98304)
    )
    layer._main_layer_ref = lambda: main_layer
    spec = layer.get_kv_cache_spec(None)
    assert spec.block_size == 128
    assert spec.tokens_per_state == 8
    assert spec.page_size_bytes == 98304
    assert spec.block_size // spec.tokens_per_state == 16


def test_summary_binding_installs_the_cache_used_by_dsa():
    tree = ast.parse(SRC.read_text())
    layer_node = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AscendStep5SparseSummaryCacheLayer"
    )
    namespace = dict(
        torch=torch,
        Any=object,
        Step5SparseSummaryCacheLayer=object,
        AscendStep5SparseSummaryBackend=object,
        AscendStep5SparseSummaryCache=_load_cache_cls(FakeKernels()),
    )
    exec(compile(ast.Module(body=[layer_node], type_ignores=[]), str(SRC), "exec"), namespace)  # noqa: S102
    layer = namespace[layer_node.name].__new__(namespace[layer_node.name])
    layer._target_impl = types.SimpleNamespace(summary_layer_name="summary")
    pages = torch.zeros((3, 1, 16, 256), dtype=torch.uint8)
    layer.bind_kv_cache(pages)
    assert layer._target_impl.summary_cache.mean.data_ptr() == pages.data_ptr()
    dsa_source = REPO_ROOT / "vllm_ascend/models/step5/dsa_attention.py"
    dsa_tree = ast.parse(dsa_source.read_text())
    impl = next(n for n in dsa_tree.body if isinstance(n, ast.ClassDef) and n.name == "AscendStep5DSAImpl")
    forward = next(n for n in impl.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
    forward.decorator_list = []
    md = types.SimpleNamespace(graph_decode=None, decode=None, num_actual_tokens=1, num_decodes=0, num_prefills=0)
    forward_namespace = dict(
        torch=torch,
        AttentionLayer=object,
        AscendStep5DSAMetadata=types.SimpleNamespace,
        AscendStep5SparseSummaryMetadata=types.SimpleNamespace,
        get_forward_context=lambda: types.SimpleNamespace(attn_metadata={"dsa": md, "summary": md}),
    )
    exec(compile(ast.Module(body=[forward], type_ignores=[]), str(dsa_source), "exec"), forward_namespace)  # noqa: S102
    observed = []
    layer._target_impl._maybe_update_summary_cache = lambda **kw: observed.append(kw["summary_cache"])
    output = torch.zeros((1, 1))
    result = forward_namespace["forward"](
        layer._target_impl,
        types.SimpleNamespace(layer_name="dsa"),
        output,
        (output, output),
        output,
        output,
        output,
        output,
        output,
        output,
    )
    assert result is output
    assert observed == [layer._target_impl.summary_cache]


def _load_summary_backend_types():
    tree = ast.parse(SRC.read_text())
    names = {
        "AscendStep5SparseSummaryMetadata",
        "AscendStep5SparseSummaryMetadataBuilder",
        "AscendStep5SparseSummaryBackend",
    }
    upstream = Path(os.environ["VLLM_SOURCE_ROOT"]) / "vllm/models/step5/nvidia/sparse_summary_cache.py"
    base_tree = ast.parse(upstream.read_text())
    backend = next(n for n in base_tree.body if isinstance(n, ast.ClassDef) and n.name == "Step5SparseSummaryBackend")
    namespace = dict(
        torch=torch,
        dataclass=dataclass,
        AttentionMetadata=object,
        Step5SparseSummaryMetadataBuilder=object,
        AttentionBackend=object,
    )
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    nodes = [future, backend, *(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in names)]
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(SRC), "exec"), namespace)
    return types.SimpleNamespace(**namespace)


def test_summary_constructor_installs_independent_metadata_layer_name():

    upstream = Path(os.environ["VLLM_SOURCE_ROOT"]) / "vllm/models/step5/nvidia/sparse_summary_cache.py"
    tree = ast.parse(upstream.read_text())
    base = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Step5SparseSummaryCacheLayer")
    base.bases = [ast.Name(id="object", ctx=ast.Load())]
    source_tree = ast.parse(SRC.read_text())
    layer = next(
        n for n in source_tree.body if isinstance(n, ast.ClassDef) and n.name == "AscendStep5SparseSummaryCacheLayer"
    )
    backend = _load_summary_backend_types().AscendStep5SparseSummaryBackend
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    namespace = dict(torch=torch, weakref=weakref, AscendStep5SparseSummaryBackend=backend)
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[future, base, layer], type_ignores=[])), str(SRC), "exec"),
        namespace,
    )
    target = types.SimpleNamespace()
    main = FakeRing(1)
    context = {}
    summary_layer = namespace[layer.name](
        prefix="summary", target_impl=target, main_layer=main, static_forward_context=context
    )
    assert target.summary_layer_name == "summary"
    assert context["summary"] is summary_layer
    assert summary_layer.get_attn_backend() is backend
    assert isinstance(summary_layer, namespace[base.name])


def test_summary_backend_retains_registered_name_and_graph_buffer_addresses():
    classes = _load_summary_backend_types()
    backend = classes.AscendStep5SparseSummaryBackend
    assert backend.get_name() == "STEP5_CSA_SUMMARY"
    builder = backend.get_builder_cls().__new__(backend.get_builder_cls())
    capture = Path(os.environ["VLLM_SOURCE_ROOT"]) / "vllm/v1/attention/backend.py"
    tree = ast.parse(capture.read_text())
    base = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AttentionMetadataBuilder")
    method = next(n for n in base.body if isinstance(n, ast.FunctionDef) and n.name == "build_for_cudagraph_capture")
    namespace = {}
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[future, method], type_ignores=[])), str(capture), "exec"),
        namespace,
    )
    builder.build_for_cudagraph_capture = types.MethodType(namespace[method.name], builder)
    common = _cpu_common([1], [7])
    common.slot_mapping = torch.tensor([2 * 128 + 7])
    common.block_table_tensor.fill_(2)
    metadata = builder.build(0, common)
    assert metadata.slot_mapping is common.slot_mapping
    assert metadata.block_table is common.block_table_tensor
    addresses = metadata.slot_mapping.data_ptr(), metadata.block_table.data_ptr()
    common.slot_mapping.add_(128)
    common.block_table_tensor.add_(1)
    rebuilt = builder.build_for_cudagraph_capture(common)
    assert (rebuilt.slot_mapping.data_ptr(), rebuilt.block_table.data_ptr()) == addresses
    assert metadata.slot_mapping.tolist() == [3 * 128 + 7]
    assert metadata.block_table.tolist() == [[3]]


@pytest.mark.parametrize("query_len", [1, 4, 9])
def test_summary_group_physical_overlay_isolated_from_main_and_ring(query_len):
    # Main/summary/ring descriptors overlay one page-strided CPU allocation.
    page_bytes, capacity = 98304, 16
    storage = torch.full((4 * page_bytes,), 83, dtype=torch.uint8)
    mean = storage.as_strided((4, 16, 256), (page_bytes, 256, 1))

    class PagedMean:
        def __setitem__(self, row, value):
            page, region = divmod(row, 16)
            mean.view(torch.float8_e4m3fn)[page, region, :2] = value.to(torch.float8_e4m3fn)

    class OverlayKernels(CpuRingKernels):
        def update_decode(self, key, z, slots, positions, bases, mean, ring, capacity):
            super().update_decode(key, z, slots, positions, bases, PagedMean(), ring, capacity)

        def update_prefill(self, key, z, metadata, positions, bases, ring_slots, mean, ring, capacity):
            super().update_prefill(key, z, metadata, positions, bases, ring_slots, PagedMean(), ring, capacity)

    # Ring page 3 lives at its descriptor's physical stride, not summary page 2.
    ring_rows = storage.view(torch.float32).as_strided((4, capacity, 2), (page_bytes // 4, 2, 1))
    ring_page = ring_rows[3]
    ring_page[:7] = torch.tensor([1.0, 2.0])
    before = storage.clone()
    positions = torch.arange(7, 7 + query_len)
    common = _cpu_common([query_len], [7])
    common.slot_mapping = 128 + positions
    common.block_table_tensor.fill_(1)
    md = _load_cpu_builder().build(0, common)
    summary_common = _cpu_common([query_len], [7])
    summary_common.slot_mapping = 2 * 128 + positions
    summary_common.block_table_tensor.fill_(2)
    classes = _load_summary_backend_types()
    builder = classes.AscendStep5SparseSummaryMetadataBuilder.__new__(classes.AscendStep5SparseSummaryMetadataBuilder)
    summary = builder.build(0, summary_common)
    ring = types.SimpleNamespace(
        rows=ring_page,
        capacity=capacity,
        metadata=lambda: types.SimpleNamespace(
            base=torch.zeros(query_len, dtype=torch.int64), slot_mapping=positions % capacity
        ),
    )
    cache = _load_cache_cls(OverlayKernels())(mean)
    impl = _cpu_summary_impl(ring)
    impl.summary_cache, impl.summary_layer_name = cache, "summary"
    impl.num_kv_groups, impl.region_size, impl.topk_regions = 1, 8, 512
    impl._max_regions, impl.scale = 16, 0.125
    calls = []

    def scorer(query, weights, summary_mean, table, *args, **kwargs):
        calls.append("scorer")
        assert table.tolist() == [[2]]
        assert summary_mean.data_ptr() == mean.data_ptr()
        assert summary_mean[int(table[0, 0]), 0, :2].float().tolist() == [8, 16]
        return None, torch.zeros(query.shape[0], 512, dtype=torch.int32)

    def attention(query, key_cache, value_cache, ids, pos, table, *args, **kwargs):
        calls.append("attention")
        assert table.tolist() == [[1]]
        assert key_cache.data_ptr() == storage[page_bytes:].data_ptr()
        return torch.ones_like(query)

    namespace = dict(
        torch=torch,
        AscendStep5DSAMetadata=type(md),
        AscendStep5SparseSummaryMetadata=type(summary),
        get_forward_context=lambda: types.SimpleNamespace(attn_metadata={"dsa": md, "summary": summary}),
        indexer_score_topk=scorer,
        gbsa_sparse_attention=attention,
    )
    for path in ("decode", "verify", "prefill"):
        name = "_forward_sparse_gqa_" + path
        setattr(impl, name, types.MethodType(_load_dsa_method(name, namespace), impl))
    forward = _load_dsa_method("forward", namespace)
    key = torch.ones(query_len, 1, 1)
    output = torch.zeros_like(key)
    forward(
        impl,
        types.SimpleNamespace(layer_name="dsa"),
        key,
        (storage[page_bytes:], storage[page_bytes:]),
        key,
        key,
        key * 2,
        key,
        output,
        positions,
    )
    assert calls == ["scorer", "attention"]
    assert output.eq(1).all()
    assert mean.view(torch.float8_e4m3fn)[2, 0, :2].float().tolist() == [8, 16]
    assert torch.equal(storage[page_bytes : 2 * page_bytes], before[page_bytes : 2 * page_bytes])
    assert torch.equal(storage[:page_bytes], before[:page_bytes])
    summary_page = storage[2 * page_bytes : 3 * page_bytes].clone()
    summary_page[:2] = before[2 * page_bytes : 2 * page_bytes + 2]
    if query_len == 9:
        summary_page[256:258] = before[2 * page_bytes + 256 : 2 * page_bytes + 258]
    assert torch.equal(summary_page, before[2 * page_bytes : 3 * page_bytes])
    assert ring_page[7].tolist() == [1, 2]


@pytest.mark.parametrize("summary_entry", ["absent", None])
def test_forward_requires_independent_summary_metadata(summary_entry):
    md = types.SimpleNamespace(graph_decode=None, decode=None, num_actual_tokens=1, num_decodes=0, num_prefills=0)
    context = {"dsa": md}
    if summary_entry != "absent":
        context["summary"] = summary_entry
    namespace = dict(
        torch=torch,
        AscendStep5DSAMetadata=type(md),
        AscendStep5SparseSummaryMetadata=type("SummaryMetadata", (), {}),
        get_forward_context=lambda: types.SimpleNamespace(attn_metadata=context),
    )
    forward = _load_dsa_method("forward", namespace)
    impl = types.SimpleNamespace(
        summary_cache=object(), summary_layer_name="summary", _maybe_update_summary_cache=lambda **kw: None
    )
    output = torch.zeros(1, 1)
    with pytest.raises(RuntimeError, match="summary.*metadata"):
        forward(impl, types.SimpleNamespace(layer_name="dsa"), output, (output, output), *([output] * 6))


@pytest.mark.parametrize("block", [0, 910, 911, 16384, 21999])
@pytest.mark.parametrize("stride", [98304, 2359296])
def test_summary_store_preserves_wide_page_addresses(block, stride):
    """Execute the production address expression with actual int32 CPU tensors.

    Numerical FP8 conversion is stubbed; pointer offsets are recorded without
    allocating a huge backing. Compiled NPU IR and model evidence cover device
    execution separately.
    """
    source = REPO_ROOT / "vllm_ascend/models/step5/summary_kernels.py"
    node = next(
        n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "_store_summary"
    )
    node.decorator_list = []
    stores = []

    class CastValue:
        def to(self, *args, **kwargs):
            return self

    class Pointer:
        def __init__(self, offset=0):
            self.offset = offset

        def __add__(self, offset):
            return Pointer(self.offset + offset)

    tl = types.SimpleNamespace(
        int64=torch.int64,
        bfloat16=torch.bfloat16,
        float8e4nv=torch.float8_e4m3fn,
        uint8=torch.uint8,
        div_rn=lambda *args: CastValue(),
        store=lambda ptr, value: stores.append(ptr.offset),
    )
    namespace = dict(tl=tl, _maximum=max)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[future, node], type_ignores=[])), str(source), "exec"),
        namespace,
    )  # noqa: S102
    slot = torch.tensor(block * 128 + 15, dtype=torch.int32)
    dims = torch.arange(256, dtype=torch.int64)
    namespace["_store_summary"](Pointer(), slot, dims, 1.0, 1.0, stride, 128)
    assert len(stores) == 1
    assert torch.equal(stores[0], block * stride + 256 + dims)
