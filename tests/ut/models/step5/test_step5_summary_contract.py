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
import types
from pathlib import Path

import pytest

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
    cache.update_prefill("key", "z", Slice(5), "positions", ring)
    name, args, _ = kernels.calls[0]
    assert name == "update_prefill"
    assert args[0] == "key" and args[1] == "z" and args[3] == "positions"
    assert args[2] == Slice(5)
    assert args[4] == ring.base[:5]
    assert args[5] == ring.slot_mapping[:5]
    assert args[6] is cache.mean
    assert args[7] is ring.rows
    assert args[8] == ring.capacity


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
