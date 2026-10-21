"""Step5 summary-cache contracts under main's TailRing architecture.

AST isolation: the class definitions are compiled from source so these run on
CPU-only hosts as well as in the NPU unit-test suite. Only the ring is faked.

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


def _load_cache_cls():
    """Compile AscendStep5SparseSummaryCache out of the source file."""
    tree = ast.parse(SRC.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AscendStep5SparseSummaryCache")
    namespace = {"torch": __import__("torch")}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SRC), "exec"), namespace)  # noqa: S102
    return namespace["AscendStep5SparseSummaryCache"]


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


class FakeKernels:
    """Records the arguments each update entry point forwards."""

    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def record(*args, **kwargs):
            self.calls.append((name, args, kwargs))

        return record


@pytest.fixture
def kernels(monkeypatch):
    fake = FakeKernels()
    import vllm_ascend.models.step5.summary_kernels as mod

    for name in ("update_decode", "update_verify", "update_prefill"):
        monkeypatch.setattr(mod, name, getattr(fake, name), raising=False)
    return fake


@pytest.fixture
def cache():
    return _load_cache_cls()(mean=object())


def test_cache_holds_only_the_mean_tensor(cache):
    # The TailRing design keeps no per-request state: no accumulators, no
    # open-region cursor, no undo journal, no host snapshots.
    public = [name for name in vars(cache) if not name.startswith("_")]
    assert public == ["mean"]


def test_update_decode_addresses_the_ring(cache, kernels):
    ring = FakeRing(3)
    cache.update_decode("key", "z", "slots", "positions", ring)
    name, args, _ = kernels.calls[0]
    assert name == "update_decode"
    # (key, z, slots, positions, ring_base, mean, ring_rows, capacity)
    assert args[0:4] == ("key", "z", "slots", "positions")
    assert args[4] == ring.base[:3]
    assert args[5] is cache.mean
    assert args[6] is ring.rows
    assert args[7] == ring.capacity


def test_update_verify_addresses_the_ring(cache, kernels):
    ring = FakeRing(4)
    cache.update_verify("key", "z", "slots", "qsl", "seq_lens", ring)
    name, args, _ = kernels.calls[0]
    assert name == "update_verify"
    assert args[0:5] == ("key", "z", "slots", "qsl", "seq_lens")
    assert args[5] == ring.base[:4]
    assert args[6] is cache.mean
    assert args[7] is ring.rows
    assert args[8] == ring.capacity


def test_update_prefill_passes_both_ring_views(cache, kernels):
    # Prefill is the only path that also consumes slot_mapping, because a
    # long chunk may store open-region rows after summarising closed ones.
    ring = FakeRing(5)
    cache.update_prefill("key", "z", "metadata", "positions", ring)
    name, args, _ = kernels.calls[0]
    assert name == "update_prefill"
    assert args[0:4] == ("key", "z", "metadata", "positions")
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
