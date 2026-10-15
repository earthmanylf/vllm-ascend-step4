# SPDX-License-Identifier: Apache-2.0
"""Actual DSA journal dispatch with independent scalar request/page state."""

import ast
import bisect
import types
import unittest
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[4] / "vllm_ascend/models/step5/dsa_attention.py"


class Rows:
    device = "cpu"
    dtype = "int64"

    def __init__(self, values):
        self.values = list(values)

    def __getitem__(self, index):
        if isinstance(index, Rows):
            return Rows([self.values[i] for i in index.values])
        return Rows(self.values[index]) if isinstance(index, slice) else self.values[index]

    def __sub__(self, other):
        return Rows([value - other for value in self.values])

    def __lt__(self, other):
        return Rows([value < other for value in self.values])

    def numel(self):
        return len(self.values)

    def contiguous(self):
        return self

    def long(self):
        return self

    def to(self, dtype):
        return self

    def masked_fill(self, mask, replacement):
        return Rows([replacement if masked else value for value, masked in zip(self.values, mask.values)])

    def view(self, *shape):
        return self


class ScalarJournalCache:
    """Independent state model, retaining only the dispatch-visible semantics."""

    block_size = 16

    def __init__(self):
        self.pages = {}
        self.mean = {}
        self.journal = {}
        self.calls = []

    def apply_row(self, key, slot):
        if slot < 0:
            return
        page = slot // self.block_size
        if slot % 8 == 0:
            self.pages[page] = []
        self.pages[page].append(key)
        self.mean[slot // 8] = sum(self.pages[page]) / len(self.pages[page])

    def update_request_journal(self, key, z, slots, starts, seq_lens, states, prefix_metadata=None):
        self.calls.append(("journal", key, z, slots, starts, seq_lens, states, prefix_metadata))
        width = 2
        # Fixed SWF restores every request before applying non-journal prefix.
        for request, state in enumerate(states.values):
            start, end = starts[request], starts[request + 1]
            if state < 0 or start == end or slots[start] < 0:
                continue
            saved = self.journal.get((state, (seq_lens[request] - (end - start)) % width))
            if saved is not None and saved[0] == slots[start]:
                self.pages[slots[start] // self.block_size] = saved[1][:]
        if prefix_metadata is not None:
            for row in prefix_metadata:
                self.apply_row(key[row], slots[row])
        for request, state in enumerate(states.values):
            start, end = starts[request], starts[request + 1]
            if state < 0 or start == end or slots[start] < 0:
                continue
            for row in range(max(start, end - width), end):
                slot = slots[row]
                if slot < 0:
                    continue
                page = slot // self.block_size
                position = seq_lens[request] - (end - row)
                self.journal[state, position % width] = (slot, self.pages.get(page, [])[:])
                self.apply_row(key[row], slot)

    def update_decode(self, key, z, slots):
        self.calls.append(("decode", key, z, slots))
        for row, slot in enumerate(slots.values):
            self.apply_row(key[row], slot)

    def update(self, key, z, slots, metadata):
        self.calls.append(("prefill", key, z, slots, metadata))
        for row in metadata:
            self.apply_row(key[row], slots[row])


class TestSummaryJournalDispatch(unittest.TestCase):
    def setUp(self):
        self.groups = []

        def build_groups(slots, positions, owners, starts, block_size, prefix_width=0, *, out=None):
            # Independent per-request prefix specification; production kernel
            # grouping/rounding is checked separately on NPU.
            selected = []
            for row, slot in enumerate(slots.values):
                request = owners[row]
                start, end = starts[request], starts[request + 1]
                limit = max(start, end - prefix_width) if prefix_width else end
                if slot >= 0 and slots[start] >= 0 and row < limit:
                    selected.append(row)
            self.groups.append((slots, positions, owners, starts, block_size, prefix_width, selected))
            return selected

        tree = ast.parse(SOURCE.read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AscendStep5DSAImpl")
        cls.bases = []
        wanted = {"_maybe_update_summary_cache", "_summary_owners", "forward"}
        cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
        namespace = {
            "torch": types.SimpleNamespace(
                int32="int32",
                arange=lambda count, **kwargs: Rows(range(count)),
                searchsorted=lambda starts, rows, **kwargs: Rows(
                    [bisect.bisect_right(starts.values, row) for row in rows.values]
                ),
            ),
            "build_prefill_groups": build_groups,
            "AscendStep5DSAMetadata": types.SimpleNamespace,
        }
        body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), cls]
        exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), str(SOURCE), "exec"), namespace)
        self.namespace = namespace
        self.impl = namespace[cls.name]()
        self.impl._summary_metadata_buffer = lambda rows, device: None
        self.impl.num_speculative_tokens = 1
        self.impl.num_kv_heads = 1
        self.impl.head_size = 1
        self.cache = ScalarJournalCache()

    def metadata(self, starts, sequences, states, *, prefills=0, decode_tokens=None, query_len=1):
        starts = Rows(starts)
        count = len(states) - prefills
        tokens = starts[-1]
        return types.SimpleNamespace(
            num_actual_tokens=tokens,
            num_decodes=count,
            num_decode_tokens=tokens if decode_tokens is None else decode_tokens,
            num_prefills=prefills,
            step5_request_state=Rows(states),
            query_start_loc=starts,
            seq_lens=Rows(sequences),
            decode=types.SimpleNamespace(query_len=query_len) if count else None,
            prefill=types.SimpleNamespace(query_start_loc=Rows([0, tokens - (decode_tokens or 0)]))
            if prefills
            else None,
        )

    def update(self, md, keys, slots, positions):
        md.slot_mapping = Rows(slots)
        self.impl._maybe_update_summary_cache(
            summary_cache=self.cache,
            md=md,
            index_k=Rows(keys),
            index_z=Rows([0] * len(keys)),
            positions=Rows(positions),
        )

    def test_mixed_prefix_is_per_request_and_padding_never_updates_a_live_page(self):
        self.cache.pages[0] = list(range(1, 8))
        md = self.metadata([0, 1, 8, 10], [8, 7, 2], [5, 2, -1], prefills=2, decode_tokens=1)
        self.update(md, [8, *range(10, 17), 1000, 2000], [7, *range(16, 23), 0, 1], [7, *range(7), 0, 1])
        self.assertEqual([call[0] for call in self.cache.calls], ["journal"])
        call = self.cache.calls[0]
        self.assertIs(call[6], md.step5_request_state)
        self.assertEqual(call[6].values, [5, 2, -1])
        self.assertEqual(call[3].values[-2:], [-1, -1])
        self.assertEqual(self.groups[0][2].values, [0, 1, 1, 1, 1, 1, 1, 1, 2, 2])
        self.assertEqual(self.groups[0][5:], (2, [1, 2, 3, 4, 5]))
        self.assertEqual(self.cache.mean[0], 4.5)
        self.assertEqual(self.cache.mean[2], 13)
        self.assertEqual(self.cache.pages[0], list(range(1, 9)))

    def test_query_one_uses_same_journal_and_restores_rejected_tail(self):
        self.cache.pages[0] = list(range(1, 7))
        md = self.metadata([0, 2], [8], [5], query_len=2)
        self.update(md, [7, 999], [6, 7], [6, 7])
        md = self.metadata([0, 1], [8], [5], query_len=1)
        self.update(md, [8], [7], [7])
        self.assertEqual([call[0] for call in self.cache.calls], ["journal", "journal"])
        self.assertEqual(self.cache.mean[0], 4.5)
        self.assertEqual(self.cache.pages[0], list(range(1, 9)))
        self.assertEqual(self.groups, [])

    def test_each_long_prefill_excludes_its_own_last_journal_window(self):
        md = self.metadata([0, 5, 12], [5, 7], [5, 2], prefills=2, decode_tokens=0)
        self.update(md, [*range(1, 6), *range(10, 17)], [*range(5), *range(16, 23)], [*range(5), *range(7)])
        self.assertEqual(self.groups[0][-1], [0, 1, 2, 5, 6, 7, 8, 9])
        self.assertEqual(self.cache.mean, {0: 3, 2: 13})
        self.assertEqual([call[0] for call in self.cache.calls], ["journal"])

    def test_pure_spec_decode_passes_reordered_request_indices_unchanged(self):
        md = self.metadata([0, 2, 4], [2, 2], [5, 2], query_len=2)
        self.update(md, [1, 3, 10, 14], [0, 1, 16, 17], [0, 1, 0, 1])
        self.assertIs(self.cache.calls[0][6], md.step5_request_state)
        self.assertEqual(self.cache.mean, {0: 2, 2: 12})
        self.assertEqual(set(state for state, position in self.cache.journal), {5, 2})

    def test_missing_request_state_fails_and_empty_batch_does_not_launch(self):
        md = self.metadata([0, 1], [1], [5])
        md.step5_request_state = None
        with self.assertRaisesRegex(RuntimeError, "persistent"):
            self.update(md, [1], [0], [0])
        md.num_actual_tokens = 0
        self.update(md, [], [], [])
        self.assertEqual(self.cache.calls, [])

    def test_nonspec_mixed_dispatch_uses_rebased_prefill_queries(self):
        self.impl.num_speculative_tokens = 0
        md = self.metadata([0, 1, 4], [1, 3], [5, 2], prefills=1, decode_tokens=1)
        self.update(md, [2, 10, 12, 14], [0, 16, 17, 18], [0, 0, 1, 2])
        self.assertEqual([call[0] for call in self.cache.calls], ["decode", "prefill"])
        self.assertEqual(self.groups[0][1].values, [0, 1, 2])
        self.assertEqual(self.groups[0][2].values, [0, 0, 0])
        self.assertEqual(self.groups[0][3].values, [0, 3])
        self.assertEqual(self.cache.mean, {0: 2, 2: 12})

    def test_nonspec_extra_request_cannot_launch_beyond_actual_decode_tokens(self):
        self.impl.num_speculative_tokens = 0
        # FULL/FIA metadata has an extra request. Only two actual tokens are
        # eligible to address the cache's two padded scratch pages P+row.
        md = self.metadata([0, 1, 2, 3], [1, 1, 0], [5, 2, -1], decode_tokens=3)
        md.num_actual_tokens = 2
        self.update(md, [2, 4, 999], [-1, -1, -1], [0, 0, 0])
        self.assertEqual(len(self.cache.calls), 1)
        self.assertEqual(self.cache.calls[0][1].values, [2, 4])
        self.assertEqual(self.cache.calls[0][3].numel(), 2)

    def test_summary_update_precedes_attention_and_has_no_preview_or_staged_path(self):
        md = self.metadata([0, 2], [2], [5], query_len=2)
        md.slot_mapping = Rows([0, 1])
        self.namespace["get_forward_context"] = lambda: types.SimpleNamespace(attn_metadata={"layer": md})
        self.impl._summary_cache = self.cache
        seen = []

        def attention(**kwargs):
            seen.append((self.cache.mean.copy(), len(self.cache.calls)))

        self.impl._forward_sparse_gqa_verify = attention
        tensors = [Rows([1, 3]) for _ in range(7)]
        self.impl.forward(
            types.SimpleNamespace(layer_name="layer"),
            tensors[0],
            [Rows([]), Rows([])],
            *tensors[1:6],
            Rows([0, 1]),
        )
        self.assertEqual(seen, [({0: 2}, 1)])
        tree = ast.parse(SOURCE.read_text())
        impl = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AscendStep5DSAImpl")
        names = {node.attr for node in ast.walk(impl) if isinstance(node, ast.Attribute)}
        names |= {node.id for node in ast.walk(impl) if isinstance(node, ast.Name)}
        self.assertTrue(
            names.isdisjoint({"csa_verify_preview_logits", "csa_verify_stage_update", "csa_deferred_flush"})
        )


if __name__ == "__main__":
    unittest.main()
