# SPDX-License-Identifier: Apache-2.0
"""Numeric regressions for actual DSA dispatch with scalar kernel stand-ins.

The stand-ins retain the real algorithms' distinct contracts: prefill folds
only FP32 prefix and current rows; decode/verify own the staged token tail.
Real Triton coverage lives in validation/verify_summary_state.py.
"""

import ast
import types
import unittest
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[4] / "vllm_ascend/models/step5/dsa_attention.py"


class Rows:
    device = "cpu"

    def __init__(self, values):
        self.values = list(values)

    def __getitem__(self, index):
        return Rows(self.values[index]) if isinstance(index, slice) else self.values[index]

    def numel(self):
        return len(self.values)

    def contiguous(self):
        return self


class Means(dict):
    def view(self, dtype):
        return self


class TestSummaryDispatch(unittest.TestCase):
    def test_verify_preview_reuses_provided_storage_and_bounds_window(self):
        kernel_source = SOURCE.with_name("summary_kernels.py")
        tree = ast.parse(kernel_source.read_text())
        function = next(
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "csa_verify_preview_logits"
        )
        launches = []

        class Kernel:
            def __getitem__(self, grid):
                return lambda *args, **kwargs: launches.append((grid, args, kwargs))

        class Buffer:
            shape = (2, 1, 1, 256)

            def numel(self):
                return 4

            def stride(self, axis):
                return 256 if axis == 0 else 1

        namespace = {
            "_csa_verify_preview_logits_kernel": Kernel(),
            "triton": types.SimpleNamespace(next_power_of_2=lambda value: value),
        }
        module = ast.Module(
            body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), function],
            type_ignores=[],
        )
        exec(compile(ast.fix_missing_locations(module), str(kernel_source), "exec"), namespace)
        inputs = [Buffer() for _ in range(5)]
        namespace[function.name](*inputs, self.cache, query_len=1)
        self.assertEqual(launches, [])
        # Only the active-ID buffer needs an exposed capacity in the wrapper.
        self.cache._step5_csa_active_region_ids = Buffer()
        for name in ("numerator_cache", "denominator_cache", "max_cache"):
            setattr(self.cache, "_step5_csa_" + name, Buffer())
        namespace[function.name](*inputs, self.cache, query_len=2)
        self.assertEqual(launches[0][0], (2, 1))
        self.assertEqual(launches[0][1][:5], tuple(inputs))
        self.assertEqual(launches[0][2]["QUERY_LEN"], 2)
        with self.assertRaises(ValueError):
            namespace[function.name](*inputs, self.cache, query_len=9)

    def test_deferred_flush_launches_for_single_token_verify_fallback(self):
        kernel_source = SOURCE.with_name("summary_kernels.py")
        tree = ast.parse(kernel_source.read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "csa_deferred_flush")
        launches = []

        class Kernel:
            def __getitem__(self, grid):
                return lambda *args, **kwargs: launches.append((grid, args, kwargs))

        class Buffer:
            shape = (1, 8, 1, 256)

            def numel(self):
                return 1

            def stride(self, axis):
                return 1

        namespace = {
            "_csa_deferred_flush_kernel": Kernel(),
            "triton": types.SimpleNamespace(next_power_of_2=lambda value: value),
        }
        module = ast.Module(
            body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), function],
            type_ignores=[],
        )
        exec(compile(ast.fix_missing_locations(module), str(kernel_source), "exec"), namespace)
        kwargs = dict(num_reqs=1, num_pages=4, summaries_per_page=2, region_block_size=8, page_size=16)
        namespace[function.name](*[Buffer() for _ in range(11)], query_len=1, **kwargs)
        self.assertEqual(len(launches), 1)
        self.assertEqual(launches[0][1][12], 1)
        namespace[function.name](*[Buffer() for _ in range(11)], query_len=0, **kwargs)
        self.assertEqual(len(launches), 1)

    def setUp(self):
        self.mean = Means()
        self.prefix = {}
        self.staging = {}
        self.calls = []
        names = {
            "_maybe_update_summary_cache",
            "_update_summary_cache_verify",
            "_flush_committed_decode_summary",
            "_round_up",
        }
        tree = ast.parse(SOURCE.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AscendStep5DSAImpl")
        cls.bases = []
        cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
        namespace = {
            "torch": types.SimpleNamespace(float8_e4m3fn="fp8", int32="i32"),
            "csa_compact_decode_update": self.decode,
            "csa_compact_prefill_update": self.prefill,
            "csa_deferred_flush": self.flush,
            "csa_verify_stage_update": self.verify,
        }
        module = ast.Module(
            body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), cls],
            type_ignores=[],
        )
        exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), namespace)
        self.impl = namespace[cls.name]()
        self.impl.num_speculative_tokens = 0
        self.impl.proxy_dim = 1
        self.impl.region_size = 8
        self.impl._begin_csa_allocation_check = lambda cache: object()
        self.impl._assert_csa_allocation_success = lambda success: None
        self.impl._get_dsa_tensor_buffer_at_least = lambda *args, **kwargs: object()
        self.cache = types.SimpleNamespace(
            mean_cache=self.mean, num_pages=4, summaries_per_page=2, region_block_size=8, page_size=16
        )
        for name in (
            "active_region_ids",
            "active_slot_by_region",
            "numerator_cache",
            "denominator_cache",
            "max_cache",
            "active_token_k",
            "active_token_z",
            "active_token_valid",
        ):
            setattr(self.cache, "_step5_csa_" + name, object())

    def update(self, positions, regions, values, decode_rows=0, prefills=0):
        layout = types.SimpleNamespace(
            token_flat_slot=Rows(regions),
            reset_slots=Rows([r if p % 8 == 0 else -1 for p, r in zip(positions, regions)]),
            token_valid=Rows([True] * len(values)),
            token_positions=Rows(positions),
        )
        self.impl._get_or_build_runtime_layout = lambda **kwargs: layout
        decode = (
            types.SimpleNamespace(
                query_len=decode_rows, seq_lens=Rows([positions[0] + decode_rows]), block_table=Rows([0])
            )
            if decode_rows
            else None
        )
        md = types.SimpleNamespace(
            num_actual_tokens=len(values),
            num_decodes=int(decode is not None),
            num_decode_tokens=decode_rows,
            num_prefills=prefills,
            decode=decode,
        )
        self.impl._maybe_update_summary_cache(
            summary_cache=self.cache, md=md, index_k=Rows(values), index_z=Rows([0] * len(values))
        )

    def prefill(self, *args, **kwargs):
        regions, positions, values = args[11].values, args[14].values, args[15].values
        self.calls.append(("prefill", positions))
        for region, position, value in zip(regions, positions, values):
            if position % 8 == 0:
                self.prefix[region] = []
            self.prefix.setdefault(region, []).append(value)
            if position % 8 == 7:
                # This kernel has no access to staging K/Z.
                prefix = self.prefix.pop(region)
                self.mean[region] = sum(prefix) / len(prefix)
                self.staging.pop(region, None)

    def decode(self, *args, **kwargs):
        regions, positions, values = args[10].values, args[13].values, args[14].values
        self.calls.append(("decode", positions))
        for region, position, value in zip(regions, positions, values):
            self.staging.setdefault(region, {})[position % 8] = value
            if position % 8 == 7:
                tail = self.prefix.pop(region, []) + list(self.staging.pop(region).values())
                self.mean[region] = sum(tail) / len(tail)

    def verify(self, *args, **kwargs):
        regions, positions, values = args[10].values, args[13].values, args[14].values
        self.calls.append(("verify", positions))
        for region, position, value in zip(regions, positions, values):
            self.staging.setdefault(region, {})[position % 8] = value

    def flush(self, *args, **kwargs):
        context = args[9][0] - kwargs["query_len"]
        self.calls.append(("flush", context))
        region = context // 8 - 1
        if region in self.staging:
            tail = self.prefix.pop(region, []) + list(self.staging.pop(region).values())
            self.mean[region] = sum(tail) / len(tail)

    def seed_decode_tail(self):
        self.update(list(range(5)), [0] * 5, list(range(1, 6)), prefills=1)
        for position in (5, 6):
            self.update([position], [0], [position + 1], decode_rows=1)

    def test_mixed_batch_preserves_previously_staged_accepted_tokens(self):
        self.seed_decode_tail()
        self.update([7, 0, 1], [0, 4, 4], [8, 21, 22], decode_rows=1, prefills=1)
        self.assertEqual(self.mean[0], 4.5)
        self.assertEqual(self.prefix[4], [21, 22])
        self.assertEqual(self.calls[-2:], [("decode", [7]), ("prefill", [0, 1])])

    def test_verify_four_to_one_flushes_committed_region_and_rewrites_rejection(self):
        self.impl.num_speculative_tokens = 3
        self.update(list(range(5)), [0] * 5, list(range(1, 6)), prefills=1)
        self.update([5, 6, 7, 8], [0, 0, 0, 1], [6, 7, 8, 999], decode_rows=4)
        self.assertNotIn(0, self.mean)
        self.update([8], [1], [9], decode_rows=1)
        self.assertEqual(self.mean[0], 4.5)
        self.assertEqual(self.staging[1], {0: 9})
        self.assertEqual(self.calls[-2:], [("flush", 8), ("decode", [8])])

    def test_mixed_verify_keeps_prefill_rows_out_of_provisional_staging(self):
        self.impl.num_speculative_tokens = 3
        self.update([5, 6, 7, 8, 0, 1], [0, 0, 0, 1, 4, 4], [6, 7, 8, 999, 21, 22], decode_rows=4, prefills=1)
        self.assertNotIn(4, self.staging)
        self.assertEqual(self.prefix[4], [21, 22])


if __name__ == "__main__":
    unittest.main()
