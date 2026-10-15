# SPDX-License-Identifier: Apache-2.0
"""Exercise real Step5 request-index preparation without NPU imports."""

import ast
import types
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

MODEL_DIR = Path(__file__).resolve().parents[4] / "vllm_ascend/models/step5"


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


class BaseState:
    def __init__(self, config, model, encoder_cache, device):
        self.max_num_reqs = config.scheduler_config.max_num_seqs
        self.preparations = []

    def prepare_attn(self, *args, **kwargs):
        builder = args[4][0][0].get_metadata_builder(0)
        # Verify injection has happened before the NPU-specific superclass.
        self.preparations.append((args, kwargs, builder.step5_request_state.tolist()))
        return {"npu_metadata": (args[0].seq_lens_np, args[0].attn_state)}


class TestStep5ModelState(unittest.TestCase):
    def setUp(self):
        self.builder_type = type("AscendStep5DSAMetadataBuilder", (), {})
        self.builder = self.builder_type()
        self.other_builder = types.SimpleNamespace()
        self.groups = [
            [
                types.SimpleNamespace(get_metadata_builder=lambda idx: self.builder),
                types.SimpleNamespace(get_metadata_builder=lambda idx: self.other_builder),
            ]
        ]
        backend_module = types.ModuleType("vllm_ascend.models.step5.dsa_attention")
        backend_module.AscendStep5DSAMetadataBuilder = self.builder_type
        self.module_patch = patch.dict("sys.modules", {backend_module.__name__: backend_module})
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)
        tree = ast.parse((MODEL_DIR / "model_state.py").read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
        namespace = {
            "AscendModelState": BaseState,
            "torch": types.SimpleNamespace(int64="int64", full=lambda shape, fill, **kwargs: Buffer([fill] * shape[0])),
        }
        module = ast.Module(
            body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), cls],
            type_ignores=[],
        )
        exec(compile(ast.fix_missing_locations(module), str(MODEL_DIR / "model_state.py"), "exec"), namespace)
        config = types.SimpleNamespace(scheduler_config=types.SimpleNamespace(max_num_seqs=6))
        self.state = namespace[cls.name](config, object(), None, "cpu")

    def prepare(self, mapping, *, padded=None, for_capture=False, ubatch_idx=0):
        batch = types.SimpleNamespace(
            num_reqs=len(mapping),
            num_reqs_after_padding=len(mapping) if padded is None else padded,
            idx_mapping=Buffer(mapping),
            seq_lens_np=[9] * len(mapping),
            attn_state="NPU_DECODE",
        )
        output = self.state.prepare_attn(
            batch,
            "FULL",
            ("blocks",),
            "slots",
            self.groups,
            "kv_config",
            for_capture=for_capture,
            ubatch_idx=ubatch_idx,
        )
        return batch, output

    def test_reorder_and_shrink_keep_persistent_indices_and_storage(self):
        allocation = self.state.request_state.storage
        self.prepare([5, 2, 4])
        self.assertEqual(self.state.request_state.tolist(), [5, 2, 4, -1, -1, -1, -1])
        self.prepare([2, 5])
        self.assertEqual(self.state.request_state.tolist(), [2, 5, -1, -1, -1, -1, -1])
        self.assertIs(self.state.request_state.storage, allocation)
        self.assertIs(self.builder.step5_request_state, self.state.request_state)
        self.assertFalse(hasattr(self.other_builder, "step5_request_state"))

    def test_extra_fia_request_is_minus_one_and_ascend_metadata_survives(self):
        batch, output = self.prepare([5, 4, 3, 2, 1, 0], padded=7, for_capture=True)
        self.assertEqual(self.state.request_state[-1], -1)
        self.assertEqual(output, {"npu_metadata": (batch.seq_lens_np, "NPU_DECODE")})
        args, kwargs, mapping = self.state.preparations[-1]
        self.assertEqual(args[1:], ("FULL", ("blocks",), "slots", self.groups, "kv_config"))
        self.assertEqual(kwargs, {"for_capture": True, "ubatch_idx": 0})
        self.assertEqual(mapping, [5, 4, 3, 2, 1, 0, -1])

    def test_request_state_index_reuse_does_not_create_a_second_allocator(self):
        self.prepare([5, 2])
        self.prepare([2])
        self.prepare([5])  # A new request can reuse upstream state index 5.
        self.assertEqual(self.builder.step5_request_state.tolist(), [5, -1, -1, -1, -1, -1, -1])
        # Journal invalidation remains in per-page cache lifecycle hooks.
        self.assertEqual(len(self.state.preparations), 3)

    def test_capacity_and_dbo_fail_before_superclass(self):
        with self.assertRaises(ValueError):
            self.prepare([0], padded=8)
        with self.assertRaises(ValueError):
            self.prepare(list(range(7)))
        with self.assertRaisesRegex(AssertionError, "DBO"):
            self.prepare([0], ubatch_idx=1)
        self.assertEqual(self.state.preparations, [])

    def test_both_model_frontends_choose_the_ascend_journal_state(self):
        state_module = types.ModuleType("vllm_ascend.models.step5.model_state")
        state_module.AscendStep5ModelState = type(self.state)
        for filename, name in (
            ("model.py", "AscendStep5ForCausalLM"),
            ("vl.py", "AscendStep5ForConditionalGeneration"),
        ):
            tree = ast.parse((MODEL_DIR / filename).read_text())
            cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)
            method = next(
                node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "get_model_state_cls"
            )
            method.decorator_list = []
            namespace = {}
            exec(compile(ast.Module(body=[method], type_ignores=[]), filename, "exec"), namespace)
            with patch.dict("sys.modules", {state_module.__name__: state_module}):
                self.assertIs(namespace[method.name](), type(self.state))


class TestStep5RequestMetadata(unittest.TestCase):
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
        self.builder.context_len_buffer = Buffer([0] * 5)
        self.builder.step5_request_state = Buffer([5, 2, 4, -1, -1])
        self.common = types.SimpleNamespace(
            query_start_loc=Buffer([0, 4, 8, 11, 11]),
            query_start_loc_cpu=Buffer([0, 4, 8, 11, 11]),
            seq_lens=Buffer([16, 24, 7, 0]),
            block_table_tensor=Buffer([10, 11, 12, -1]),
            num_reqs=4,
            num_actual_tokens=11,
            slot_mapping=Buffer(range(11)),
        )

    def test_decode_and_prefill_take_request_views_not_token_views(self):
        md = self.builder.build(0, self.common)
        self.assertEqual(md.step5_request_state.tolist(), [5, 2, 4, -1])
        self.assertEqual(md.decode.step5_request_state.tolist(), [5, 2])
        self.assertEqual(md.decode.query_len, 4)
        self.assertEqual(md.prefill.step5_request_state.tolist(), [4])
        self.assertEqual(md.prefill.query_start_loc.tolist(), [0, 3])
        self.assertEqual(md.prefill.context_lens.tolist(), [4])
        # A new step rewrites the persistent storage observed by all views.
        self.builder.step5_request_state[2:3].fill_(1)
        self.assertEqual(md.prefill.step5_request_state.tolist(), [1])

    def test_missing_state_stays_explicit_and_short_buffer_is_rejected(self):
        self.builder.step5_request_state = None
        md = self.builder.build(0, self.common)
        self.assertIsNone(md.step5_request_state)
        self.assertIsNone(md.decode.step5_request_state)
        self.assertIsNone(md.prefill.step5_request_state)
        self.builder.step5_request_state = Buffer([5, 2])
        with self.assertRaisesRegex(ValueError, "request count"):
            self.builder.build(0, self.common)


if __name__ == "__main__":
    unittest.main()
