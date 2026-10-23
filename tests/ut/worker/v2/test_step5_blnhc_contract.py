"""CPU storage tests for the actual upstream planner, allocator and BLNHC views."""

import copy
import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch
from vllm.v1.attention.backend import MultipleOf
from vllm.v1.core.kv_cache_planning import get_kv_cache_config_from_groups, get_kv_cache_groups
from vllm.v1.kv_cache_interface import CircularBufferSpec, FullAttentionSpec, SlidingWindowSpec, UniformTypeKVCacheSpecs
from vllm.v1.kv_cache_layout import KVCacheLayout
from vllm.v1.worker.utils import allocate_kv_cache

ROOT = Path(__file__).resolve().parents[4]
module_spec = importlib.util.spec_from_file_location("step5_blnhc_storage", ROOT / "vllm_ascend/core/step5_kv_cache.py")
helper = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(helper)


def config():
    return NS(
        use_v2_model_runner=True,
        model_config=NS(hf_text_config=NS(model_type="step5")),
        kv_transfer_config=None,
        additional_config={},
        attention_config=NS(hisparse_config=None),
        parallel_config=NS(prefill_context_parallel_size=1, decode_context_parallel_size=1, pipeline_parallel_size=1),
        scheduler_config=NS(disable_hybrid_kv_cache_manager=False),
        cache_config=NS(
            get_resolved_kv_cache_layout=lambda: KVCacheLayout.BLNHC,
            num_gpu_blocks_override=None,
            prefix_cache_retention_interval=0,
        ),
    )


def specs():
    result = {}
    for i in range(2):
        result[f"model.layers.{i}.attn"] = FullAttentionSpec(
            block_size=128,
            num_kv_heads=1,
            head_size=192,
            head_size_v=192,
            dtype=torch.bfloat16,
            block_stride_alignment=384,
        )
        result[f"model.layers.{i}.attn.sparse_summary_cache"] = FullAttentionSpec(
            block_size=128,
            num_kv_heads=1,
            head_size=256,
            head_size_v=0,
            dtype=torch.uint8,
            tokens_per_state=8,
        )
        result[f"model.layers.{i}.attn.csa_tail_ring"] = CircularBufferSpec(
            block_size=128,
            num_kv_heads=1,
            head_size=384,
            head_size_v=0,
            dtype=torch.bfloat16,
            block_stride_alignment=1024,
        )
    for i in range(3, 9):
        result[f"model.layers.{i}.attn"] = SlidingWindowSpec(
            block_size=128,
            num_kv_heads=1,
            head_size=192,
            head_size_v=192,
            dtype=torch.bfloat16,
            sliding_window=512,
            extra_retained_tokens=3,
            block_stride_alignment=MultipleOf(128),
        )
    return result


def plan(num_blocks=4):
    conf = config()
    layers = specs()
    groups = get_kv_cache_groups(conf, layers)
    # Actual upstream planner yields same block tile for main+summary; no private budget formula.
    conf.cache_config.num_gpu_blocks_override = 1
    one = get_kv_cache_config_from_groups(conf, groups, 10 * 1024 * 1024)
    conf.cache_config.num_gpu_blocks_override = None
    assert one.num_blocks == 1
    stride = one.kv_cache_tensors[0].block_stride
    return conf, get_kv_cache_config_from_groups(conf, groups, num_blocks * stride)


class BLNHCStorageTests(unittest.TestCase):
    def allocate(self, blocks=4):
        conf, planned = plan(blocks)
        layers = helper.validate_step5_blnhc_descriptors(planned, KVCacheLayout.BLNHC)
        with patch.object(torch, "zeros", wraps=torch.zeros) as alloc:
            canonical = allocate_kv_cache(planned, "cpu", KVCacheLayout.BLNHC, [128] * len(planned.kv_cache_groups))
        self.assertEqual(alloc.call_count, 1)
        views = helper.step5_blnhc_layer_views(canonical, layers)
        return conf, planned, layers, canonical, views

    def test_real_summary_page_and_manager_ids_are_upstream(self):
        _, planned, layers, _, _ = self.allocate()
        full = [g for g in planned.kv_cache_groups if any(n.endswith("sparse_summary_cache") for n in g.layer_names)]
        self.assertEqual(len(full), 1)
        self.assertIsInstance(full[0].kv_cache_spec, UniformTypeKVCacheSpecs)
        self.assertEqual(len(full[0].layer_names), 4)
        for i in range(2):
            self.assertIn(f"model.layers.{i}.attn", full[0].layer_names)
            self.assertEqual(layers[f"model.layers.{i}.attn.sparse_summary_cache"].page_size_bytes, 4096)

    def test_budget_single_storage_and_zero_copy_kv_views(self):
        _, planned, _, canonical, views = self.allocate()
        unique = {t.untyped_storage().data_ptr(): t.untyped_storage().nbytes() for t in canonical.values()}
        self.assertEqual(len(unique), 1)
        self.assertEqual(sum(unique.values()), planned.kv_cache_tensors[0].size)
        self.assertEqual(planned.kv_cache_tensors[0].block_stride, 205824)
        name = "model.layers.0.attn"
        k, v = views[name]
        self.assertEqual(k.shape, (4, 128, 1, 192))
        self.assertEqual(k.stride(), (102912, 192, 192, 1))
        self.assertEqual(v.storage_offset() - k.storage_offset(), 24576)
        self.assertEqual(k.untyped_storage().data_ptr(), canonical[name].untyped_storage().data_ptr())
        k[1].fill_(3)
        v[1].fill_(5)
        self.assertTrue(torch.all(k[1] == 3))
        self.assertTrue(torch.all(v[1] == 5))

    def test_every_distinct_id_is_physical_tile_isolated(self):
        _, planned, _, canonical, _ = self.allocate()
        for block in range(4):
            for name, tensor in canonical.items():
                # Every group's complete layer span lies within this ID's tile.
                start = (
                    tensor.storage_offset() * tensor.element_size() + block * tensor.stride(0) * tensor.element_size()
                )
                span = tensor.element_size() + sum(
                    (s - 1) * d * tensor.element_size() for s, d in zip(tensor.shape[1:], tensor.stride()[1:])
                )
                self.assertGreaterEqual(start, block * 205824, name)
                self.assertLessEqual(start + span, (block + 1) * 205824, name)
        canonical["model.layers.0.attn"][1].fill_(7)
        for tensor in canonical.values():
            self.assertTrue(torch.all(tensor[0] == 0))
            self.assertTrue(torch.all(tensor[2] == 0))
            self.assertTrue(torch.all(tensor[3] == 0))

    def test_null_recycle_and_chained_cow_preserve_other_ids(self):
        _, planned, _, canonical, views = self.allocate()
        main = canonical["model.layers.0.attn"]
        summary = canonical["model.layers.0.attn.sparse_summary_cache"]
        main[1].fill_(2)
        summary[1].fill_(9)
        main[2].fill_(4)
        summary[2].fill_(11)
        # The production COW takes snapshots before any writes; exercise exactly that function.
        import ast

        source = ROOT / "vllm_ascend/worker/utils.py"
        tree = ast.parse(source.read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "copy_kv_cache_blocks_inplace")
        namespace = {"torch": torch}
        exec(
            compile(
                ast.fix_missing_locations(
                    ast.Module(
                        body=[
                            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                            node,
                        ],
                        type_ignores=[],
                    )
                ),
                str(source),
                "exec",
            ),
            namespace,
        )
        namespace["copy_kv_cache_blocks_inplace"](
            views.values(), 4, [NS(src_block_id=1, dst_block_id=2), NS(src_block_id=2, dst_block_id=3)]
        )
        self.assertTrue(torch.all(main[2] == 2))
        self.assertTrue(torch.all(summary[2] == 9))
        self.assertTrue(torch.all(main[3] == 4))
        self.assertTrue(torch.all(summary[3] == 11))
        raw = torch.empty(0, dtype=torch.int8).set_(main.untyped_storage()).view(4, -1)
        raw[2].zero_()
        self.assertTrue(torch.all(main[2] == 0))
        self.assertTrue(torch.all(summary[2] == 0))
        self.assertTrue(torch.all(main[1] == 2))
        self.assertTrue(torch.all(main[3] == 4))
        for tensor in canonical.values():
            self.assertTrue(torch.all(tensor[0] == 0))

    def test_rank_shrink_replans_exact_budget_and_offsets(self):
        for n in (1, 2, 7):
            _, planned, _, canonical, _ = self.allocate(n)
            self.assertEqual(planned.num_blocks, n)
            self.assertEqual(next(iter(canonical.values())).untyped_storage().nbytes(), 205824 * n)
            self.assertEqual({t.block_stride for t in planned.kv_cache_tensors}, {205824})

    def test_missing_duplicate_size_stride_offset_rejected_before_allocation(self):
        _, original = plan()
        mutations = [
            lambda c: c.kv_cache_tensors.pop(),
            lambda c: c.kv_cache_tensors.append(copy.deepcopy(c.kv_cache_tensors[0])),
            lambda c: setattr(c.kv_cache_tensors[0], "offset", -1),
            lambda c: setattr(c.kv_cache_tensors[0], "offset", 205824),
            lambda c: setattr(c.kv_cache_tensors[0], "size", 1),
            lambda c: setattr(c.kv_cache_tensors[0], "block_stride", 98304),
            lambda c: setattr(c.kv_cache_tensors[0], "layer_stride", 1),
        ]
        for change in mutations:
            candidate = copy.deepcopy(original)
            change(candidate)
            with (
                patch.object(torch, "zeros", side_effect=AssertionError("must reject before allocation")),
                self.assertRaises(ValueError),
            ):
                helper.validate_step5_blnhc_descriptors(candidate, KVCacheLayout.BLNHC)

    def test_unsupported_path_rejected_and_non_step5_unchanged(self):
        c = config()
        c.model_config.hf_text_config.model_type = "other"
        self.assertFalse(helper.uses_step5_blnhc(c))
        c = config()
        c.kv_transfer_config = object()
        with self.assertRaises(ValueError):
            helper.uses_step5_blnhc(c)
        _, planned = plan()
        with self.assertRaises(ValueError):
            helper.validate_step5_blnhc_descriptors(planned, KVCacheLayout.LBNHC)


if __name__ == "__main__":
    unittest.main()
