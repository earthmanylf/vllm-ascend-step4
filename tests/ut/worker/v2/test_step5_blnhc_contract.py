"""CPU storage tests for the actual upstream planner, allocator and BLNHC views."""

import ast
import copy
import importlib
import importlib.util
import unittest
from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch
from vllm.v1.attention.backend import MultipleOf
from vllm.v1.core.kv_cache_planning import (
    get_kv_cache_config_from_groups,
    get_kv_cache_groups,
)
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.kv_cache_layout import KVCacheLayout
from vllm.v1.worker.utils import allocate_kv_cache

ROOT = Path(__file__).resolve().parents[4]
module_spec = importlib.util.spec_from_file_location("step5_blnhc_storage", ROOT / "vllm_ascend/core/step5_kv_cache.py")
helper = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(helper)


def config():
    return NS(
        use_v2_model_runner=True,
        model_config=NS(hf_text_config=NS(model_type="step5"), dtype=torch.bfloat16),
        kv_transfer_config=None,
        additional_config={},
        attention_config=NS(hisparse_config=None, indexer_kv_dtype="auto"),
        parallel_config=NS(
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            pipeline_parallel_size=1,
        ),
        scheduler_config=NS(disable_hybrid_kv_cache_manager=False),
        cache_config=NS(
            cache_dtype="auto",
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
            canonical = allocate_kv_cache(
                planned,
                "cpu",
                KVCacheLayout.BLNHC,
                [128] * len(planned.kv_cache_groups),
            )
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
            self.assertEqual(
                layers[f"model.layers.{i}.attn.sparse_summary_cache"].page_size_bytes,
                4096,
            )

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
                            ast.ImportFrom(
                                module="__future__",
                                names=[ast.alias(name="annotations")],
                                level=0,
                            ),
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
            views.values(),
            4,
            [NS(src_block_id=1, dst_block_id=2), NS(src_block_id=2, dst_block_id=3)],
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
                patch.object(
                    torch,
                    "zeros",
                    side_effect=AssertionError("must reject before allocation"),
                ),
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


class BLNHCFailClosedTests(unittest.TestCase):
    """Reject unsupported storage before allocation or native view creation."""

    def setUp(self):
        self.conf, self.planned = plan()
        self.layer_specs = helper.validate_step5_blnhc_descriptors(self.planned, KVCacheLayout.BLNHC)
        source = ROOT / "vllm_ascend/worker/v2/attn_utils.py"
        tree = ast.parse(source.read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "allocate_kv_cache_main")
        namespace = {
            "get_current_vllm_config": lambda: self.conf,
            "uses_step5_blnhc": helper.uses_step5_blnhc,
            "validate_step5_blnhc_descriptors": helper.validate_step5_blnhc_descriptors,
            "step5_blnhc_layer_views": helper.step5_blnhc_layer_views,
            "allocate_upstream_kv_cache": allocate_kv_cache,
        }
        module = ast.Module(
            body=[
                ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                node,
            ],
            type_ignores=[],
        )
        exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
        self.allocate_entrypoint = namespace["allocate_kv_cache_main"]
        self.names = [
            "model.layers.0.attn",
            "model.layers.0.attn.sparse_summary_cache",
            "model.layers.0.attn.csa_tail_ring",
            "model.layers.3.attn",
        ]

    def canonical(self):
        return allocate_kv_cache(
            self.planned,
            "cpu",
            KVCacheLayout.BLNHC,
            [128] * len(self.planned.kv_cache_groups),
        )

    def test_packed_overrides_rejected_before_upstream_allocation(self):
        for name in self.names:
            for field, value in (
                ("num_head_slots", 1),
                ("num_head_slots", 2),
                ("state_content_bytes", self.layer_specs[name].state_content_size_bytes),
                ("state_content_bytes", self.layer_specs[name].state_content_size_bytes // 2),
            ):
                with self.subTest(layer=name, field=field, value=value):
                    groups = []
                    for group in self.planned.kv_cache_groups:
                        if name not in group.layer_names:
                            groups.append(group)
                            continue
                        changed = dict(group.kv_cache_spec.kv_cache_specs)
                        changed[name] = replace(changed[name], **{field: value})
                        groups.append(
                            replace(group, kv_cache_spec=replace(group.kv_cache_spec, kv_cache_specs=changed))
                        )
                    planned = get_kv_cache_config_from_groups(self.conf, groups, 10 * 1024 * 1024)
                    with (
                        patch.dict(
                            self.allocate_entrypoint.__globals__,
                            allocate_upstream_kv_cache=unittest.mock.Mock(
                                side_effect=AssertionError("packed spec reached allocation"),
                            ),
                        ),
                        patch.object(torch, "zeros", side_effect=AssertionError("must reject before allocation")),
                        self.assertRaisesRegex(ValueError, "packed head slots or state content"),
                    ):
                        self.allocate_entrypoint(
                            planned,
                            "cpu",
                            KVCacheLayout.BLNHC,
                            [128] * len(groups),
                        )

    def test_dim0_alias_rejected_before_native_views_for_every_cache_type(self):
        canonical = self.canonical()
        for name in self.names:
            cache = canonical[name]
            page_elements = cache[0].numel()
            for stride in (0, 1, page_elements // 2, page_elements - 1):
                with self.subTest(layer=name, stride=stride):
                    bad = cache.as_strided(cache.shape, (stride, *cache.stride()[1:]))
                    with (
                        patch.object(
                            torch.Tensor, "as_strided", side_effect=AssertionError("bad cache reached native views")
                        ),
                        self.assertRaisesRegex(ValueError, "dense and non-overlapping"),
                    ):
                        helper.step5_blnhc_layer_views({name: bad}, {name: self.layer_specs[name]})

    def test_inner_stride_crossing_into_next_id_rejected(self):
        for name in self.names:
            with self.subTest(layer=name):
                spec = self.layer_specs[name]
                shape = (2, spec.num_kv_heads, spec.num_states, spec.head_size + spec.head_size_v)
                page_elements = shape[1] * shape[2] * shape[3]
                backing = torch.zeros(3 * page_elements, dtype=spec.dtype)
                bad = backing.as_strided(shape, (page_elements, shape[3], 2 * shape[3], 1))
                with self.assertRaisesRegex(ValueError, "dense and non-overlapping"):
                    helper.step5_blnhc_layer_views({name: bad}, {name: spec})

    def test_singleton_head_stride_is_not_a_false_alias(self):
        for name in self.names:
            spec = self.layer_specs[name]
            shape = (2, spec.num_kv_heads, spec.num_states, spec.head_size + spec.head_size_v)
            dense = torch.zeros(shape, dtype=spec.dtype)
            for head_stride in (dense.stride(1), 0):
                with self.subTest(layer=name, head_stride=head_stride):
                    strides = (dense.stride(0), head_stride, *dense.stride()[2:])
                    cache = dense.as_strided(shape, strides)
                    bound = helper.step5_blnhc_layer_views({name: cache}, {name: spec})[name]
                    tensors = bound if isinstance(bound, tuple) else (bound,)
                    for tensor in tensors:
                        self.assertEqual(tensor.untyped_storage().data_ptr(), dense.untyped_storage().data_ptr())
                    tensors[0][1].fill_(7)
                    self.assertTrue(torch.all(tensors[0][0] == 0))

    def test_legal_entrypoint_keeps_one_upstream_backing_and_page_geometry(self):
        with (
            patch.dict(
                self.allocate_entrypoint.__globals__,
                allocate_upstream_kv_cache=unittest.mock.Mock(wraps=allocate_kv_cache),
            ),
            patch.object(torch, "zeros", wraps=torch.zeros) as alloc,
        ):
            views = self.allocate_entrypoint(
                self.planned,
                "cpu",
                KVCacheLayout.BLNHC,
                [128] * len(self.planned.kv_cache_groups),
            )
            self.assertEqual(alloc.call_count, 1)
            self.allocate_entrypoint.__globals__["allocate_upstream_kv_cache"].assert_called_once()
        self.assertEqual(set(views), set(self.layer_specs))
        storage_sizes = {}
        for name, bound in views.items():
            tensors = bound if isinstance(bound, tuple) else (bound,)
            for tensor in tensors:
                self.assertEqual(
                    tensor.stride(0) * tensor.element_size(), self.planned.kv_cache_tensors[0].block_stride
                )
                storage_sizes[tensor.untyped_storage().data_ptr()] = tensor.untyped_storage().nbytes()
            if name.endswith("sparse_summary_cache"):
                self.assertEqual(self.layer_specs[name].page_size_bytes, 4096)
                self.assertEqual(self.layer_specs[name].tokens_per_state, 8)
                self.assertIsNone(self.layer_specs[name].page_size_padded)
        self.assertEqual(len(storage_sizes), 1)
        self.assertEqual(sum(storage_sizes.values()), self.planned.kv_cache_tensors[0].size)

    def test_full_layer_counts_with_and_without_mtp_keep_upstream_tiles(self):
        main, summary, ring, swa = (self.layer_specs[name] for name in self.names)
        for mtp, expected_stride in ((False, 2356224), (True, 2359296)):
            with self.subTest(mtp=mtp):
                layers = {}
                for index in range(95 if mtp else 92):
                    name = f"model.layers.{index}.attn"
                    if index % 4 != 3:
                        layers[name] = swa
                    else:
                        layers[name] = main
                        layers[name + ".sparse_summary_cache"] = summary
                        layers[name + ".csa_tail_ring"] = ring
                groups = get_kv_cache_groups(self.conf, layers)
                self.conf.cache_config.num_gpu_blocks_override = 4
                try:
                    planned = get_kv_cache_config_from_groups(self.conf, groups, 64 * 1024 * 1024)
                finally:
                    self.conf.cache_config.num_gpu_blocks_override = None
                with patch.object(torch, "zeros", wraps=torch.zeros) as alloc:
                    bound = self.allocate_entrypoint(planned, "cpu", KVCacheLayout.BLNHC, [128] * len(groups))
                self.assertEqual(alloc.call_count, 1)
                self.assertEqual(set(bound), set(layers))
                self.assertEqual({d.block_stride for d in planned.kv_cache_tensors}, {expected_stride})
                for value in bound.values():
                    for tensor in value if isinstance(value, tuple) else (value,):
                        self.assertEqual(tensor.shape[0], planned.num_blocks)
                        self.assertEqual(tensor.stride(0) * tensor.element_size(), expected_stride)


class BLNHCCollectorTests(unittest.TestCase):
    """A missing backend hook must not reach native GBSA with illegal strides."""

    def collector(self, layers, *, customize=True):
        from vllm.config import set_current_vllm_config
        from vllm.v1.worker.utils import customize_attention_spec

        base = importlib.import_module("vllm_ascend.worker.v2.attn_utils")
        source = ROOT / "vllm_ascend/worker/v2/attn_utils.py"
        tree = ast.parse(source.read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "get_kv_cache_spec")
        namespace = dict(base.__dict__)
        namespace.update(
            fields=fields,
            get_layers_from_vllm_config=lambda *args: layers,
            enable_sfa_dcp_replicated_indexer=lambda *args: False,
            is_c8_mxfp_kv_quant=lambda *args: False,
            kv_cache_dtype_str_to_dtype=lambda *args: torch.bfloat16,
            customize_attention_spec=customize_attention_spec,
            set_current_vllm_config=set_current_vllm_config,
            uses_step5_blnhc=helper.uses_step5_blnhc if customize else lambda conf: False,
        )
        module = ast.Module(
            body=[
                ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                function,
            ],
            type_ignores=[],
        )
        exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
        return namespace["get_kv_cache_spec"]

    def layers(self, mtp):
        from vllm.models.step5.nvidia.csa_tail_ring import Step5CSATailRingBackend

        from vllm_ascend.attention.attention_v1 import AscendAttentionBackend
        from vllm_ascend.models.step5.dsa_attention import AscendStep5DSABackend
        from vllm_ascend.models.step5.summary_cache import (
            AscendStep5SparseSummaryBackend,
        )

        result = {}

        def add(name, spec, backend):
            result[name] = NS(
                get_kv_cache_spec=lambda conf, s=spec: s,
                get_attn_backend=lambda b=backend: b,
                is_draft_layer=int(name.split(".")[2]) >= 92,
            )

        for index in range(92 + (3 if mtp else 0)):
            name = f"model.layers.{index}.attn"
            if index % 4 != 3:
                add(
                    name,
                    SlidingWindowSpec(
                        block_size=128,
                        num_kv_heads=1,
                        head_size=192,
                        head_size_v=192,
                        dtype=torch.bfloat16,
                        sliding_window=512,
                        extra_retained_tokens=3,
                    ),
                    AscendAttentionBackend,
                )
                continue
            add(
                name,
                FullAttentionSpec(
                    block_size=128,
                    num_kv_heads=1,
                    head_size=192,
                    head_size_v=192,
                    dtype=torch.bfloat16,
                ),
                AscendStep5DSABackend,
            )
            add(
                name + ".sparse_summary_cache",
                FullAttentionSpec(
                    block_size=128,
                    num_kv_heads=1,
                    head_size=256,
                    head_size_v=0,
                    dtype=torch.uint8,
                    tokens_per_state=8,
                ),
                AscendStep5SparseSummaryBackend,
            )
            add(
                name + ".csa_tail_ring",
                CircularBufferSpec(
                    block_size=128,
                    num_kv_heads=1,
                    head_size=384,
                    head_size_v=0,
                    dtype=torch.bfloat16,
                    block_stride_alignment=1024,
                ),
                Step5CSATailRingBackend,
            )
        return result

    def test_draft_marker_survives_collector_without_changing_grouping(self):
        conf = config()
        layers = self.layers(True)
        draft_names = {name for name, layer in layers.items() if layer.is_draft_layer}
        self.assertEqual(len(draft_names), 3)
        for customize in (False, True):
            with self.subTest(customize=customize):
                collected = self.collector(layers, customize=customize)(conf)
                self.assertEqual({name for name, spec in collected.items() if spec.is_draft}, draft_names)
                unmarked = {name: replace(spec, is_draft=False) for name, spec in collected.items()}
                self.assertEqual(get_kv_cache_groups(conf, collected), get_kv_cache_groups(conf, unmarked))

    def test_collector_allows_modules_without_draft_marker(self):
        conf = config()
        layers = self.layers(True)
        for layer in layers.values():
            del layer.is_draft_layer
        collected = self.collector(layers)(conf)
        self.assertFalse(any(spec.is_draft for spec in collected.values()))

    def test_backend_alignment_survives_collector_and_real_planner(self):
        from vllm.config import get_current_vllm_config_or_none, set_current_vllm_config

        for mtp in (False, True):
            with self.subTest(mtp=mtp):
                conf = config()
                layers = self.layers(mtp)
                previous = get_current_vllm_config_or_none()
                original_specs = self.collector(layers, customize=False)(conf)
                fixed_specs = self.collector(layers)(conf)
                self.assertIs(get_current_vllm_config_or_none(), previous)
                dsa = [n for n in layers if n.endswith(".attn") and int(n.split(".")[2]) % 4 == 3]
                self.assertTrue(all(original_specs[n].block_stride_alignment is None for n in dsa))
                self.assertTrue(all(fixed_specs[n].block_stride_alignment == 384 for n in dsa))
                summary = [s for n, s in fixed_specs.items() if n.endswith("sparse_summary_cache")]
                self.assertTrue(all(s.page_size_bytes == 4096 and s.page_size_padded is None for s in summary))
                groups = get_kv_cache_groups(conf, fixed_specs)
                planned = get_kv_cache_config_from_groups(conf, groups, 24 * 2359296)
                self.assertEqual(
                    planned.kv_cache_tensors[0].block_stride,
                    2359296 if mtp else 2356224,
                )
                bound_specs = helper.validate_step5_blnhc_descriptors(planned, KVCacheLayout.BLNHC)
                canonical = allocate_kv_cache(planned, "cpu", KVCacheLayout.BLNHC, [128] * len(groups))
                views = helper.step5_blnhc_layer_views(canonical, bound_specs)
                self.assertTrue(all(views[n][0].stride(0) % 192 == 0 for n in dsa))
                old_groups = get_kv_cache_groups(conf, original_specs)
                invalid = get_kv_cache_config_from_groups(conf, old_groups, 24 * 2359296)
                self.assertEqual(
                    invalid.kv_cache_tensors[0].block_stride,
                    2359296 if mtp else 2355200,
                )
                for descriptor in invalid.kv_cache_tensors:
                    descriptor.block_stride = 2355200
                    descriptor.size = invalid.num_blocks * 2355200
                with self.assertRaisesRegex(ValueError, "native row alignment 384 bytes"):
                    helper.validate_step5_blnhc_descriptors(invalid, KVCacheLayout.BLNHC)
                outer = config()
                outer.model_config.hf_text_config.model_type = "other_model"
                with set_current_vllm_config(outer):
                    self.collector(layers)(conf)
                    self.assertIs(get_current_vllm_config_or_none(), outer)

    def test_collector_does_not_customize_non_target_models_or_legacy_runner(self):
        layers = self.layers(False)
        for excluded in ("non_step5", "non_v2"):
            with self.subTest(excluded=excluded):
                conf = config()
                if excluded == "non_step5":
                    conf.model_config.hf_text_config.model_type = "other_model"
                else:
                    conf.use_v2_model_runner = False
                collector = self.collector(layers)
                original = self.collector(layers, customize=False)
                with patch.dict(
                    collector.__globals__,
                    customize_attention_spec=unittest.mock.Mock(
                        side_effect=AssertionError("excluded paths must retain the old collector")
                    ),
                ):
                    self.assertEqual(collector(conf), original(conf))


if __name__ == "__main__":
    unittest.main()
