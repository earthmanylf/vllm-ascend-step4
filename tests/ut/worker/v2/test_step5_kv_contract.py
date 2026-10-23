"""CPU behavioral checks using real torch storage and the exact candidate functions.

Only vLLM spec/config containers and unrelated device backends are doubles.
Run: python -m pytest tests/ut/worker/v2/test_step5_shared_kv.py
"""

import ast
import copy
import importlib.util
import os
import sys
import unittest
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[4]


def vllm_source_file(relative):
    source_root = os.environ.get("VLLM_SOURCE_ROOT")
    if source_root:
        return Path(source_root) / "vllm" / relative
    installed = importlib.util.find_spec("vllm")
    if installed is None or installed.origin is None:
        raise RuntimeError("Install vLLM or set VLLM_SOURCE_ROOT to its source repository.")
    return Path(installed.origin).parent / relative


class AttentionSpec:
    def __init__(self, **kwargs):
        self.block_size = 128
        self.num_kv_heads = 1
        self.head_size = self.head_size_v = 192
        self.dtype = torch.bfloat16
        self.page_size_padded = None
        self.extra_budget_page_size_bytes = 512
        self.extra_budget_fixed_size_bytes = 1024
        self.requires_zeroing = True
        self.__dict__.update(kwargs)

    @property
    def page_size_bytes(self):
        return self.page_size_padded or (
            self.block_size * self.num_kv_heads * (self.head_size + self.head_size_v) * self.dtype.itemsize
        )


class FullAttentionSpec(AttentionSpec):
    pass


class SlidingWindowSpec(AttentionSpec):
    pass


class CircularBufferSpec(AttentionSpec):
    pass


class UniformTypeKVCacheSpecs:
    def __init__(self, specs):
        self.kv_cache_specs = specs


stub = ModuleType("vllm.v1.kv_cache_interface")
stub.FullAttentionSpec = FullAttentionSpec
stub.CircularBufferSpec = CircularBufferSpec
stub.SlidingWindowSpec = SlidingWindowSpec
stub.UniformTypeKVCacheSpecs = UniformTypeKVCacheSpecs
module_spec = importlib.util.spec_from_file_location(
    "step5_shared_candidate", ROOT / "vllm_ascend/core/step5_kv_cache.py"
)
helper = importlib.util.module_from_spec(module_spec)
with patch.dict(sys.modules, {"vllm.v1.kv_cache_interface": stub}):
    module_spec.loader.exec_module(helper)


def config():
    layout = NS(is_layer_compact=True, is_block_compact=True)
    return NS(
        use_v2_model_runner=True,
        model_config=NS(hf_text_config=NS(model_type="step5")),
        kv_transfer_config=None,
        additional_config={},
        parallel_config=NS(decode_context_parallel_size=1, prefill_context_parallel_size=1),
        cache_config=NS(get_resolved_kv_cache_layout=lambda: layout),
    )


def caches():
    groups = [
        NS(layer_names=["dsa0", "dsa1"], kv_cache_spec=FullAttentionSpec()),
        NS(
            layer_names=["swa0", "swa1"],
            kv_cache_spec=SlidingWindowSpec(extra_budget_page_size_bytes=0),
        ),
        NS(layer_names=["mtp"], kv_cache_spec=FullAttentionSpec()),
    ]
    num_blocks = 4
    page = groups[0].kv_cache_spec.page_size_bytes
    layer_size = num_blocks * page
    descriptors = [
        NS(
            size=2 * layer_size,
            layers=g.layer_names,
            offset=0,
            layer_stride=layer_size,
            block_stride=page,
        )
        for g in groups
    ]
    return NS(
        kv_cache_groups=groups,
        kv_cache_tensors=descriptors,
        num_blocks=num_blocks,
        needs_base_kv_cache_zeroing=False,
    )


def extract(path, name, globals_dict):
    tree = ast.parse(path.read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name)
    exec(  # noqa: S102 - execute only the named function from the reviewed local source
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
            str(path),
            "exec",
        ),
        globals_dict,
    )
    return globals_dict[name]


def shaped(raw, cfg):
    dims = (cfg.num_blocks, 128, 1, 192)
    return {name: tuple(part.view(torch.bfloat16).view(dims) for part in pair) for name, pair in raw.items()}


def sidecar_view(name, spec, raw, blocks, *, is_ring=False):
    namespace = dict(torch=torch, get_dtype_size=lambda dtype: dtype.itemsize)
    source = ROOT / "vllm_ascend/worker/utils.py"
    namespace["row_major_strides"] = extract(source, "row_major_strides", {})
    extract(source, "make_page_strided_cache_view", namespace)
    view = extract(ROOT / "vllm_ascend/worker/v2/attn_utils.py", "view_step5_sidecar_cache", namespace)
    return view(name, spec, raw, blocks, is_ring=is_ring)


class SharedCacheTests(unittest.TestCase):
    def test_tailring_shared_backing_uses_descriptor_offsets_and_one_allocation(self):
        page = 98304
        specs = {
            "dsa": FullAttentionSpec(),
            "summary": FullAttentionSpec(
                head_size=256, head_size_v=0, dtype=torch.uint8, tokens_per_state=8, page_size_padded=page
            ),
            "ring": CircularBufferSpec(head_size=384, head_size_v=0),
        }
        groups = [NS(layer_names=[name], kv_cache_spec=spec) for name, spec in specs.items()]
        kv = NS(
            num_blocks=4,
            kv_cache_groups=groups,
            kv_cache_tensors=[
                NS(
                    size=4 * page,
                    layers=[name],
                    offset=0,
                    layer_stride=4 * page,
                    block_stride=page,
                    host_resident=False,
                )
                for name in specs
            ],
        )
        self.assertTrue(helper.supports_step5_tailring_shared_backing(config(), groups))
        with patch.object(helper.torch, "zeros", wraps=torch.zeros) as allocation:
            caches = helper.allocate_step5_tailring_shared_backing(kv, "cpu")
        self.assertEqual(allocation.call_count, 1)
        k, v = caches["dsa"]
        self.assertEqual(k.stride(0), page // 2)
        self.assertEqual(caches["summary"].shape, (4, 1, 16, 256))
        self.assertEqual(caches["ring"].shape, (4, 1, 128, 384))
        k[1].fill_(7)
        v[1].fill_(9)
        caches["summary"][2].fill_(11)
        caches["ring"][3].fill_(13)
        self.assertTrue(torch.all(k[1] == 7))
        self.assertTrue(torch.all(v[1] == 9))
        self.assertTrue(torch.all(caches["summary"][2] == 11))
        self.assertTrue(torch.all(caches["ring"][3] == 13))
        self.assertEqual(k.untyped_storage().nbytes(), 4 * page)
        env = dict(
            get_current_vllm_config=config,
            uses_step5_blnhc=lambda _: False,
            supports_step5_tailring_shared_backing=helper.supports_step5_tailring_shared_backing,
            allocate_step5_tailring_shared_backing=helper.allocate_step5_tailring_shared_backing,
        )
        allocate = extract(ROOT / "vllm_ascend/worker/v2/attn_utils.py", "allocate_kv_cache_main", env)
        allocated = allocate(kv, "cpu", None, [128] * 3)
        self.assertEqual(allocated["summary"].shape, (4, 1, 16, 256))
        env.update(get_kv_cache_groups=lambda *_: groups, UniformTypeKVCacheSpecs=UniformTypeKVCacheSpecs)
        scale = extract(ROOT / "vllm_ascend/worker/worker.py", "_scale_kv_cache_memory_for_multi_group", env)
        worker = NS(vllm_config=config(), get_kv_cache_spec=lambda: specs)
        self.assertEqual(scale(worker, 123456), 123456)
        for mutate in (
            lambda c: setattr(c.kv_cache_tensors[0], "offset", -page),
            lambda c: setattr(c.kv_cache_tensors[0], "block_stride", page // 2),
            lambda c: setattr(c.kv_cache_tensors[0], "host_resident", True),
        ):
            invalid = copy.deepcopy(kv)
            mutate(invalid)
            with (
                patch.object(helper.torch, "zeros", side_effect=AssertionError("validate before allocating")),
                self.assertRaises(ValueError),
            ):
                helper.allocate_step5_tailring_shared_backing(invalid, "cpu")

    def test_step5_summary_page_view_preserves_padding_and_block_ids(self):
        spec = NS(
            dtype=torch.uint8,
            num_heads=1,
            num_states=16,
            state_content_size_bytes=256,
            page_size_bytes=8192,
        )
        raw = torch.full((3 * spec.page_size_bytes,), 91, dtype=torch.int8)
        view = sidecar_view("summary", spec, raw, 3)
        self.assertEqual(view.shape, (3, 1, 16, 256))
        self.assertEqual(view.stride(0), spec.page_size_bytes)
        view[1].fill_(12)
        self.assertTrue(torch.all(raw[:8192] == 91))
        self.assertTrue(torch.all(raw[8192:12288] == 12))
        self.assertTrue(torch.all(raw[12288:] == 91))

    def test_step5_ring_view_uses_typed_page_stride_and_rejects_short_storage(self):
        spec = NS(
            dtype=torch.bfloat16,
            num_heads=1,
            num_states=128,
            state_content_size_bytes=768,
            page_size_bytes=98304,
        )
        raw = torch.zeros(3 * spec.page_size_bytes, dtype=torch.int8)
        view = sidecar_view("ring", spec, raw, 3)
        self.assertEqual(view.shape, (3, 1, 128, 384))
        self.assertEqual(view.stride(0), 49152)
        self.assertEqual(view[0].numel() // 512, 96)
        with self.assertRaises(ValueError):
            sidecar_view("ring", spec, raw[:-1], 3)
        with self.assertRaises(ValueError):
            sidecar_view("ring", spec, raw[:98304], 3)

    def test_reshape_step5_sidecar_does_not_require_backend_shape_hook(self):
        spec = FullAttentionSpec(
            head_size=256,
            head_size_v=0,
            dtype=torch.uint8,
            num_heads=1,
            num_states=16,
            state_content_size_bytes=256,
            page_size_padded=8192,
        )
        kv = NS(num_blocks=3, kv_cache_groups=[NS(layer_names=["summary"])])
        backend = NS(get_name=lambda: "STEP5_CSA_SUMMARY")
        group = NS(kv_cache_group_id=0, kv_cache_spec=spec, layer_names=["summary"], backend=backend)
        unused = type("UnusedSpec", (), {})
        env = dict(
            torch=torch,
            get_current_vllm_config=config,
            uses_step5_blnhc=lambda _: False,
            _is_dsv4_model=lambda _: False,
            is_deepseek_v41_cache=lambda _: False,
            _get_layer_kv_cache_specs=lambda _: {"summary": spec},
            get_storage_block_size=lambda s: s.block_size,
            FullAttentionSpec=FullAttentionSpec,
            AttentionSpec=AttentionSpec,
            is_c8_mxfp_kv_quant=lambda _: False,
            is_hidden_state_cache_spec=lambda _: False,
            AscendSFAIndexerCacheSpec=unused,
            AscendIndexerKPoolTailSpec=unused,
            AscendMLAAttentionSpec=unused,
            AscendSlidingWindowMLASpec=unused,
            MambaSpec=unused,
            get_single_raw_mla_backing=lambda _: None,
            view_step5_sidecar_cache=sidecar_view,
        )
        reshape = extract(ROOT / "vllm_ascend/worker/v2/attn_utils.py", "_reshape_kv_cache_v2", env)
        raw = torch.zeros(3 * spec.page_size_bytes, dtype=torch.int8)
        cache = reshape([group], {"summary": raw}, "auto", [128], {}, kv)["summary"]
        self.assertIsInstance(cache, torch.Tensor)
        self.assertEqual(cache.shape, (3, 1, 16, 256))
        self.assertEqual(cache.stride(0), 8192)

    def test_allocator_step5_sidecars_use_one_tensor_per_layer(self):
        spec = AttentionSpec(head_size=256, head_size_v=0, dtype=torch.uint8)
        specs = {"summary": spec, "ring": spec}
        cfg = config()
        layers = {name: NS(get_attn_backend=lambda: NS(get_name=lambda: "STEP5_CSA_SUMMARY")) for name in specs}
        descriptor = NS(layers=list(specs), layer_stride=spec.page_size_bytes * 3)
        kv = NS(
            num_blocks=3,
            kv_cache_groups=[NS(layer_names=list(specs))],
            kv_cache_tensors=[descriptor],
        )
        unused = type("UnusedSpec", (), {})
        env = dict(
            torch=torch,
            get_current_vllm_config=lambda: cfg,
            KVPPConfig=NS(from_vllm_config=lambda _: NS(size=1)),
            supports_step5_shared_kv_backing=lambda *_: False,
            _is_dsv4_model=lambda _: False,
            _get_layer_kv_cache_specs=lambda _: specs,
            is_deepseek_v41_cache=lambda _: False,
            MambaSpec=unused,
            AttentionSpec=AttentionSpec,
            AscendSFAIndexerCacheSpec=unused,
            AscendIndexerKPoolTailSpec=unused,
            get_layerwise_reuse_config=lambda _: None,
            get_kv_cache_tensor_layers=lambda d: d.layers,
            get_layers_from_vllm_config=lambda *_: layers,
            AttentionLayerBase=object,
            _allocate_int8_cache_tensor=lambda size, _, device: torch.zeros(size, dtype=torch.int8, device=device),
            is_hidden_state_cache_spec=lambda _: False,
            _uses_single_raw_mla_cache=lambda *_: False,
            enable_sfa=lambda _: False,
            FullAttentionSpec=FullAttentionSpec,
            is_c8_mxfp_kv_quant=lambda _: False,
            requires_contiguous_pa_kv_cache=lambda *_: False,
            _get_attention_kv_cache_dims=lambda *_: (256, 0),
            enable_fa_quant=lambda _: False,
            calc_split_factor=lambda _: (_ for _ in ()).throw(ZeroDivisionError("zero V must not be split")),
        )
        allocator = extract(ROOT / "vllm_ascend/worker/v2/attn_utils.py", "_allocate_kv_cache", env)
        raw = allocator(kv, {}, torch.device("cpu"))
        for cache in raw.values():
            self.assertIsInstance(cache, torch.Tensor)
            self.assertEqual(cache.numel(), 3 * spec.page_size_bytes)
        self.assertNotEqual(raw["summary"].data_ptr(), raw["ring"].data_ptr())

    def test_real_ring_binding_addresses_each_manager_page(self):
        spec = NS(
            dtype=torch.bfloat16,
            num_heads=1,
            num_states=128,
            state_content_size_bytes=768,
            page_size_bytes=98304,
        )
        raw = torch.zeros(3 * spec.page_size_bytes, dtype=torch.int8)
        cache = sidecar_view("ring", spec, raw, 3)
        binder = extract(
            vllm_source_file("models/step5/nvidia/csa_tail_ring.py"),
            "bind_kv_cache",
            {"torch": torch, "_ROW_ELEMENTS": 512},
        )
        ring = NS()
        binder(ring, cache)
        self.assertEqual(ring.capacity, 96)
        self.assertEqual(ring.rows_per_block, 96)
        self.assertEqual(ring.rows.shape, (288, 512))
        ring.rows[96].fill_(17)
        self.assertTrue(torch.all(cache[1].flatten()[:512] == 17))
        self.assertTrue(torch.all(cache[0] == 0))
        self.assertTrue(torch.all(cache[2] == 0))

    def test_ring_padding_is_rejected_before_metadata_capacity_can_diverge(self):
        spec = NS(
            dtype=torch.bfloat16,
            num_heads=1,
            num_states=128,
            state_content_size_bytes=768,
            page_size_bytes=196608,
        )
        raw = torch.zeros(3 * spec.page_size_bytes, dtype=torch.int8)
        with self.assertRaisesRegex(ValueError, "ring.*padding"):
            sidecar_view("ring", spec, raw, 3, is_ring=True)

    def test_sidecar_filter_keeps_original_descriptor_layer_offsets(self):
        class StateSpec:
            page_size_bytes = 98304

        dense = FullAttentionSpec()
        summary = AttentionSpec(head_size=256, head_size_v=0, dtype=torch.uint8)
        specs = {"summary": summary, "attention": dense, "state": StateSpec()}
        size = dense.page_size_bytes * 3
        descriptors = [
            NS(
                layers=["summary", "attention"],
                size=2 * size,
                offset=0,
                layer_stride=size,
                block_stride=dense.page_size_bytes,
            ),
            NS(layers=["state"], size=2 * size, offset=0, layer_stride=size, block_stride=dense.page_size_bytes),
        ]
        cfg = config()
        layers = {
            "summary": NS(get_attn_backend=lambda: NS(get_name=lambda: "STEP5_CSA_SUMMARY")),
            "attention": NS(get_attn_backend=lambda: NS(get_name=lambda: "ASCEND")),
            "state": NS(get_attn_backend=lambda: NS(get_name=lambda: "MAMBA")),
        }
        kv = NS(num_blocks=3, kv_cache_tensors=descriptors, kv_cache_groups=[NS(layer_names=list(specs))])
        unused = type("UnusedSpec", (), {})
        env = dict(
            torch=torch,
            get_current_vllm_config=lambda: cfg,
            KVPPConfig=NS(from_vllm_config=lambda _: NS(size=1)),
            supports_step5_shared_kv_backing=lambda *_: False,
            _is_dsv4_model=lambda _: False,
            _get_layer_kv_cache_specs=lambda _: specs,
            is_deepseek_v41_cache=lambda _: False,
            MambaSpec=StateSpec,
            AttentionSpec=AttentionSpec,
            AscendSFAIndexerCacheSpec=unused,
            AscendIndexerKPoolTailSpec=unused,
            AttentionLayerBase=object,
            get_layerwise_reuse_config=lambda _: None,
            get_kv_cache_tensor_layers=lambda d: d.layers,
            get_layers_from_vllm_config=lambda *_: layers,
            _allocate_int8_cache_tensor=lambda count, _, device: torch.zeros(count, dtype=torch.int8, device=device),
            is_hidden_state_cache_spec=lambda _: False,
        )
        allocator = extract(ROOT / "vllm_ascend/worker/v2/attn_utils.py", "_allocate_kv_cache", env)
        raw = allocator(kv, {}, torch.device("cpu"))
        self.assertEqual(raw["attention"].data_ptr() - raw["state"].data_ptr(), size)
        raw["attention"].fill_(19)
        self.assertTrue(torch.all(raw["state"] == 0))
        self.assertTrue(torch.all(raw["summary"] == 0))

    def test_budget_and_allocator_use_same_predicate(self):
        cfg = config()
        kv = caches()
        env = {
            "supports_step5_shared_kv_backing": helper.supports_step5_shared_kv_backing,
            "allocate_step5_shared_kv_backing": helper.allocate_step5_shared_kv_backing,
            "get_current_vllm_config": lambda: cfg,
            "KVPPConfig": NS(from_vllm_config=lambda _: NS(size=1)),
        }
        allocator = extract(ROOT / "vllm_ascend/worker/v2/attn_utils.py", "_allocate_kv_cache", env)
        raw = allocator(kv, {}, torch.device("cpu"))
        self.assertEqual(
            len({x.untyped_storage().data_ptr() for pair in raw.values() for x in pair}),
            1,
        )
        env.update(
            get_kv_cache_groups=lambda *_: kv.kv_cache_groups,
            uses_step5_blnhc=lambda _: False,
            supports_step5_tailring_shared_backing=helper.supports_step5_tailring_shared_backing,
            UniformTypeKVCacheSpecs=UniformTypeKVCacheSpecs,
        )
        budget = extract(
            ROOT / "vllm_ascend/worker/worker.py",
            "_scale_kv_cache_memory_for_multi_group",
            env,
        )
        worker = NS(vllm_config=cfg, get_kv_cache_spec=lambda: {"x": 1})
        self.assertEqual(budget(worker, 1234567), 1234567)

    def test_cross_group_different_blocks_are_isolated_and_planes_contiguous(self):
        kv = caches()
        raw = helper.allocate_step5_shared_kv_backing(kv, "cpu")
        views = shaped(raw, kv)
        for pair in views.values():
            self.assertTrue(all(x.is_contiguous() for x in pair))
        views["dsa0"][0][1].fill_(11)
        views["dsa0"][1][1].fill_(12)
        views["swa0"][0][2].fill_(21)
        views["swa0"][1][2].fill_(22)
        views["mtp"][0][3].fill_(31)
        views["mtp"][1][3].fill_(32)
        self.assertTrue(torch.all(views["dsa0"][0][1] == 11))
        self.assertTrue(torch.all(views["dsa0"][1][1] == 12))
        self.assertTrue(torch.all(views["swa0"][0][2] == 21))
        self.assertTrue(torch.all(views["swa0"][1][2] == 22))
        self.assertTrue(torch.all(views["mtp"][0][3] == 31))
        self.assertTrue(torch.all(views["mtp"][1][3] == 32))
        self.assertTrue(torch.all(views["dsa1"][0] == 0))
        self.assertTrue(torch.all(views["swa1"][1] == 0))
        self.assertEqual(raw["dsa0"][0].data_ptr(), raw["swa0"][0].data_ptr())

    def test_overlaid_block_copy_is_independent_of_view_order(self):
        page_elements = 128 * 192 * 2
        for summary_first in (True, False):
            raw = torch.zeros(4 * page_elements, dtype=torch.bfloat16)
            k = raw.as_strided((4, 128, 1, 192), (page_elements, 192, 192, 1))
            v = raw.as_strided((4, 128, 1, 192), (page_elements, 192, 192, 1), 128 * 192)
            summary = raw.view(torch.uint8).as_strided((4, 1, 16, 256), (page_elements * 2, 4096, 256, 1))
            ring = raw.as_strided((4, 1, 128, 384), (page_elements, page_elements, 384, 1))
            k[1].fill_(7)
            v[1].fill_(11)
            k[2].fill_(17)
            v[2].fill_(19)
            original = raw.clone().reshape(4, page_elements)
            entries = [summary, (k, v), ring] if summary_first else [(k, v), ring, summary]
            copy_blocks = extract(
                ROOT / "vllm_ascend/worker/utils.py", "copy_kv_cache_blocks_inplace", {"torch": torch}
            )
            copy_blocks(entries, 4, [NS(src_block_id=1, dst_block_id=2), NS(src_block_id=2, dst_block_id=3)])
            actual = raw.reshape(4, page_elements)
            self.assertTrue(torch.equal(actual[0], original[0]))
            self.assertTrue(torch.equal(actual[1], original[1]))
            self.assertTrue(torch.equal(actual[2], original[1]))
            self.assertTrue(torch.equal(actual[3], original[2]))

    def test_only_one_backing_has_exact_upstream_size(self):
        kv = caches()
        with patch.object(helper.torch, "zeros", wraps=torch.zeros) as allocate:
            raw = helper.allocate_step5_shared_kv_backing(kv, "cpu")
        self.assertEqual(allocate.call_count, 1)
        self.assertEqual(raw["mtp"][0].untyped_storage().nbytes(), kv.kv_cache_tensors[0].size)

    def test_original_reshape_consumes_pairs_without_block_interleaving(self):
        kv = caches()
        raw = helper.allocate_step5_shared_kv_backing(kv, "cpu")
        unused = type("UnusedSpecialSpec", (), {})
        env = {
            "get_current_vllm_config": config,
            "_is_dsv4_model": lambda _: False,
            "is_deepseek_v41_cache": lambda _: False,
            "_get_layer_kv_cache_specs": lambda _: helper._layer_specs(kv.kv_cache_groups),
            "get_storage_block_size": lambda spec: spec.block_size,
            "get_kv_cache_compression_ratio": lambda _: 1,
            "is_hidden_state_cache_spec": lambda _: False,
            "AscendSFAIndexerCacheSpec": unused,
            "AscendIndexerKPoolTailSpec": unused,
            "MambaSpec": unused,
            "AttentionSpec": AttentionSpec,
            "FullAttentionSpec": FullAttentionSpec,
            "is_c8_mxfp_kv_quant": lambda _: False,
            "AscendSlidingWindowMLASpec": unused,
            "get_single_raw_mla_backing": lambda _: None,
            "AscendMLAAttentionSpec": unused,
            "MLAAttentionSpec": unused,
            "enable_sfa": lambda _: False,
            "torch": torch,
        }
        reshape = extract(ROOT / "vllm_ascend/worker/v2/attn_utils.py", "_reshape_kv_cache_v2", env)
        groups = [
            NS(
                kv_cache_group_id=i,
                kv_cache_spec=g.kv_cache_spec,
                layer_names=g.layer_names,
                backend=NS(
                    get_name=lambda: "ASCEND_STEP5_DSA",
                    get_kv_cache_shape=lambda blocks, bs, heads, dim, _: (
                        2,
                        blocks,
                        bs,
                        heads,
                        dim,
                    ),
                ),
            )
            for i, g in enumerate(kv.kv_cache_groups)
        ]
        views = reshape(groups, raw, "auto", [128] * 3, {}, kv)
        for name, pair in views.items():
            self.assertTrue(all(x.is_contiguous() for x in pair))
            self.assertEqual(pair[0].data_ptr(), raw[name][0].data_ptr())
            self.assertEqual(pair[1].data_ptr(), raw[name][1].data_ptr())

    def test_real_ascend_cow_uses_segment_views_and_alias_dedup(self):
        kv = caches()
        views = shaped(helper.allocate_step5_shared_kv_backing(kv, "cpu"), kv)
        views["dsa0"][0][1].fill_(11)
        views["dsa0"][1][1].fill_(12)
        views["dsa1"][0][1].fill_(21)
        views["dsa1"][1][1].fill_(22)
        source = ROOT / "vllm_ascend/worker/utils.py"
        cow = extract(source, "copy_kv_cache_blocks_inplace", {"torch": torch})
        # These chained copies require snapshots; copying duplicate aliases
        # twice would replace original block 2 in the second destination.
        views["dsa0"][0][2].fill_(31)
        cow(
            list(views.values()),
            kv.num_blocks,
            [NS(src_block_id=1, dst_block_id=2), NS(src_block_id=2, dst_block_id=3)],
        )
        self.assertTrue(torch.all(views["dsa0"][0][2] == 11))
        self.assertTrue(torch.all(views["dsa0"][0][3] == 31))
        self.assertTrue(torch.all(views["dsa0"][1][2] == 12))
        self.assertTrue(torch.all(views["dsa1"][0][2] == 21))
        self.assertTrue(torch.all(views["dsa1"][1][2] == 22))

    def test_real_zeroer_without_segments_does_not_write_main_cache(self):
        kv = caches()
        raw = helper.allocate_step5_shared_kv_backing(kv, "cpu")
        for pair in raw.values():
            for tensor in pair:
                tensor.fill_(7)
        zeroer = NS(_meta=None, device="cpu")
        source = vllm_source_file("v1/worker/utils.py")
        zero = extract(source, "zero_block_ids", {"async_tensor_h2d": torch.tensor, "torch": torch})
        zero(zeroer, [1, 2])
        for pair in raw.values():
            self.assertTrue(all(torch.all(tensor == 7) for tensor in pair))

    def test_dsa_sidecar_budgets_are_retained_and_separate(self):
        kv = caches()
        before = copy.deepcopy([g.kv_cache_spec.__dict__ for g in kv.kv_cache_groups])
        raw = helper.allocate_step5_shared_kv_backing(kv, "cpu")
        sidecache = torch.full((kv.num_blocks, 512), 42, dtype=torch.uint8)
        for pair in raw.values():
            for part in pair:
                part.fill_(7)
        self.assertTrue(torch.all(sidecache == 42))
        self.assertEqual(before, [g.kv_cache_spec.__dict__ for g in kv.kv_cache_groups])

    def test_geometry_model_runner_and_layout_mismatch_fallback(self):
        kv = caches()
        self.assertTrue(helper.supports_step5_shared_kv_backing(config(), kv.kv_cache_groups))
        cases = [
            lambda c, k: setattr(c, "use_v2_model_runner", False),
            lambda c, k: setattr(c.model_config.hf_text_config, "model_type", "step3p5"),
            lambda c, k: setattr(c, "kv_transfer_config", object()),
            lambda c, k: c.additional_config.update(enable_kvpp=True),
            lambda c, k: c.additional_config.update(sparse_kv_offload_config={"enabled": True}),
            lambda c, k: setattr(c.parallel_config, "decode_context_parallel_size", 2),
            lambda c, k: setattr(c.cache_config.get_resolved_kv_cache_layout(), "is_block_compact", False),
            lambda c, k: setattr(k.kv_cache_groups[1].kv_cache_spec, "num_kv_heads", 2),
            lambda c, k: setattr(k.kv_cache_groups[1].kv_cache_spec, "dtype", torch.float16),
            lambda c, k: setattr(k.kv_cache_groups[1].kv_cache_spec, "block_size", 256),
            lambda c, k: setattr(k.kv_cache_groups[1].kv_cache_spec, "page_size_padded", 196608),
            lambda c, k: setattr(k.kv_cache_groups[1].kv_cache_spec, "kv_quant_mode", 1),
        ]
        for mutate in cases:
            c, k = config(), caches()
            mutate(c, k)
            self.assertFalse(helper.supports_step5_shared_kv_backing(c, k.kv_cache_groups))

    def test_uniform_group_preserves_per_layer_geometry_checks(self):
        kv = caches()
        group = kv.kv_cache_groups[0]
        group.kv_cache_spec = UniformTypeKVCacheSpecs({name: FullAttentionSpec() for name in group.layer_names})
        self.assertTrue(helper.supports_step5_shared_kv_backing(config(), kv.kv_cache_groups))
        group.kv_cache_spec.kv_cache_specs["dsa1"].head_size = 128
        self.assertFalse(helper.supports_step5_shared_kv_backing(config(), kv.kv_cache_groups))

    def test_host_resident_group_falls_back_before_descriptor_validation(self):
        for group_index in range(3):
            kv = caches()
            kv.kv_cache_groups[group_index].host_resident = True
            # Descriptors remain device-resident and otherwise valid. The
            # budget gate itself must reject the host-resident group.
            self.assertFalse(helper.supports_step5_shared_kv_backing(config(), kv.kv_cache_groups))

    def test_common_backing_rejects_empty_negative_and_partial_layer_slots(self):
        reference = caches().kv_cache_tensors[0].size
        for invalid_size in (0, -reference, reference + 1):
            kv = caches()
            for descriptor in kv.kv_cache_tensors:
                descriptor.size = invalid_size
            # All descriptors agree, and the positive partial-slot case
            # still contains every layer region. This checks whole-pool
            # alignment rather than the existing bounds/common-size checks.
            with (
                patch.object(
                    helper.torch,
                    "zeros",
                    side_effect=AssertionError("must validate first"),
                ),
                self.assertRaises(ValueError),
            ):
                helper.allocate_step5_shared_kv_backing(kv, "cpu")

    def test_malformed_descriptors_fail_before_allocation(self):
        mutations = [
            lambda k: setattr(k, "needs_base_kv_cache_zeroing", True),
            lambda k: setattr(k.kv_cache_tensors[0], "offset", 1),
            lambda k: setattr(k.kv_cache_tensors[0], "layer_stride", 0),
            lambda k: setattr(k.kv_cache_tensors[0], "block_stride", 1),
            lambda k: setattr(k.kv_cache_tensors[0], "host_resident", True),
            lambda k: setattr(k.kv_cache_tensors[0], "size", 1),
            lambda k: setattr(k.kv_cache_tensors[0], "layers", ["unknown"]),
        ]
        for mutate in mutations:
            kv = caches()
            mutate(kv)
            with (
                patch.object(
                    helper.torch,
                    "zeros",
                    side_effect=AssertionError("must validate first"),
                ),
                self.assertRaises(ValueError),
            ):
                helper.allocate_step5_shared_kv_backing(kv, "cpu")


if __name__ == "__main__":
    unittest.main()
