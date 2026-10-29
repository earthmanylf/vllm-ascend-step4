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
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass, replace
from math import prod
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
stub.compute_layer_kv_cache_shape_bytes = lambda *a, **kw: None
stub.compute_layout_strides = lambda *a, **kw: None
module_spec = importlib.util.spec_from_file_location(
    "step5_shared_candidate", ROOT / "vllm_ascend/core/step5_kv_cache.py"
)
helper = importlib.util.module_from_spec(module_spec)
layout_spec = importlib.util.spec_from_file_location("cpu_kv_cache_layout", vllm_source_file("v1/kv_cache_layout.py"))
layout_stub = importlib.util.module_from_spec(layout_spec)
layout_spec.loader.exec_module(layout_stub)
with patch.dict(
    sys.modules,
    {"vllm.v1.kv_cache_interface": stub, "vllm.v1.kv_cache_layout": layout_stub},
):
    module_spec.loader.exec_module(helper)


def config():
    layout = NS(is_layer_compact=True, is_block_compact=True)
    return NS(
        use_v2_model_runner=True,
        model_config=NS(hf_text_config=NS(model_type="step5"), is_multimodal_model=False),
        speculative_config=None,
        kv_transfer_config=None,
        additional_config={},
        parallel_config=NS(pipeline_parallel_size=1, decode_context_parallel_size=1, prefill_context_parallel_size=1),
        cache_config=NS(get_resolved_kv_cache_layout=lambda: layout, enable_prefix_caching=False),
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


def tailring_config():
    page = 98304
    specs = {
        "main": FullAttentionSpec(),
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
                size=8 * page,
                layers=[name],
                offset=4 * page,
                layer_stride=4 * page,
                block_stride=page,
                host_resident=False,
            )
            for name in specs
        ],
        num_blocks_of=lambda descriptor: 4,
    )
    cfg = config()
    cfg.kv_transfer_config = NS(
        is_kv_transfer_instance=True,
        kv_connector="MooncakeConnectorV2",
        kv_connector_module_path=None,
        kv_role="kv_producer",
    )
    return cfg, kv


def load_cpu_upstream(relative, dependencies):
    """Load real upstream containers/planning, excluding device imports only."""
    source = vllm_source_file(relative)
    tree = ast.parse(source.read_text())
    tree.body = [
        node for node in tree.body if not isinstance(node, ast.ImportFrom) or not (node.module or "").startswith("vllm")
    ]
    # Local CPU Python is 3.10; upstream's Self is annotation-only here.
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "typing":
            node.names = [alias for alias in node.names if alias.name != "Self"]

    class StripBackendImports(ast.NodeTransformer):
        def visit_ImportFrom(self, node):
            return None if node.module == "vllm.v1.attention.backend" else node

    tree = StripBackendImports().visit(tree)
    tree.body.insert(0, ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0))
    ast.fix_missing_locations(tree)
    module = ModuleType("cpu_upstream_" + source.stem)
    module.__dict__.update({key: value for key, value in dependencies.items() if not key.startswith("__")})
    with patch.dict(sys.modules, {module.__name__: module}):
        exec(compile(tree, str(source), "exec"), module.__dict__)  # noqa: S102
    return module


def pd_upstream_fixture(counts=(2, 2, 2)):
    logger = NS(info=lambda *a: None, warning=lambda *a: None)
    backend = ast.parse(vllm_source_file("v1/attention/backend.py").read_text())
    multiple = next(node for node in backend.body if getattr(node, "name", None) == "MultipleOf")
    env = {"dataclass": dataclass}
    exec(compile(ast.Module(body=[multiple], type_ignores=[]), "cpu-multiple-of", "exec"), env)
    deps = dict(
        MultipleOf=env["MultipleOf"],
        init_logger=lambda *a: logger,
        cdiv=lambda n, d: (n + d - 1) // d,
        round_up=lambda n, a: (n + a - 1) // a * a,
        get_dtype_size=lambda dtype: dtype.itemsize,
        format_gib=lambda n: str(n / 1024**3),
    )
    layout = load_cpu_upstream("v1/kv_cache_layout.py", {})
    registry = load_cpu_upstream("v1/kv_cache_spec_registry.py", deps)
    interface = load_cpu_upstream(
        "v1/kv_cache_interface.py",
        dict(
            deps,
            KVCacheSpecRegistry=registry.KVCacheSpecRegistry,
            MambaAttentionBackendEnum=NS(MAMBA2="mamba2"),
            KVCacheLayout=layout.KVCacheLayout,
            _DIM_B=1,
            _DIM_L=0,
        ),
    )
    for name in ("FullAttentionSpec", "CircularBufferSpec", "SlidingWindowSpec"):
        registry.KVCacheSpecRegistry.register(getattr(interface, name), manager_class=object)
    cfg, _ = tailring_config()
    cfg.cache_config.get_resolved_kv_cache_layout = lambda: layout.KVCacheLayout.LBNHC
    cfg.cache_config.num_gpu_blocks_override = None
    cfg.cache_config.prefix_cache_retention_interval = None
    cfg.attention_config = NS(hisparse_config=None)
    cfg.scheduler_config = NS(disable_hybrid_kv_cache_manager=False)
    cfg.model_config.max_model_len = cfg.model_config.original_max_model_len = 1
    cfg.parallel_config.tensor_parallel_size = 1
    cfg.num_prefill_lookahead_tokens = 0
    geometry = dict(block_size=128, num_kv_heads=1, head_size=192, dtype=torch.bfloat16)
    main = interface.FullAttentionSpec(**geometry)
    summary = interface.FullAttentionSpec(
        **dict(geometry, head_size=256, head_size_v=0, dtype=torch.uint8),
        tokens_per_state=8,
        page_size_padded=main.page_size_bytes,
    )
    ring = interface.CircularBufferSpec(**dict(geometry, head_size=384, head_size_v=0))
    specs = {
        f"{name}.{i}": spec
        for name, spec, count in zip(("main", "summary", "ring"), (main, summary, ring), counts)
        for i in range(count)
    }
    planner = load_cpu_upstream(
        "v1/core/kv_cache_planning.py",
        dict(interface.__dict__, **deps, KVCacheConfigBuilder=object, get_hisparse_kv_cache_groups=lambda *a: None),
    )
    builder = NS(
        get_kv_cache_groups=planner.get_kv_cache_groups,
        _get_pool_bytes_per_block=lambda *a: None,
        get_kv_cache_config_from_groups=lambda c, g, num_blocks: planner.get_kv_cache_config_from_groups(
            c, g, num_blocks * planner._pool_bytes_per_block(c, g)
        ),
    )
    builder_module = ModuleType("vllm.v1.core.kv_cache_config_builder")
    builder_module.get_kv_cache_config_builder = lambda c: builder
    candidate = importlib.util.module_from_spec(module_spec)
    with patch.dict(
        sys.modules,
        {"vllm.v1.kv_cache_interface": interface, "vllm.v1.kv_cache_layout": layout},
    ):
        module_spec.loader.exec_module(candidate)
    env = dict(planner.__dict__)
    env.update(candidate.__dict__)
    worker = NS(vllm_config=cfg, get_kv_cache_spec=lambda: dict(specs))
    source = ROOT / "vllm_ascend/worker/worker.py"
    worker._apply_step5_tailring_alignment_reserve = extract(
        source, "_apply_step5_tailring_alignment_reserve", env
    ).__get__(worker)
    scale = extract(source, "_scale_kv_cache_memory_for_multi_group", env)
    return NS(
        cfg=cfg,
        specs=specs,
        planner=planner,
        helper=candidate,
        worker=worker,
        scale=scale,
        env=env,
        builder_module=builder_module,
    )


class SharedCacheTests(unittest.TestCase):
    def test_prefix_cache_pd_uses_real_pages_and_private_ring(self):
        f = pd_upstream_fixture()
        f.cfg.cache_config.enable_prefix_caching = True
        for role in ("kv_producer", "kv_consumer"):
            with self.subTest(role=role):
                f.cfg.kv_transfer_config.kv_role = role
                f.helper.validate_step5_kv_transfer_config(f.cfg)
                self.assertTrue(f.helper.uses_step5_blnhc(f.cfg))
                specs = {name: replace(spec, page_size_padded=None) for name, spec in f.specs.items()}
                groups = f.planner.get_kv_cache_groups(f.cfg, specs)
                for group in groups:
                    spec = group.kv_cache_spec
                    if isinstance(spec, f.planner.CircularBufferSpec):
                        self.assertFalse(spec.prefix_cacheable)
                    else:
                        self.assertTrue(spec.prefix_cacheable)
                self.assertEqual(specs["summary.0"].page_size_bytes, 4096)

    def test_mtp3_prefix_pd_requires_validated_eager_draft_profile(self):
        f = pd_upstream_fixture()
        f.cfg.cache_config.enable_prefix_caching = True
        f.cfg.model_config.enforce_eager = True
        f.cfg.model_config.hf_text_config.hidden_size = 4096
        draft = NS(model_type="step5_mtp", num_nextn_predict_layers=3, hidden_size=4096)
        f.cfg.speculative_config = NS(
            method="mtp",
            num_speculative_tokens=3,
            enforce_eager=True,
            draft_model_config=NS(hf_text_config=draft),
            num_speculative_tokens_per_batch_size=None,
        )
        f.helper.validate_step5_kv_transfer_config(f.cfg)
        self.assertTrue(f.helper.uses_step5_blnhc(f.cfg))
        f.cfg.speculative_config.enforce_eager = False
        with self.assertRaisesRegex(ValueError, "eager"):
            f.helper.validate_step5_kv_transfer_config(f.cfg)
        f.cfg.speculative_config.enforce_eager = True
        f.cfg.speculative_config.num_speculative_tokens_per_batch_size = None
        f.helper.validate_step5_kv_transfer_config(f.cfg)
        f.cfg.kv_transfer_config.kv_role = "kv_consumer"
        f.helper.validate_step5_kv_transfer_config(f.cfg)

    def test_vendor_graph_async_dp2_tp4_profile_and_negative_controls(self):
        f = pd_upstream_fixture()
        f.cfg.cache_config.enable_prefix_caching = True
        f.cfg.model_config.enforce_eager = False
        f.cfg.model_config.hf_text_config.hidden_size = 4096
        f.cfg.parallel_config.tensor_parallel_size = 4
        f.cfg.parallel_config.data_parallel_size = 2
        f.cfg.parallel_config.data_parallel_size_local = 2
        f.cfg.parallel_config.enable_expert_parallel = True
        f.cfg.scheduler_config.max_num_seqs = 256
        f.cfg.scheduler_config.async_scheduling = True
        f.cfg.compilation_config = NS(
            mode=3, cudagraph_mode=NS(name="FULL_DECODE_ONLY"), cudagraph_capture_sizes=[4, 8, 512]
        )
        f.cfg.speculative_config = NS(
            method="mtp",
            num_speculative_tokens=3,
            enforce_eager=False,
            draft_model_config=NS(
                hf_text_config=NS(model_type="step5_mtp", num_nextn_predict_layers=3, hidden_size=4096)
            ),
            num_speculative_tokens_per_batch_size=None,
        )
        f.helper.validate_step5_kv_transfer_config(f.cfg)
        f.cfg.parallel_config.data_parallel_size = 4
        f.cfg.kv_transfer_config.kv_role = "kv_consumer"
        f.helper.validate_step5_kv_transfer_config(f.cfg)
        for obj, name, wrong in (
            (f.cfg.parallel_config, "data_parallel_size", 3),
            (f.cfg.scheduler_config, "async_scheduling", False),
            (f.cfg.compilation_config, "mode", 0),
            (f.cfg.compilation_config, "cudagraph_capture_sizes", [4, 510]),
            (f.cfg.model_config, "enforce_eager", True),
        ):
            old = getattr(obj, name)
            setattr(obj, name, wrong)
            with self.assertRaises(ValueError):
                f.helper.validate_step5_kv_transfer_config(f.cfg)
            setattr(obj, name, old)

    def test_pp4_tp4_graph_producer_only_profile(self):
        f = pd_upstream_fixture()
        cfg = f.cfg
        cfg.kv_transfer_config.kv_role = "kv_producer"
        cfg.model_config.enforce_eager = False
        cfg.model_config.hf_text_config.hidden_size = 4096
        cfg.parallel_config.pipeline_parallel_size = 4
        cfg.parallel_config.tensor_parallel_size = 4
        cfg.parallel_config.data_parallel_size = 1
        cfg.parallel_config.data_parallel_size_local = 1
        cfg.parallel_config.enable_expert_parallel = True
        cfg.scheduler_config.max_num_seqs = 32
        cfg.scheduler_config.async_scheduling = True
        cfg.compilation_config = NS(
            mode=3, cudagraph_mode=NS(name="FULL_DECODE_ONLY"), cudagraph_capture_sizes=[4, 8, 512]
        )
        cfg.speculative_config = NS(
            method="mtp",
            num_speculative_tokens=3,
            enforce_eager=False,
            draft_model_config=NS(
                hf_text_config=NS(model_type="step5_mtp", num_nextn_predict_layers=3, hidden_size=4096)
            ),
            num_speculative_tokens_per_batch_size=None,
        )
        f.helper.validate_step5_kv_transfer_config(cfg)
        cfg.speculative_config.num_speculative_tokens_per_batch_size = None
        f.helper.validate_step5_kv_transfer_config(cfg)
        for obj, key, value in (
            (cfg.kv_transfer_config, "kv_role", "kv_consumer"),
            (cfg.parallel_config, "tensor_parallel_size", 8),
            (cfg.parallel_config, "pipeline_parallel_size", 2),
            (cfg.scheduler_config, "async_scheduling", False),
        ):
            previous = getattr(obj, key)
            setattr(obj, key, value)
            with self.subTest(key=key), self.assertRaises(ValueError):
                f.helper.validate_step5_kv_transfer_config(cfg)
            setattr(obj, key, previous)

    def test_prefix_cache_pd_still_rejects_non_v2_runner(self):
        f = pd_upstream_fixture()
        f.cfg.cache_config.enable_prefix_caching = True
        f.cfg.use_v2_model_runner = False
        with self.assertRaisesRegex(ValueError, "V2"):
            f.helper.validate_step5_kv_transfer_config(f.cfg)

    def test_supported_v2_pd_selects_validated_blnhc(self):
        f = pd_upstream_fixture()
        for connector in ("MooncakeConnectorV2", "MooncakePullConnector"):
            for role in ("kv_producer", "kv_consumer"):
                with self.subTest(connector=connector, role=role):
                    f.cfg.kv_transfer_config.kv_connector = connector
                    f.cfg.kv_transfer_config.kv_role = role
                    f.helper.validate_step5_kv_transfer_config(f.cfg)
                    self.assertTrue(f.helper.uses_step5_blnhc(f.cfg))
                    groups = f.planner.get_kv_cache_groups(f.cfg, dict(f.specs))
                    self.assertTrue(f.helper.supports_step5_tailring_shared_backing(f.cfg, groups))
        f.cfg.kv_transfer_config.is_kv_transfer_instance = False
        self.assertTrue(f.helper.uses_step5_blnhc(f.cfg))
        f.cfg.kv_transfer_config = None
        self.assertTrue(f.helper.uses_step5_blnhc(f.cfg))

    def test_supported_v2_pd_allocator_uses_aligned_compact_page_addresses(self):
        f = pd_upstream_fixture()
        groups = f.planner.get_kv_cache_groups(f.cfg, dict(f.specs))
        planned = f.planner.get_kv_cache_config_from_groups(f.cfg, groups, 4 * 196608)
        with patch.object(torch, "zeros", wraps=torch.zeros) as alloc:
            views = f.helper.allocate_step5_tailring_shared_backing(planned, "cpu", align_for_transfer=True)
        self.assertEqual(alloc.call_count, 1)
        k, v = views["main.0"]
        self.assertEqual(k.data_ptr() % f.helper.STEP5_PD_BUFFER_ALIGNMENT, 0)
        self.assertEqual(k.stride(0) * k.element_size(), 98304)
        self.assertEqual(v.data_ptr() - k.data_ptr(), 49152)
        self.assertEqual(views["main.1"][0].data_ptr() - k.data_ptr(), 4 * 98304)
        self.assertEqual(views["summary.0"].shape, (4, 1, 16, 256))
        self.assertEqual(views["summary.0"].stride(0), 98304)
        self.assertEqual(f.specs["summary.0"].page_size_padded, 98304)
        tensors = [t for value in views.values() for t in (value if isinstance(value, tuple) else (value,))]
        self.assertEqual({t.untyped_storage().data_ptr() for t in tensors}, {k.untyped_storage().data_ptr()})
        self.assertEqual(k.untyped_storage().nbytes(), planned.kv_cache_tensors[0].size + 2097152)

    def test_supported_v2_pd_budget_reserves_once_before_real_planner(self):
        f = pd_upstream_fixture()
        budget = 4 * 196608 + f.helper.STEP5_PD_BUFFER_ALIGNMENT + 19
        reduced = f.scale(f.worker, budget)
        self.assertEqual(reduced, budget - f.helper.STEP5_PD_BUFFER_ALIGNMENT)
        self.assertEqual(f.worker._apply_step5_tailring_alignment_reserve(budget), reduced)
        groups = f.planner.get_kv_cache_groups(f.cfg, dict(f.specs))
        planned = f.planner.get_kv_cache_config_from_groups(f.cfg, groups, reduced)
        self.assertEqual(planned.num_blocks, 4)
        views = f.helper.allocate_step5_tailring_shared_backing(planned, "cpu", align_for_transfer=True)
        self.assertLessEqual(views["main.0"][0].untyped_storage().nbytes(), budget)

    def test_pd_layout_selection_validates_capability(self):
        f = pd_upstream_fixture()
        f.cfg.kv_transfer_config.kv_connector = "UnsupportedConnector"
        with self.assertRaisesRegex(ValueError, "registered Ascend"):
            f.helper.uses_step5_blnhc(f.cfg)
        with self.assertRaisesRegex(ValueError, "registered Ascend"):
            f.helper.validate_step5_kv_transfer_config(f.cfg)

    def test_pd_backend_and_summary_spec_select_real_page_blnhc(self):
        f = pd_upstream_fixture()
        namespace = dict(f.env, AttentionBackend=object, replace=replace)
        source = ROOT / "vllm_ascend/models/step5/dsa_attention.py"
        backend = next(
            n for n in ast.parse(source.read_text()).body if getattr(n, "name", None) == "AscendStep5DSABackend"
        )
        backend.body = [
            n for n in backend.body if getattr(n, "name", None) in ("supported_kv_cache_layouts", "customize_spec")
        ]
        exec(compile(ast.Module(body=[backend], type_ignores=[]), str(source), "exec"), namespace)
        config_module = ModuleType("vllm.config")
        config_module.get_current_vllm_config_or_none = lambda: f.cfg
        with patch.dict(sys.modules, {"vllm.config": config_module}):
            self.assertEqual(namespace[backend.name].supported_kv_cache_layouts(), (f.helper.KVCacheLayout.BLNHC,))
            self.assertEqual(namespace[backend.name].customize_spec(f.specs["main.0"]).block_stride_alignment, 384)

        class Base:
            def get_kv_cache_spec(self, _):
                return replace(f.specs["summary.0"], page_size_padded=None)

        source = ROOT / "vllm_ascend/models/step5/summary_cache.py"
        layer = next(
            n
            for n in ast.parse(source.read_text()).body
            if getattr(n, "name", None) == "AscendStep5SparseSummaryCacheLayer"
        )
        namespace.update(Step5SparseSummaryCacheLayer=Base, Any=object, AscendStep5SparseSummaryBackend=object)
        exec(compile(ast.Module(body=[layer], type_ignores=[]), str(source), "exec"), namespace)
        summary = namespace[layer.name].__new__(namespace[layer.name])
        main = NS(get_kv_cache_spec=lambda _: f.specs["main.0"])
        summary._main_layer_ref = lambda: main
        padded = summary.get_kv_cache_spec(f.cfg)
        self.assertEqual(padded.page_size_bytes, 4096)
        self.assertIsNone(padded.page_size_padded)
        self.assertEqual(padded.tokens_per_state, 8)
        f.cfg.kv_transfer_config = None
        local = summary.get_kv_cache_spec(f.cfg)
        self.assertEqual(local.page_size_bytes, 4096)
        self.assertIsNone(local.page_size_padded)
        with patch.dict(sys.modules, {"vllm.config": config_module}):
            self.assertEqual(namespace[backend.name].supported_kv_cache_layouts(), (f.helper.KVCacheLayout.BLNHC,))

    def test_pd_override_upstream_reproduces_original_budget_overrun(self):
        f = pd_upstream_fixture()
        f.cfg.cache_config.num_gpu_blocks_override = 5
        budget = 2883603
        groups = f.planner.get_kv_cache_groups(f.cfg, dict(f.specs))
        self.assertEqual(f.planner._pool_bytes_per_block(f.cfg, groups), 196608)
        planned = f.planner.get_kv_cache_config_from_groups(f.cfg, groups, budget - 2097152)
        raw = f.helper.allocate_step5_tailring_shared_backing(planned, "cpu", align_for_transfer=True)
        actual = raw["main.0"][0].untyped_storage().nbytes()
        self.assertEqual(actual, 3080192)
        self.assertEqual(actual - budget, 196589)
        # The real full planner accepts even a one-byte reported budget: override
        # capacity replaces it before admission, so it cannot enforce this guard.
        with patch.dict(sys.modules, {"vllm.v1.core.kv_cache_config_builder": f.builder_module}):
            full = f.planner.get_kv_cache_configs(f.cfg, [dict(f.specs)], [1])[0]
        self.assertEqual(full.num_blocks, 5)
        self.assertEqual(full.kv_cache_tensors[0].size, 983040)

    def test_pd_override_original_budget_guard_profile_and_explicit(self):
        for path in ("profile", "explicit"):
            for override, budget, safe in (
                (4, 2883603, True),
                (5, 2883603, False),
                (5, 3080192, True),
                (5, 3080191, False),
                (0, 2883603, False),
                (-1, 2883603, False),
            ):
                with self.subTest(path=path, override=override, budget=budget):
                    f = pd_upstream_fixture()
                    f.cfg.cache_config.num_gpu_blocks_override = override
                    apply = f.scale if path == "profile" else lambda w, b: w._apply_step5_tailring_alignment_reserve(b)
                    if not safe:
                        with self.assertRaisesRegex(ValueError, "num_gpu_blocks_override"):
                            apply(f.worker, budget)
                        continue
                    reduced = apply(f.worker, budget)
                    self.assertEqual(reduced, budget - 2097152)
                    self.assertEqual(f.cfg.cache_config.num_gpu_blocks_override, override)
                    with patch.dict(sys.modules, {"vllm.v1.core.kv_cache_config_builder": f.builder_module}):
                        planned = f.planner.get_kv_cache_configs(f.cfg, [dict(f.specs)], [reduced])[0]
                    self.assertEqual(planned.num_blocks, override)
                    raw = f.helper.allocate_step5_tailring_shared_backing(planned, "cpu", align_for_transfer=True)
                    self.assertLessEqual(raw["main.0"][0].untyped_storage().nbytes(), budget)

    def test_pd_override_group_padding_and_aliasing_use_actual_planner_bytes(self):
        # Uneven buckets generate padding and multiple aliased groups. Neither
        # sum(all layer pages) nor total layers / group count is the pool divisor.
        for counts in ((3, 4, 3), (2, 5, 2)):
            for path in ("profile", "explicit"):
                with self.subTest(counts=counts, path=path):
                    f = pd_upstream_fixture(counts)
                    groups = f.planner.get_kv_cache_groups(f.cfg, dict(f.specs))
                    self.assertEqual(len(groups), 3 if counts == (3, 4, 3) else 5)
                    per_block = f.planner._pool_bytes_per_block(f.cfg, groups)
                    self.assertEqual(per_block, (4 if counts == (3, 4, 3) else 2) * 98304)
                    f.cfg.cache_config.num_gpu_blocks_override = 6
                    exact_budget = 6 * per_block + 2097152
                    apply = f.scale if path == "profile" else lambda w, b: w._apply_step5_tailring_alignment_reserve(b)
                    self.assertEqual(apply(f.worker, exact_budget), exact_budget - 2097152)
                    with self.assertRaisesRegex(ValueError, "num_gpu_blocks_override"):
                        apply(f.worker, exact_budget - 1)

    def test_pd_override_determine_memory_rejects_before_planning_or_allocation(self):
        for path in ("profile", "explicit"):
            for override in (4, 5):
                with self.subTest(path=path, override=override):
                    f = pd_upstream_fixture()
                    budget = 2883603
                    f.cfg.cache_config.num_gpu_blocks_override = override
                    worker = f.worker
                    worker.cache_config = f.cfg.cache_config
                    worker.cache_config.kv_cache_memory_bytes = budget if path == "explicit" else None
                    worker.parallel_config = NS(data_parallel_size=1)
                    worker.init_snapshot = NS(free_memory=budget * 2)
                    worker.requested_memory = budget + 100
                    worker.device = "cpu"
                    worker.model_runner = NS(profile_run=lambda **kwargs: None, model_memory_usage=100)
                    worker._scale_kv_cache_memory_for_multi_group = f.scale.__get__(worker)
                    worker._apply_kvpp_memory_budget = lambda b: b
                    worker._apply_kv_offload_decode_memory_constraints = lambda b: b
                    profile = NS(
                        before_profile=NS(torch_peak=0),
                        after_profile=NS(free_memory=budget),
                        non_torch_increase=0,
                        weights_memory=100,
                    )
                    f.env.update(
                        torch=NS(inference_mode=torch.inference_mode, npu=NS(memory_stats=lambda d: {})),
                        GiB_bytes=1024**3,
                        maybe_apply_startup_plan=lambda w: None,
                        envs=NS(VLLM_RANDOMIZE_DP_DUMMY_INPUTS=False),
                        memory_profiling=lambda *a, result=profile, **kw: nullcontext(result),
                        set_current_vllm_config=lambda c: nullcontext(),
                        get_layerwise_reuse_config=lambda c: None,
                        logger=NS(info=lambda *a: None, info_once=lambda *a, **kw: None, debug=lambda *a: None),
                    )
                    determine = extract(ROOT / "vllm_ascend/worker/worker.py", "determine_available_memory", f.env)
                    with (
                        patch.object(f.planner, "get_kv_cache_configs", side_effect=AssertionError("too late")),
                        patch.object(torch, "zeros", side_effect=AssertionError("allocation reached")),
                    ):
                        if override == 5:
                            with self.assertRaisesRegex(ValueError, "requires 3080192 bytes.*2883603"):
                                determine(worker)
                        else:
                            self.assertEqual(determine(worker), budget - 2097152)

    def test_non_pd_override_keeps_upstream_semantics(self):
        f = pd_upstream_fixture()
        f.cfg.kv_transfer_config = None
        for override in (5, 0, -1):
            f.cfg.cache_config.num_gpu_blocks_override = override
            self.assertEqual(f.scale(f.worker, 1), 1)
            self.assertEqual(f.worker._apply_step5_tailring_alignment_reserve(1), 1)
            groups = f.planner.get_kv_cache_groups(f.cfg, dict(f.specs))
            planned = f.planner.get_kv_cache_config_from_groups(f.cfg, groups, 1)
            self.assertEqual(planned.num_blocks, override)

    def test_pd_tailring_eligibility_and_negative_controls(self):
        cfg, kv = tailring_config()
        for connector in ("MooncakeConnectorV2", "MooncakePullConnector"):
            for role in ("kv_producer", "kv_consumer"):
                cfg.kv_transfer_config.kv_connector = connector
                cfg.kv_transfer_config.kv_role = role
                self.assertTrue(helper.supports_step5_tailring_shared_backing(cfg, kv.kv_cache_groups))
        mutations = [
            lambda c: setattr(c.kv_transfer_config, "kv_connector", "OtherConnector"),
            lambda c: setattr(c.kv_transfer_config, "kv_connector_module_path", "custom.module"),
            lambda c: setattr(c.kv_transfer_config, "kv_role", "kv_both"),
            lambda c: setattr(c.kv_transfer_config, "is_kv_transfer_instance", False),
            lambda c: setattr(c.parallel_config, "pipeline_parallel_size", 2),
            lambda c: setattr(c.parallel_config, "prefill_context_parallel_size", 2),
            lambda c: setattr(c.parallel_config, "decode_context_parallel_size", 2),
            lambda c: setattr(c, "speculative_config", NS(method="mtp")),
            lambda c: (
                setattr(c.cache_config, "enable_prefix_caching", True),
                setattr(c, "use_v2_model_runner", False),
            ),
            lambda c: c.additional_config.update(enable_kvpp=True),
            lambda c: c.additional_config.update(sparse_kv_offload_config={"enabled": True}),
            lambda c: setattr(c, "attention_config", NS(hisparse_config={})),
        ]
        for mutate in mutations:
            invalid = copy.deepcopy(cfg)
            mutate(invalid)
            self.assertFalse(helper.supports_step5_tailring_shared_backing(invalid, kv.kv_cache_groups))
        self.assertTrue(helper.supports_step5_tailring_shared_backing(config(), kv.kv_cache_groups))
        self.assertFalse(helper.supports_step5_shared_kv_backing(cfg, caches().kv_cache_groups))

    def test_pd_accepts_unified_vl_weights_without_text_only_view(self):
        """The vendor ships one unified VL checkpoint (Step5ForConditionalGeneration
        with a vision sub-config). Step5 PD must accept it directly instead of
        forcing a text-only target/draft split, which is what forced
        ``--model /work/run/target-text92`` plus a separate
        ``/work/run/draft-mtp3``."""
        cfg = config()
        cfg.kv_transfer_config = NS(
            kv_connector="MooncakeConnectorV2",
            kv_connector_module_path=None,
            kv_role="kv_producer",
            is_kv_transfer_instance=True,
        )
        cfg.model_config.is_multimodal_model = True
        # No speculative config: this test isolates the multimodal rejection.
        self.assertIsNone(cfg.speculative_config)
        helper.validate_step5_kv_transfer_config(cfg)

    def test_pd_aligned_slab_views_keep_raw_storage_offset_and_registration(self):
        cfg, kv = tailring_config()
        alignment = 2 * 1024 * 1024
        allocated = helper.allocate_step5_tailring_shared_backing(kv, "cpu", align_for_transfer=True)
        env = dict(
            get_current_vllm_config=lambda: cfg,
            uses_step5_blnhc=lambda _: False,
            step5_tailring_alignment_reserve=helper.step5_tailring_alignment_reserve,
            supports_step5_tailring_shared_backing=helper.supports_step5_tailring_shared_backing,
            allocate_step5_tailring_shared_backing=helper.allocate_step5_tailring_shared_backing,
        )
        dispatch = extract(ROOT / "vllm_ascend/worker/v2/attn_utils.py", "allocate_kv_cache_main", env)
        dispatched = dispatch(kv, "cpu", None, [128] * 3)
        self.assertEqual(dispatched["main"][0].untyped_storage().nbytes(), kv.kv_cache_tensors[0].size + alignment)
        invalid = copy.deepcopy(kv)
        invalid.kv_cache_tensors[0].offset = -1
        with (
            self.assertRaises(ValueError),
            patch.object(helper.torch, "zeros", side_effect=AssertionError("no fallback")),
        ):
            dispatch(invalid, "cpu", None, [128] * 3)
        k, v = allocated["main"]
        storage = k.untyped_storage()
        base = (storage.data_ptr() + alignment - 1) // alignment * alignment
        self.assertEqual(storage.nbytes(), kv.kv_cache_tensors[0].size + alignment)
        self.assertEqual(k.data_ptr(), base + kv.kv_cache_tensors[0].offset)
        self.assertEqual(v.data_ptr() - k.data_ptr(), 128 * 192 * 2)
        self.assertEqual(allocated["summary"].data_ptr(), k.data_ptr())
        self.assertEqual(allocated["ring"].data_ptr(), k.data_ptr())
        k[1].fill_(7)
        v[1].fill_(9)
        self.assertTrue(torch.all(k[1] == 7))
        self.assertTrue(torch.all(v[1] == 9))
        self.assertTrue(torch.all(k[0] == 0))
        namespace = dict(torch=torch, dataclass=dataclass, get_kv_cache_tensor_layers=lambda d: d.layers)
        source = ROOT / "vllm_ascend/distributed/kv_transfer/utils/utils.py"
        tree = ast.parse(source.read_text())
        node = next(n for n in tree.body if getattr(n, "name", None) == "RegisterRegions")
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
        namespace["tensor_storage_key"] = extract(source, "tensor_storage_key", namespace)
        source = ROOT / "vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake/utils.py"
        namespace["KV_CACHE_BUFFER_ALIGNMENT"] = alignment
        for name in ("as_kv_cache_tensors", "_get_storage_nbytes", "_get_tensor_span_nbytes"):
            extract(source, name, namespace)
        collect = extract(source, "collect_configured_register_regions", namespace)
        regions = collect(kv, allocated)
        self.assertEqual(regions.ptrs, [base])
        self.assertEqual(regions.lengths, [kv.kv_cache_tensors[0].size])

    def test_pd_invalid_geometry_and_descriptors_fail_before_allocating(self):
        _, kv = tailring_config()
        mutations = [
            lambda c: setattr(c.kv_cache_groups[2].kv_cache_spec, "head_size", 768),
            lambda c: setattr(c.kv_cache_groups[2].kv_cache_spec, "head_size", 0),
            lambda c: c.kv_cache_groups[2].kv_cache_spec.__dict__.update(head_size=768, page_size_padded=98304),
            lambda c: c.kv_cache_groups[2].kv_cache_spec.__dict__.update(head_size=192, page_size_padded=98304),
            lambda c: setattr(c.kv_cache_groups[1].kv_cache_spec, "tokens_per_state", 0),
            lambda c: setattr(c.kv_cache_groups[1].kv_cache_spec, "page_size_padded", 2048),
            lambda c: setattr(c.kv_cache_tensors[0], "offset", 1),
            lambda c: setattr(c.kv_cache_tensors[0], "layer_stride", -1),
            lambda c: setattr(c.kv_cache_tensors[0], "layer_stride", 1),
            lambda c: setattr(c.kv_cache_tensors[0], "size", c.kv_cache_tensors[0].size + 1),
            lambda c: setattr(c, "num_blocks_of", lambda d: 3),
        ]
        for mutate in mutations:
            invalid = copy.deepcopy(kv)
            mutate(invalid)
            with (
                patch.object(helper.torch, "zeros", side_effect=AssertionError("validate before allocating")),
                self.assertRaises(ValueError),
            ):
                helper.allocate_step5_tailring_shared_backing(invalid, "cpu")

    def test_pd_profile_and_explicit_budget_reserve_once(self):
        cfg, kv = tailring_config()
        env = dict(
            get_kv_cache_groups=lambda *_: kv.kv_cache_groups,
            uses_step5_blnhc=lambda _: False,
            supports_step5_tailring_shared_backing=helper.supports_step5_tailring_shared_backing,
            step5_tailring_alignment_reserve=helper.step5_tailring_alignment_reserve,
            supports_step5_shared_kv_backing=helper.supports_step5_shared_kv_backing,
            UniformTypeKVCacheSpecs=UniformTypeKVCacheSpecs,
            AttentionSpec=AttentionSpec,
            MambaSpec=type("MambaSpec", (), {}),
            logger=NS(info=lambda *args: None),
        )
        source = ROOT / "vllm_ascend/worker/worker.py"
        scale = extract(source, "_scale_kv_cache_memory_for_multi_group", env)
        worker = NS(vllm_config=cfg, get_kv_cache_spec=lambda: {"main": FullAttentionSpec()})
        worker._apply_step5_tailring_alignment_reserve = extract(
            source, "_apply_step5_tailring_alignment_reserve", env
        ).__get__(worker)
        budget = 16 * 1024 * 1024
        self.assertEqual(scale(worker, budget), budget - 2 * 1024 * 1024)
        self.assertEqual(scale(worker, 1), 0)
        cfg.kv_transfer_config.kv_connector = "OtherConnector"
        worker.model_config = cfg.model_config
        self.assertEqual(scale(worker, budget), int(budget * (1 / 3)))
        cfg.kv_transfer_config.kv_connector = "MooncakeConnectorV2"
        # Execute the actual explicit-memory fast path; no NPU profiling is reached.
        worker.cache_config = NS(kv_cache_memory_bytes=budget)
        worker.parallel_config = NS(data_parallel_size=1)
        worker.model_runner = NS(profile_run=lambda **kwargs: None)
        worker.init_snapshot = NS(free_memory=budget * 2)
        worker._apply_step5_tailring_alignment_reserve = extract(
            source, "_apply_step5_tailring_alignment_reserve", env
        ).__get__(worker)
        worker._apply_kvpp_memory_budget = lambda b: b
        worker._apply_kv_offload_decode_memory_constraints = lambda b: b
        env.update(
            torch=torch,
            GiB_bytes=1024**3,
            maybe_apply_startup_plan=lambda w: None,
            envs=NS(VLLM_RANDOMIZE_DP_DUMMY_INPUTS=False),
            set_current_vllm_config=lambda c: nullcontext(),
            logger=NS(info=lambda *a: None),
        )
        determine = extract(source, "determine_available_memory", env)
        self.assertEqual(determine(worker), budget - 2 * 1024 * 1024)
        worker.vllm_config = config()
        worker.get_kv_cache_spec = lambda: (_ for _ in ()).throw(AssertionError("non-PD fast path must stay fast"))
        self.assertEqual(determine(worker), budget)

    def test_pd_upstream_planner_descriptors_fit_reserved_slab(self):
        cfg, kv = tailring_config()
        layout = cfg.cache_config.get_resolved_kv_cache_layout()
        layout.is_block_outermost = False
        layout.stride_order = (0, 1, 3, 2, 4)  # LBNHC
        cfg.attention_config = NS(hisparse_config=None)
        cfg.cache_config.num_gpu_blocks_override = None
        cfg.cache_config.prefix_cache_retention_interval = None
        for group in kv.kv_cache_groups:
            group.layer_names.append(group.layer_names[0] + ".second")
            spec = group.kv_cache_spec
            spec.get_block_stride_alignment = lambda: 1
            spec.num_heads = spec.num_kv_heads
            spec.state_content_size_bytes = (spec.head_size + spec.head_size_v) * spec.dtype.itemsize
            spec.get_num_kernel_states = lambda bs, s=spec: bs // getattr(s, "tokens_per_state", 1)
        cfg.cache_config.enable_prefix_caching = False
        env = dict(
            UniformTypeKVCacheSpecs=UniformTypeKVCacheSpecs,
            defaultdict=defaultdict,
            prod=prod,
            KVCacheTensor=lambda **kw: NS(host_resident=False, **kw),
            KVCacheConfig=lambda **kw: NS(**kw),
            round_up=lambda n, a: (n + a - 1) // a * a,
            _DIM_L=0,
            _DIM_B=1,
        )
        interface = vllm_source_file("v1/kv_cache_interface.py")
        extract(interface, "compute_layer_kv_cache_shape_bytes", env)
        extract(interface, "compute_layout_strides", env)
        source = vllm_source_file("v1/core/kv_cache_planning.py")
        for name in (
            "_get_per_layer_spec",
            "_get_layer_block_stride",
            "_get_kv_cache_bytes_per_block",
            "validate_kv_cache_layout",
            "may_override_num_blocks",
        ):
            extract(source, name, env)
        plan = extract(source, "get_kv_cache_config_from_groups", env)
        budget = 2 * 1024 * 1024 + 8 * 98304 + 19
        reserved = budget - helper.step5_tailring_alignment_reserve(cfg, kv.kv_cache_groups)
        planned = plan(cfg, kv.kv_cache_groups, reserved)
        planned.num_blocks_of = extract(interface, "num_blocks_of", {}).__get__(planned)
        self.assertEqual(planned.num_blocks, 4)
        allocated = helper.allocate_step5_tailring_shared_backing(planned, "cpu", align_for_transfer=True)
        self.assertLessEqual(allocated["main"][0].untyped_storage().nbytes(), budget)
        self.assertEqual(allocated["ring"].shape, (4, 1, 128, 384))
        self.assertEqual(allocated["main.second"][0].data_ptr() - allocated["main"][0].data_ptr(), 4 * 98304)
        allocated["main"][0][1].fill_(7)
        self.assertTrue(torch.all(allocated["main.second"][0] == 0))

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
            step5_tailring_alignment_reserve=helper.step5_tailring_alignment_reserve,
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
        worker._apply_step5_tailring_alignment_reserve = extract(
            ROOT / "vllm_ascend/worker/worker.py", "_apply_step5_tailring_alignment_reserve", env
        ).__get__(worker)
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
