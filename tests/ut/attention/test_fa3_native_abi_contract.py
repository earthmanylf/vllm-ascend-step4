# SPDX-License-Identifier: Apache-2.0
"""CPU checks for the exact native ABI and retained BLNHC integration."""

from __future__ import annotations

import ast
import importlib.util
import json
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[3]
BUNDLE = ROOT / "tools/step5/fa3_950"


def checker():
    spec = importlib.util.spec_from_file_location("fa3_native_checker", BUNDLE / "check_library.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_symbol_requires_both_booleans_and_both_strides():
    module = checker()
    expected = module.EXPECTED_SYMBOL
    assert module.is_requested_symbol(expected)
    assert not module.is_requested_symbol(expected.replace("ELb0ELb0", "ELb0"))
    assert not module.is_requested_symbol(expected.removesuffix("mm"))
    assert not module.is_requested_symbol(expected + "m")
    assert not module.is_requested_symbol(expected.replace("EL12MaskCategory4", "EL12MaskCategory0"))


def test_native_patch_hash_and_stride_propagation():
    import hashlib

    identity = json.loads((BUNDLE / "source-identity.json").read_text())
    patch = (BUNDLE / "native.patch").read_bytes()
    assert hashlib.sha256(patch).hexdigest() == identity["patch_sha256"]
    text = patch.decode()
    assert "a.k_block_stride, a.v_block_stride" in text
    assert "PagedRowIndex" in text
    assert "FA3_BLOCK_STRIDE_ABI_VERSION" in text
    assert "LSE_MODE, HAS_SOFTCAP" in text


def test_fa3_does_not_replace_blnhc_allocator_or_budget():
    allocation = (ROOT / "vllm_ascend/worker/v2/attn_utils.py").read_text()
    worker = (ROOT / "vllm_ascend/worker/worker.py").read_text()
    assert "allocate_fa3_compatible_raw_cache" not in allocation
    assert "UNIFORM_LAYER_COMPACT" not in allocation
    assert "VLLM_ASCEND_ENABLE_FA3" not in worker
    assert "allocate_upstream_kv_cache" in allocation
    helper = (ROOT / "vllm_ascend/core/step5_kv_cache.py").read_text()
    assert "Step5LayoutState" not in helper
    assert "Step5 V2 requires the upstream BLNHC layout" in helper


def test_fa3_profiling_flag_is_forwarded():
    tree = ast.parse((ROOT / "vllm_ascend/worker/v2/model_runner.py").read_text())
    runner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "NPUModelRunner")
    method = next(n for n in runner.body if isinstance(n, ast.FunctionDef) and n.name == "initialize_kv_cache")
    assert "is_profiling" in [arg.arg for arg in method.args.args]
    calls = [
        node
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "initialize_kv_cache"
    ]
    assert len(calls) == 1
    assert any(
        keyword.arg == "is_profiling" and isinstance(keyword.value, ast.Name) and keyword.value.id == "is_profiling"
        for keyword in calls[0].keywords
    )


def test_step5_dsa_and_fa3_keep_blnhc_only_capability():
    # Use the baseline's actual eligibility function; isolate imports only.
    helper_path = ROOT / "vllm_ascend/core/step5_kv_cache.py"
    tree = ast.parse(helper_path.read_text())
    eligibility = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name in ("uses_step5_blnhc", "validate_step5_kv_transfer_config")
    ]
    namespace = {}
    exec(compile(ast.Module(body=eligibility, type_ignores=[]), str(helper_path), "exec"), namespace)
    cfg = SimpleNamespace(
        use_v2_model_runner=True,
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(model_type="step5")),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1, prefill_context_parallel_size=1, decode_context_parallel_size=1
        ),
        additional_config={},
        attention_config=SimpleNamespace(hisparse_config=None),
        kv_transfer_config=None,
    )
    config_module = ModuleType("vllm.config")
    config_module.get_current_vllm_config_or_none = lambda: cfg
    helper_module = ModuleType("vllm_ascend.core.step5_kv_cache")
    helper_module.uses_step5_blnhc = namespace["uses_step5_blnhc"]
    layout_module = ModuleType("vllm.v1.kv_cache_layout")
    layout_module.KVCacheLayout = SimpleNamespace(BLNHC="BLNHC", LBNHC="LBNHC")

    @dataclass
    class Spec:
        num_kv_heads: int = 1
        head_size: int = 192
        dtype: object = torch.bfloat16
        block_stride_alignment: int = 128

    for filename, name in [
        ("models/step5/dsa_attention.py", "AscendStep5DSABackend"),
        ("attention/flash_attention_v3.py", "AscendFlashAttentionBackend"),
    ]:
        path = ROOT / "vllm_ascend" / filename
        cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == name)
        cls.bases = [ast.Name(id="object", ctx=ast.Load())]
        # Registration is covered by the real registry round-trip test.
        cls.decorator_list = []
        cls.body = [
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name in ("supported_kv_cache_layouts", "customize_spec")
        ]
        local = {
            "replace": replace,
            "KVCacheLayout": layout_module.KVCacheLayout,
            "uses_step5_blnhc": namespace["uses_step5_blnhc"],
        }
        exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), str(path), "exec"), local)
        with patch.dict(
            "sys.modules",
            {
                "vllm.config": config_module,
                "vllm_ascend.core.step5_kv_cache": helper_module,
                "vllm.v1.kv_cache_layout": layout_module,
            },
        ):
            backend = local[name]
            assert backend.supported_kv_cache_layouts() == ("BLNHC",)
            assert backend.customize_spec(Spec()).block_stride_alignment == 384
