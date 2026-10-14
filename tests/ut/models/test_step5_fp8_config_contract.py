# SPDX-License-Identifier: Apache-2.0
"""Exercise native FP8 config selection without accelerator imports."""

import ast
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

CONFIG_FILE = Path(__file__).resolve().parents[3] / "vllm_ascend/quantization/configs/fp8_config.py"


class TestStep5Fp8Config(unittest.TestCase):
    def setUp(self):
        class BaseConfig:
            def get_quant_method(self, layer, prefix, tid2eid=None):
                return ("dense", layer, prefix, tid2eid)

        tree = ast.parse(CONFIG_FILE.read_text())
        original = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Step5Fp8BlockConfig")
        method = next(n for n in original.body if isinstance(n, ast.FunctionDef) and n.name == "get_quant_method")
        isolated = ast.ClassDef(
            name=original.name,
            bases=[ast.Name(id="BaseConfig", ctx=ast.Load())],
            keywords=[],
            body=[method],
            decorator_list=[],
        )
        namespace = {
            "__package__": "vllm_ascend.quantization.configs",
            "BaseConfig": BaseConfig,
            "is_fused_moe_layer": lambda layer: hasattr(layer, "moe_config"),
            "_IS_LAYER_SKIPPED_SUPPORTS_MATCH_MODE": True,
            "is_layer_skipped": self.is_skipped,
            "FP8_METHOD": "fp8",
            "get_scheme_class": lambda *args: lambda block, config: ("scheme", block, config),
        }
        module = ast.Module(
            body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), isolated],
            type_ignores=[],
        )
        exec(compile(ast.fix_missing_locations(module), str(CONFIG_FILE), "exec"), namespace)
        self.config = namespace[original.name]()
        self.config.ignored_layers = []
        self.config.packed_modules_mapping = {}
        self.config.weight_block_size = [128, 128]
        self.config._verify_block_quantization = lambda: None
        self.layer = types.SimpleNamespace(moe_config=object())
        self.skip_calls = []
        adapters = types.ModuleType("vllm_ascend.quantization.method_adapters")
        adapters.AscendFusedMoEMethod = lambda scheme, config, tid2eid=None: ("native", scheme, config, tid2eid)
        routed = types.ModuleType("vllm_ascend.ops.fused_moe.routed_experts")
        routed.AscendUnquantizedFusedMoEMethod = lambda config, tid2eid=None: ("bf16", config, tid2eid)
        self.imports = {adapters.__name__: adapters, routed.__name__: routed}

    def is_skipped(self, prefix, ignored, mapping, *, match_mode="exact"):
        self.skip_calls.append((prefix, match_mode))
        if match_mode == "suffix":
            return any(prefix == name or prefix.endswith("." + name) for name in ignored)
        return prefix in ignored

    def get_method(self, layer=None, prefix="model.layers.88.moe.experts"):
        with patch.dict(sys.modules, self.imports):
            return self.config.get_quant_method(layer or self.layer, prefix, tid2eid="expert-map")

    def test_ignored_expert_module_or_checkpoint_weights_stay_bf16(self):
        for ignored in (
            "model.layers.88.moe",
            "model.layers.88.moe.experts",
            "model.layers.88.moe.experts.w13_weight",
            "model.layers.88.moe.experts.w2_weight",
        ):
            with self.subTest(ignored=ignored):
                self.config.ignored_layers = [ignored]
                self.assertEqual(self.get_method(), ("bf16", self.layer.moe_config, "expert-map"))

    def test_skip_honors_framework_match_mode(self):
        self.config.ignored_layers = ["layers.88.moe.experts.w13_weight"]
        self.config.ignored_layers_match_mode = "suffix"
        self.assertEqual(self.get_method()[0], "bf16")
        self.assertTrue(all(mode == "suffix" for _, mode in self.skip_calls))

    def test_router_exclusion_keeps_native_experts(self):
        self.config.ignored_layers = ["model.layers.88.moe.gate"]
        self.assertEqual(self.get_method()[0], "native")

    def test_dense_layers_delegate_to_ascend_config(self):
        layer = object()
        self.assertEqual(self.get_method(layer, "model.layers.88.self_attn.qkv_proj")[0], "dense")


if __name__ == "__main__":
    unittest.main()
