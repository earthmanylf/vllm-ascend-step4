"""Step5 model-level contracts: execution, fp8 config, dummy input.

Merged from 3 source files per design spec section 5.2. AST
isolation is preserved: class definitions are compiled from source so these
run on CPU-only hosts as well as in the NPU unit-test suite.

Path note: the source files used ``Path(__file__).resolve().parents[N]`` to
reach the repo root, and N differs with directory depth. Every such
expression is rewritten to REPO_ROOT below, so this file is depth
independent.
"""

import ast
import copy
import itertools
import math
import re
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from vllm_ascend.worker.v2 import input_batch as module

REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = REPO_ROOT / "vllm_ascend/models/step5"
CONFIG_FILE = REPO_ROOT / "vllm_ascend/quantization/configs/fp8_config.py"


# ========================================================================
# from tests/ut/models/test_step5_execution_contract.py
# ========================================================================


def load_methods(filename, class_name, method_names, namespace, base=object):
    tree = ast.parse((MODEL_DIR / filename).read_text())
    original = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    methods = [n for n in original.body if isinstance(n, ast.FunctionDef) and n.name in method_names]
    namespace = {"_Base": base, **namespace}
    isolated = ast.ClassDef(
        name=class_name, bases=[ast.Name(id="_Base", ctx=ast.Load())], keywords=[], body=methods, decorator_list=[]
    )
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), isolated],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(MODEL_DIR / filename), "exec"), namespace)
    return namespace[class_name]


def load_function(filename, name, namespace):
    tree = ast.parse((MODEL_DIR / filename).read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), function],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(MODEL_DIR / filename), "exec"), namespace)
    return namespace[name]


class FakeLinear:
    def __init__(self, *args, **kwargs):
        self.params_dtype = "bf16"
        self.quant_method = FakeUnquantized()
        self.args = args
        self.kwargs = kwargs


class FakeUnquantized:
    pass


class ContractTensor:
    """Small numeric tensor for the forward's reshape, split and gate math."""

    def __init__(self, values, shape):
        self.values = list(values)
        self.shape = tuple(shape)
        assert len(self.values) == math.prod(self.shape)

    def contiguous(self):
        return self

    def view(self, *shape):
        if -1 in shape:
            known = math.prod(size for size in shape if size != -1)
            shape = tuple(len(self.values) // known if size == -1 else size for size in shape)
        return ContractTensor(self.values, shape)

    def split(self, sizes, dim):
        assert dim == -1
        rows = len(self.values) // self.shape[-1]
        chunks = []
        offset = 0
        for size in sizes:
            values = []
            for row in range(rows):
                begin = row * self.shape[-1] + offset
                values.extend(self.values[begin : begin + size])
            chunks.append(ContractTensor(values, (*self.shape[:-1], size)))
            offset += size
        assert offset == self.shape[-1]
        return tuple(chunks)

    def unsqueeze(self, dim):
        assert dim == -1
        return ContractTensor(self.values, (*self.shape, 1))

    def sigmoid(self):
        return ContractTensor([1 / (1 + math.exp(-value)) for value in self.values], self.shape)

    def __mul__(self, other):
        assert len(self.shape) == len(other.shape)
        values = []
        for coordinates in itertools.product(*(range(size) for size in self.shape)):
            left_index = right_index = 0
            for coordinate, left_size, right_size in zip(coordinates, self.shape, other.shape):
                assert right_size in (1, left_size)
                left_index = left_index * left_size + coordinate
                right_index = right_index * right_size + (coordinate if right_size > 1 else 0)
            values.append(self.values[left_index] * other.values[right_index])
        return ContractTensor(values, self.shape)


class TestStep5DenseAttention(unittest.TestCase):
    def setUp(self):
        self.calls = []

        def attention(*args, **kwargs):
            self.calls.append((args, kwargs))
            # An explicitly supplied backend would bypass Ascend platform selection.
            self.assertNotIn("attn_backend", kwargs)
            return types.SimpleNamespace()

        def per_layer(values, idx, **kwargs):
            return values[idx] if values and idx < len(values) else kwargs["default"]

        self.cls = load_methods(
            "model.py",
            "AscendStep5Attention",
            {"__init__"},
            {
                "nn": types.SimpleNamespace(Module=object),
                "AttentionType": types.SimpleNamespace(DECODER="decoder"),
                "get_tensor_model_parallel_world_size": lambda: 4,
                "get_tensor_model_parallel_rank": lambda: 0,
                "extract_layer_index": lambda prefix: int(prefix.split(".")[1]),
                "_per_layer_value": per_layer,
                "LinearBase": FakeLinear,
                "ColumnParallelLinear": FakeLinear,
                "MergedColumnParallelLinear": FakeLinear,
                "QKVParallelLinear": FakeLinear,
                "RowParallelLinear": FakeLinear,
                "UnquantizedLinearMethod": FakeUnquantized,
                "make_step5_rms_norm": lambda *args, **kwargs: object(),
                "get_rope": lambda **kwargs: types.SimpleNamespace(
                    cos_sin_cache=types.SimpleNamespace(chunk=lambda *args, **kwargs: ("cos", "sin"))
                ),
                "Attention": attention,
                "os": types.SimpleNamespace(environ={}),
            },
        )

    def construct(self, **overrides):
        kwargs = dict(
            hidden_size=256,
            num_heads=8,
            num_kv_heads=4,
            head_dim=32,
            prefix="layers.0.self_attn",
            sliding_window=512,
            layer_types=["sliding_attention", "full_attention"],
            cache_config=types.SimpleNamespace(block_size=128),
        )
        kwargs.update(overrides)
        return self.cls(**kwargs)

    def test_swa_preserves_window_and_uses_platform_backend(self):
        layer = self.construct(use_head_wise_attn_gate=True, swa_num_attention_heads=16)
        self.assertEqual(layer.total_num_heads, 16)
        self.assertEqual(layer.num_heads, 4)
        self.assertEqual(layer.tp_size, 4)
        self.assertTrue(layer.fuse_qkv_gate)
        self.assertEqual(self.calls[-1][1]["per_layer_sliding_window"], 512)
        self.assertEqual(layer.attn.kv_cache_prefix_recompute_tokens, 128)
        self.assertFalse(layer.use_dsa_backend)
        self.assertIsNone(layer.sparse_attn)

    def test_replicated_kv_heads_keep_separate_gate_projection(self):
        layer = self.construct(num_kv_heads=2, use_head_wise_attn_gate=True)
        self.assertFalse(layer.fuse_qkv_gate)
        self.assertIsInstance(layer.g_proj, FakeLinear)
        self.assertEqual(layer.num_kv_heads, 1)

    def test_dense_full_attention_does_not_apply_sliding_window(self):
        self.construct(prefix="layers.1.self_attn")
        self.assertIsNone(self.calls[-1][1]["per_layer_sliding_window"])

    def test_rejects_invalid_tp_and_rotary_geometry(self):
        for kwargs in (
            {"num_heads": 6},
            {"num_kv_heads": 3},
            {"partial_rotary_factor": 0},
            {"partial_rotary_factor": 0.1},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.construct(**kwargs)

    def test_sparse_layers_cannot_accidentally_use_dense_constructor(self):
        with self.assertRaises(ValueError):
            self.construct(sparse_config=object())

    def test_swa_forward_gate_runs_without_removed_custom_op(self):
        for fused_gate in (False, True):
            with self.subTest(fused_gate=fused_gate):
                self._check_forward_gate(fused_gate)

    def _check_forward_gate(self, fused_gate):
        cls = load_methods(
            "model.py",
            "AscendStep5Attention",
            {"forward"},
            {
                # The removed custom op is deliberately unavailable.
                "torch": types.SimpleNamespace(ops=types.SimpleNamespace(vllm=types.SimpleNamespace())),
                "fused_qknorm_rope": lambda qkv, *args, **kwargs: qkv.split([4, 2, 2], dim=-1),
            },
        )
        layer = cls()
        layer.use_dsa_backend = False
        layer.fuse_qkv_gate = fused_gate
        layer.use_fused_qknorm_rope = True
        layer.use_head_wise_attn_gate = True
        layer.sparse_attn = None
        layer.q_size, layer.kv_size = 4, 2
        layer.num_heads, layer.num_kv_heads, layer.head_dim = 2, 1, 2
        layer.q_norm = layer.k_norm = types.SimpleNamespace(weight=None, variance_epsilon=1e-6)
        layer.rope_cos = layer.rope_sin = None
        layer._fused_rotary_pairs = 1
        layer.zero_centered = True
        qkv = ContractTensor(range(16), (2, 8))
        gate = ContractTensor([0, math.log(3), math.log(1 / 3), 0], (2, 2))
        packed_values = qkv.values[:8] + gate.values[:2] + qkv.values[8:] + gate.values[2:]
        layer.qkvg_proj = lambda hidden: (ContractTensor(packed_values, (2, 10)), None)
        layer.qkv_proj = lambda hidden: (qkv, None)
        layer.g_proj = lambda hidden: (gate, None)
        attention_output = ContractTensor([2, 4, 6, 8, 1, 3, 5, 7], (2, 4))
        layer.attn = lambda q, k, v, **kwargs: attention_output
        layer.o_proj = lambda hidden: (hidden, None)

        result = layer.forward(object(), object())
        self.assertEqual(result.shape, (2, 4))
        expected = [1, 2, 4.5, 6, 0.25, 0.75, 2.5, 3.5]
        for actual, value in zip(result.values, expected):
            self.assertAlmostEqual(actual, value)


class TestStep5DenseFFN(unittest.TestCase):
    def test_dense_ffn_calls_mlp_without_old_upstream_keywords(self):
        class StrictUpstream:
            def _forward_ffn(self, hidden_states, padding=None):
                raise AssertionError("dense MLP must keep the Ascend reduction path")

        cls = load_methods("model.py", "AscendStep5DecoderLayer", {"_forward_ffn"}, {}, StrictUpstream)
        layer = cls()
        layer.use_moe = False
        seen = []
        result = object()
        layer.mlp = lambda hidden: seen.append(hidden) or result
        hidden = object()
        self.assertIs(layer._forward_ffn(hidden, residual=object(), orig_num_tokens=2), result)
        self.assertEqual(seen, [hidden])


class TestStep5MoEConstructor(unittest.TestCase):
    def test_shared_expert_keeps_tp_weights_when_runner_shards_tokens(self):
        for enable_sp in (False, True):
            with self.subTest(enable_sp=enable_sp):
                self._check_shared_expert(enable_sp)

    def test_vl_top_level_fp8_exclusions_reach_nested_moe(self):
        for layer_idx, expected_bf16 in ((3, False), (88, True), (89, True), (90, True)):
            with self.subTest(layer_idx=layer_idx):
                self._check_shared_expert(True, layer_idx, expected_bf16)

    def test_hf_fp8_exclusions_ignore_router_and_shared_experts(self):
        layout = load_function("model.py", "_step5_fp8_expert_layout", {"re": re})
        config = types.SimpleNamespace(
            quantization_config={
                "quant_method": "fp8",
                "weight_block_size": [128, 128],
                "modules_to_not_convert": [
                    "model.layers.87.moe.gate",
                    "model.layers.87.share_expert.down_proj",
                    "model.layers.88.moe.experts.w13_weight",
                    "model.layers.88.moe.experts.w2_weight",
                ],
            }
        )
        self.assertEqual(layout(config), ({88}, 128))

    def _check_shared_expert(self, enable_sp, layer_idx=3, expected_bf16=None):
        class QuantConfig:
            weight_block_size = [128, 128]
            ignored_layers = [
                f"language_model.model.layers.{idx}.moe.experts.{weight}"
                for idx in (88, 89, 90)
                for weight in ("w13_weight", "w2_weight")
            ]

            def get_name(self):
                return "fp8"

        class NativeQuantConfig(QuantConfig):
            pass

        class StrictStep5MLP:
            # Match the current SWF constructor. Sequence parallel in this
            # class replicates weights unless it owns the SP communication.
            def __init__(
                self,
                config,
                hidden_size,
                intermediate_size,
                hidden_act,
                quant_config=None,
                reduce_results=True,
                prefix="",
                sequence_parallel=False,
                can_shard_sequence_parallel=False,
                reduce_in_fp32=False,
            ):
                self.sequence_parallel = sequence_parallel
                self.disable_tp = sequence_parallel and not can_shard_sequence_parallel
                self.reduce_results = reduce_results
                self.intermediate_size = intermediate_size

        calls = []

        def factory(**kwargs):
            calls.append(kwargs)
            return types.SimpleNamespace(moe_config=types.SimpleNamespace())

        factory_module = types.ModuleType("vllm.model_executor.layers.fused_moe")
        factory_module.FusedMoEFactory = factory
        cls = load_methods(
            "model.py",
            "AscendStep5FusedMoEBlock",
            {"__init__"},
            {
                "nn": types.SimpleNamespace(Module=object, Parameter=lambda value, **kwargs: value),
                "torch": types.SimpleNamespace(float32="fp32", zeros=lambda size, **kwargs: (0,) * size),
                "get_tensor_model_parallel_world_size": lambda: 4,
                "get_ep_group": lambda: types.SimpleNamespace(
                    device_group=types.SimpleNamespace(size=lambda: 4, rank=lambda: 1)
                ),
                "get_dp_group": lambda: types.SimpleNamespace(world_size=1),
                "extract_layer_index": lambda prefix: int(prefix.split(".")[1]),
                "Step5FP32RouterLinear": FakeLinear,
                "envs": types.SimpleNamespace(VLLM_ASCEND_ENABLE_STEP5_FP8=expected_bf16 is not None),
                "_step5_moe_reduce_policy": lambda tp_size, dp_size: (True, False),
                "Step5MLP": StrictStep5MLP,
                "AscendFp8Config": QuantConfig,
                "Step5Fp8BlockConfig": NativeQuantConfig,
                "_step5_fp8_expert_layout": load_function("model.py", "_step5_fp8_expert_layout", {"re": re}),
                "copy": copy,
                "logger": types.SimpleNamespace(info=lambda *args: None),
            },
        )
        config = types.SimpleNamespace(
            hidden_size=4096,
            moe_num_experts=352,
            moe_intermediate_size=1536,
            share_expert_dim=1536,
            moe_top_k=8,
            use_moe_router_bias=True,
            need_fp32_gate=True,
            moe_router_scaling_factor=1.0,
            norm_expert_weight=True,
            swiglu_limits=[7.0] * 92,
            # The real VL checkpoint keeps this metadata at the top level.
            quantization_config=None,
        )
        vllm_config = types.SimpleNamespace(
            model_config=types.SimpleNamespace(hf_config=config),
            quant_config=QuantConfig() if expected_bf16 is not None else None,
            parallel_config=types.SimpleNamespace(
                enable_eplb=False,
                enable_expert_parallel=True,
                use_sequence_parallel_moe=enable_sp,
                eplb_config=types.SimpleNamespace(num_redundant_experts=0),
            ),
            additional_config={},
        )
        with patch.dict(sys.modules, {factory_module.__name__: factory_module}):
            layer = cls(vllm_config=vllm_config, prefix=f"layers.{layer_idx}.moe")
        self.assertFalse(layer.share_expert.sequence_parallel)
        self.assertFalse(layer.share_expert.disable_tp)
        self.assertFalse(layer.share_expert.reduce_results)
        self.assertEqual(layer.share_expert.intermediate_size, 1536)
        self.assertIs(calls[0]["shared_experts"], layer.share_expert)
        self.assertEqual(calls[0]["num_experts"], 352)
        self.assertEqual(calls[0]["intermediate_size"], 1536)
        self.assertEqual(calls[0]["is_sequence_parallel"], enable_sp)
        self.assertEqual(calls[0]["activation"], "swiglustep")
        self.assertFalse(calls[0]["reduce_results"])
        self.assertEqual(layer.physical_expert_start, 88)
        self.assertEqual(layer.physical_expert_end, 176)
        if expected_bf16 is not None:
            if expected_bf16:
                self.assertIsNone(calls[0]["quant_config"])
            else:
                self.assertIsInstance(calls[0]["quant_config"], NativeQuantConfig)
            self.assertIs(type(vllm_config.quant_config), QuantConfig)


class TestStep5MTP(unittest.TestCase):
    def test_draft_quant_mapping_preserves_target_and_own_head(self):
        class QuantConfig:
            def __init__(self):
                self.mappings = []

            def apply_vllm_mapper(self, mapper):
                self.mappings.append(mapper.orig_to_new_prefix)

        target_quant = QuantConfig()
        config = types.SimpleNamespace(quant_config=target_quant, model_config=object())
        head = object()
        predictor = types.SimpleNamespace(
            mtp_start_layer_idx=92,
            layers={"92": types.SimpleNamespace(shared_head=types.SimpleNamespace(head=head))},
        )
        seen = []

        def replace(original, **kwargs):
            result = copy.copy(original)
            result.__dict__.update(kwargs)
            return result

        cls = load_methods(
            "mtp.py",
            "AscendStep5MTP",
            {"__init__"},
            {
                "nn": types.SimpleNamespace(Module=object),
                "deepcopy": copy.deepcopy,
                "replace": replace,
                "WeightsMapper": types.SimpleNamespace,
                "_get_mtp_config": lambda cfg: object(),
                "_get_mtp_vllm_config": lambda cfg: cfg,
                "AscendStep5MultiTokenPredictor": lambda **kwargs: seen.append(kwargs["vllm_config"]) or predictor,
                "maybe_prefix": lambda prefix, name: name,
                "_get_step5_mtp_moe_blocks": lambda model: [],
                "_set_step5_moe_protocol_metadata": lambda *args: None,
            },
        )
        draft = cls(vllm_config=config)
        self.assertIs(draft.lm_head, head)
        self.assertTrue(draft.is_dense_mtp)
        self.assertIsNot(seen[0].quant_config, target_quant)
        self.assertEqual(target_quant.mappings, [])
        self.assertEqual(seen[0].quant_config.mappings, [{"language_model.": ""}])

    def test_predictor_uses_complete_ascend_decoder_output(self):
        for fp32_residual in (False, True):
            with self.subTest(fp32_residual=fp32_residual):
                self._check_predictor_output(fp32_residual)

    def _check_predictor_output(self, fp32_residual):
        dtype_changes = []

        class Tensor:
            def __init__(self, name):
                self.name = name

            def to(self, dtype):
                dtype_changes.append((self.name, dtype))
                return self

        concat_inputs = []

        def cat(inputs, dim):
            concat_inputs.append((inputs, dim))
            return object()

        cls = load_methods(
            "mtp.py",
            "AscendStep5MultiTokenPredictorLayer",
            {"forward"},
            {
                "torch": types.SimpleNamespace(cat=cat, bfloat16="bf16", float32="fp32"),
            },
        )
        layer = cls()
        layer.fp32_residual_connection = fp32_residual
        layer.enorm = lambda tensor: ("embedding", tensor)
        layer.hnorm = lambda tensor: ("hidden", tensor)
        projected = Tensor("projected")
        layer.eh_proj = lambda concat: projected
        output = object()
        positions = object()
        decoder_calls = []

        # No use_sequence_parallel or padding/tuple contract: this is
        # exactly the Ascend decoder's complete-tensor call signature.
        def block(*, positions, hidden_states):
            decoder_calls.append((positions, hidden_states))
            return output

        layer.mtp_block = block
        embeds, previous = Tensor("embeds"), Tensor("previous")
        self.assertIs(layer.forward(None, positions, previous, embeds), output)
        self.assertEqual(concat_inputs, [([("embedding", embeds), ("hidden", previous)], -1)])
        self.assertEqual(decoder_calls, [(positions, projected)])
        self.assertEqual(
            dtype_changes,
            [("embeds", "bf16"), ("previous", "bf16"), ("projected", "fp32")] if fp32_residual else [],
        )


class TestStep5VocabularyContract(unittest.TestCase):
    def test_target_and_draft_use_fixed_swf_logits_api(self):
        import inspect

        from vllm.model_executor.layers.logits_processor import LogitsProcessor

        # Use the installed fixed SWF signature, with no legacy ModelConfig
        # vocabulary field. Only expensive weight allocation is replaced.
        signature = inspect.signature(LogitsProcessor.__init__)
        seen = []

        def build_logits(*args, **kwargs):
            signature.bind(None, *args, **kwargs)
            seen.append((args, kwargs))
            return object()

        hf_config = types.SimpleNamespace(
            vocab_size=131072,
            hidden_size=4096,
            num_hidden_layers=92,
            num_nextn_predict_layers=3,
            fp32_residual_connection=True,
            tie_word_embeddings=False,
        )
        config = types.SimpleNamespace(
            model_config=types.SimpleNamespace(hf_config=hf_config),
            quant_config=None,
        )
        target_model = types.SimpleNamespace(layers=[], make_empty_intermediate_tensors=object())
        target = load_methods(
            "model.py",
            "AscendStep5ForCausalLM",
            {"__init__"},
            {
                "nn": types.SimpleNamespace(Module=object),
                "AscendStep5Model": lambda **kwargs: target_model,
                "validate_step5_kv_transfer_config": lambda config: None,
                "get_pp_group": lambda: types.SimpleNamespace(is_last_rank=True),
                "ParallelLMHead": lambda *args, **kwargs: object(),
                "LogitsProcessor": build_logits,
                "maybe_prefix": lambda prefix, name: name,
                "PPMissingLayer": type("Missing", (), {}),
                "Step5MoE": object,
                "_set_step5_moe_protocol_metadata": lambda *args: None,
            },
        )
        target(vllm_config=config)
        predictor = load_methods(
            "mtp.py",
            "AscendStep5MultiTokenPredictor",
            {"__init__"},
            {
                "nn": types.SimpleNamespace(Module=object),
                "torch": types.SimpleNamespace(nn=types.SimpleNamespace(ModuleDict=dict)),
                "_get_mtp_config": lambda cfg: hf_config,
                "VocabParallelEmbedding": lambda *args: object(),
                "AscendStep5MultiTokenPredictorLayer": lambda **kwargs: object(),
                "LogitsProcessor": build_logits,
            },
        )
        draft = predictor(vllm_config=config)
        self.assertEqual(set(draft.layers), {"92", "93", "94"})
        self.assertEqual(seen, [((131072,), {}), ((131072,), {})])


if __name__ == "__main__":
    unittest.main()


# ========================================================================
# from tests/ut/models/test_step5_fp8_config_contract.py
# ========================================================================


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


# ========================================================================
# from tests/ut/worker/v2/test_step5_dummy_input_contract.py
# ========================================================================


@pytest.mark.parametrize("is_padding", [False, True])
@pytest.mark.parametrize("physical_tokens", [5, 8])
def test_dummy_profile_and_graph_padding_keep_distinct_masks(monkeypatch, is_padding, physical_tokens):
    # Use the real Ascend buffers and the installed SWF make_dummy on CPU.
    # Rotary table setup is a device operation unrelated to this mask contract.
    positions = []
    monkeypatch.setattr(module, "update_cos_sin", lambda value: positions.append(value.clone()))
    buffers = module.AscendInputBuffers(max_num_reqs=2, max_num_tokens=8, device=torch.device("cpu"))
    buffers.dummy_num_tokens = 5
    buffers.is_padding.fill_(not is_padding)
    buffers.input_ids.fill_(99)
    buffers.positions.fill_(99)

    batch = module.AscendInputBatch.make_dummy(
        num_reqs=2,
        num_tokens=physical_tokens,
        input_buffers=buffers,
        max_query_len=3,
        is_padding=is_padding,
    )

    assert batch.num_tokens == 5
    assert batch.num_tokens_after_padding == physical_tokens
    assert batch.is_padding.tolist() == [is_padding] * 5 + [True] * (physical_tokens - 5)
    assert batch.input_ids.tolist() == [0] * physical_tokens
    assert batch.positions.tolist() == [0] * physical_tokens
    np.testing.assert_array_equal(batch.num_scheduled_tokens, [2, 3])
    np.testing.assert_array_equal(batch.seq_lens_np, [2, 3])
    assert len(positions) == 1
    assert batch.is_dummy


def test_dummy_warmup_defaults_to_padding(monkeypatch):
    monkeypatch.setattr(module, "update_cos_sin", lambda value: None)
    buffers = module.AscendInputBuffers(max_num_reqs=2, max_num_tokens=8, device=torch.device("cpu"))
    batch = module.AscendInputBatch.make_dummy(2, 5, buffers, max_query_len=3)
    assert batch.is_padding.tolist() == [True] * 5
