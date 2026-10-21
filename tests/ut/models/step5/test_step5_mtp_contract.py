"""Step5 MTP contracts: multi-module speculator and draft quantization.

Merged from 2 source files per design spec section 5.2. AST
isolation is preserved: class definitions are compiled from source so these
run on CPU-only hosts as well as in the NPU unit-test suite.

Path note: the source files used ``Path(__file__).resolve().parents[N]`` to
reach the repo root, and N differs with directory depth. Every such
expression is rewritten to REPO_ROOT below, so this file is depth
independent.
"""

from __future__ import annotations

from pathlib import Path

# Repo root of vllm-ascend-private.
REPO_ROOT = Path(__file__).resolve().parents[4]

import importlib.util
import unittest
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
import torch
import ast
import logging
from copy import deepcopy
from dataclasses import dataclass, field, replace
import pytest
import torch.nn as nn

ROOT = REPO_ROOT


def methods(path, class_name, names, base, namespace):
    tree = ast.parse(path.read_text())
    original = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    body = [n for n in original.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in body} == set(names)
    cls = ast.ClassDef(
        name=class_name,
        bases=[ast.Name(id="TestBase", ctx=ast.Load())],
        keywords=[],
        body=body,
        decorator_list=[],
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            cls,
        ],
        type_ignores=[],
    )
    namespace = dict(namespace, TestBase=base)
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)  # noqa: S102
    return namespace[class_name]


class Mode:
    NONE = 0
    FULL = 1
    PIECEWISE = 2


MULTI = ROOT / "vllm_ascend/worker/v2/spec_decode/multi_module_mtp/speculator.py"
MIXIN = ROOT / "vllm_ascend/worker/v2/spec_decode/lmhead_tp_utils.py"
WRAPPER = ROOT / "vllm_ascend/worker/v2/aclgraph_utils.py"
ASCEND_ROOT = REPO_ROOT


# ========================================================================
# from tests/ut/worker/v2/test_multi_module_mtp_contract.py
# ========================================================================

def upstream_source():
    path = ROOT.parents[1] / "current/vllm/vllm/v1/worker/gpu/spec_decode/multi_module_mtp/speculator.py"
    if path.exists():
        return path
    spec = importlib.util.find_spec("vllm")
    if spec is not None and spec.submodule_search_locations:
        return (
            Path(next(iter(spec.submodule_search_locations)))
            / "v1/worker/gpu/spec_decode/multi_module_mtp/speculator.py"
        )
    raise unittest.SkipTest("Matching vLLM source is required")


class MultiModuleContract(unittest.TestCase):
    def test_upstream_generation_uses_all_three_model_and_sampling_heads(self):
        def cache(*args, **kwargs):
            pass

        updates = []
        cls = methods(
            upstream_source(),
            "MultiModuleMTPSpeculator",
            ["_generate_drafts", "_run_model"],
            object,
            {
                "torch": torch,
                "cache_inputs": cache,
                "CUDAGraphMode": Mode,
                "BatchDescriptor": lambda **kwargs: kwargs,
                "set_forward_context": lambda *a, **k: nullcontext(),
                "update_draft_inputs": lambda *a: updates.append(a),
            },
        )
        s = cls()
        s.last_token_indices = torch.tensor([0])
        s.input_buffers = SimpleNamespace(positions=torch.tensor([10]))
        s.idx_mapping = torch.tensor([2])
        s.inputs_embeds = None
        s.hidden_states = torch.zeros(1, 2)
        s.cached_draft_input_ids = None
        s.cached_draft_input_embeds = None
        s.cached_target_hidden_states = None
        s.num_speculative_steps = 3
        s.current_draft_step = torch.tensor(0)
        s.temperature = s.seeds = s.draft_logits = None
        s.draft_tokens = torch.zeros(1, 3, dtype=torch.int64)
        s.draft_input_id_overrides = torch.tensor([[-1, -1]])
        model_heads, sample_heads, samples_positions = [], [], []

        def forward(*args, spec_step_idx, **kwargs):
            model_heads.append(spec_step_idx)
            return torch.full((1, 2), float(spec_step_idx)), torch.full((1, 2), float(spec_step_idx + 1))

        def sample(*args, spec_step_idx):
            sample_heads.append(spec_step_idx)
            samples_positions.append(args[1].clone())
            return torch.tensor([20 + spec_step_idx])

        s.vllm_config = object()
        s.input_buffers.input_ids = torch.tensor([5])
        s.model = forward
        s.sample_draft = sample
        s._generate_drafts(1, 1, {}, {}, None)
        self.assertEqual(model_heads, [0, 1, 2])
        self.assertEqual(sample_heads, [0, 1, 2])
        self.assertEqual(s.draft_tokens.tolist(), [[20, 21, 22]])
        self.assertEqual([p.item() for p in samples_positions], [11, 12, 13])
        self.assertEqual(len(updates), 2)
        self.assertEqual(s.input_buffers.positions.tolist(), [10])

    def test_snapshot_exact_reprefill_lengths_once_in_batch_order(self):
        class Base:
            def _prepare_inputs(self):
                # Simulate the actual upstream GPU result for rejections0,1,3.
                self.input_buffers.seq_lens.values.copy_(torch.tensor([100, 200, 298, 0]))

        cls = methods(MULTI, "AscendMultiModuleMTPSpeculator", ["_prepare_inputs"], Base, {})
        s = cls()
        device = SimpleNamespace(values=torch.zeros(4, dtype=torch.int32))
        device.to = Mock(side_effect=lambda **kwargs: device.values.clone())
        s.input_buffers = SimpleNamespace(seq_lens=device, seq_lens_cpu=torch.full((4,), -9, dtype=torch.int32))
        s.idx_mapping = torch.tensor([7, 1, 9])
        s._prepare_inputs()
        device.to.assert_called_once_with(device="cpu")
        self.assertEqual(s.input_buffers.seq_lens_cpu.tolist(), [100, 200, 298, 0])

    def test_metadata_factory_gets_exact_lengths_and_cache_is_the_built_dict(self):
        factory_calls = []

        @contextmanager
        def factory(*args, **kwargs):
            factory_calls.append((args, kwargs))
            yield

        metadata = {f"head{i}": SimpleNamespace() for i in range(3)}

        class Base:
            def _build_attn_metadata(self, *args, **kwargs):
                return metadata

        cls = methods(
            MULTI,
            "AscendMultiModuleMTPSpeculator",
            ["_build_attn_metadata", "build_draft_attn_metadatas"],
            Base,
            {
                "torch": torch,
                "build_attn_metadata_factory": factory,
                "AscendAttentionState": SimpleNamespace(DecodeOnly="decode"),
            },
        )
        s = cls()
        s.draft_attn_state = "speculative"
        s.input_batch = SimpleNamespace(is_prefilling_np=np.array([False]))
        s.input_buffers = SimpleNamespace(positions=torch.tensor([0]), seq_lens_cpu=torch.tensor([298, 0]))
        s.draft_vllm_config = SimpleNamespace(parallel_config=object())
        desc = SimpleNamespace(num_tokens=4)
        self.assertIs(
            s._build_attn_metadata(1, desc, np.array([0, 4]), torch.tensor([300]), 0),
            metadata,
        )
        self.assertIs(factory_calls[0][1]["seq_lens_cpu"], s.input_buffers.seq_lens_cpu)
        self.assertEqual(factory_calls[0][1]["attn_state"], "speculative")
        self.assertIs(s.build_draft_attn_metadatas(1, 4, True)[0], metadata)
        self.assertTrue(all(md.attn_state == "speculative" for md in metadata.values()))

    def test_failed_proposal_clears_current_metadata(self):
        class Base:
            def propose(self, *args, **kwargs):
                self._multi_module_attn_metadata = {"stale": object()}
                raise RuntimeError("device failure")

        cls = methods(
            MULTI,
            "AscendMultiModuleMTPSpeculator",
            ["propose"],
            Base,
            {
                "build_attn_metadata_wrapper": nullcontext,
                "torch_gather_wrapper": nullcontext,
            },
        )
        s = cls()
        with self.assertRaisesRegex(RuntimeError, "device failure"):
            s.propose(object(), *([None] * 10))
        self.assertIsNone(s._multi_module_attn_metadata)

    def test_multi_token_state_preserves_sink_query_lengths(self):
        path = ROOT / "vllm_ascend/attention/attention_v1.py"
        if not path.exists():
            path = ROOT.parents[1] / "current/vllm-ascend/vllm_ascend/attention/attention_v1.py"
        state = SimpleNamespace(DecodeOnly="decode", SpecDecoding="speculative", PrefillNoCache="no-cache")
        fia = Mock(return_value=(torch.zeros(4, 1, 2), None))
        cls = methods(
            path,
            "AscendAttentionBackendImpl",
            ["forward_fused_infer_attention"],
            object,
            {
                "torch": torch,
                "torch_npu": SimpleNamespace(npu_fused_infer_attention_score_v2=fia),
                "_EXTRA_CTX": SimpleNamespace(capturing=False),
                "AscendAttentionState": state,
                "SWA_INT_MAX": 2147483647,
            },
        )
        s = cls()
        s.sinks = torch.zeros(1)
        s.sliding_window = 128
        s.num_heads = s.num_kv_heads = 1
        s.scale = 1
        q = torch.zeros(4, 1, 2)
        s._get_fia_params = lambda *a: (q, q, 128, torch.zeros(1, 1), [100])
        md = SimpleNamespace(actual_seq_lengths_q=[4], attn_state=state.SpecDecoding, attn_mask=None)
        s.forward_fused_infer_attention(q, q, q, md, torch.empty_like(q))
        self.assertEqual(fia.call_args.kwargs["actual_seq_qlen"], [4])

    def test_capture_factory_uses_same_multi_token_state_as_runtime(self):
        graph_path = ROOT / "vllm_ascend/worker/v2/spec_decode/autoregressive/aclgraph.py"
        prepared = []
        desc = SimpleNamespace(num_tokens=4, num_reqs=1, cg_mode=Mode.FULL, max_query_len=4, uniform_token_count=4)

        def prepare(*args, **kwargs):
            md = {"head": SimpleNamespace(attn_state="decode")}
            prepared.append(md)
            return md, {}

        class Manager:
            @staticmethod
            def capture(manager, create, progress_bar_desc):
                create(desc, True)(Mode.NONE)
                create(desc, False)(Mode.NONE)

        cls = methods(
            graph_path,
            "AutoRegressiveAclGraphManager",
            ["capture"],
            object,
            {
                "communicator_switch": nullcontext,
                "model_capture_wrapper": lambda *a: nullcontext(),
                "CudaGraphManager": Manager,
                "CUDAGraphMode": Mode,
                "prepare_inputs_to_capture": prepare,
            },
        )
        g = cls()
        g.speculator = SimpleNamespace(draft_attn_state="speculative")
        g.is_draft_model_prefill = True
        g.dp_size = 1
        g.max_num_reqs = 1
        g.capture(lambda *a: None, *([None] * 5))
        self.assertEqual(len(prepared), 2)
        self.assertTrue(all(md["head"].attn_state == "speculative" for md in prepared))

    def test_capture_delegates_to_upstream_merged_generator(self):
        base = methods(
            upstream_source(),
            "MultiModuleMTPSpeculator",
            ["capture"],
            object,
            {"logger": SimpleNamespace(info=lambda *a: None)},
        )
        cls = methods(
            MULTI, "AscendMultiModuleMTPSpeculator", ["capture"], base, {"build_attn_metadata_wrapper": nullcontext}
        )
        s = cls()
        s.last_token_indices = torch.tensor([9])
        s.idx_mapping = torch.tensor([9])
        s.cudagraph_manager = SimpleNamespace(use_breakable_cg=False, capture=Mock())
        s._generate_drafts = Mock()
        s.model_state = s.input_buffers = s.block_tables = s.attn_groups = s.kv_cache_config = object()
        s.capture()
        self.assertIs(s.cudagraph_manager.capture.call_args.args[0], s._generate_drafts)
        self.assertEqual(s.idx_mapping.tolist(), [0])
        self.assertEqual(s.last_token_indices.tolist(), [0])
        self.assertFalse(hasattr(s, "prefill_cudagraph_manager"))
        self.assertFalse(hasattr(s, "decode_cudagraph_manager"))

    def test_skip_attention_dummy_is_eager_and_success_clears_metadata(self):
        seen = []

        class Base:
            def propose(self, *args, **kwargs):
                seen.append(kwargs)
                self._multi_module_attn_metadata = {"live": object()}
                return "drafts"

        cls = methods(
            MULTI,
            "AscendMultiModuleMTPSpeculator",
            ["propose"],
            Base,
            {"build_attn_metadata_wrapper": nullcontext, "torch_gather_wrapper": nullcontext},
        )
        s = cls()
        result = s.propose(object(), *([None] * 10), dummy_run=True, skip_attn_for_dummy_run=True)
        self.assertEqual(result, "drafts")
        self.assertTrue(seen[0]["is_profile"])
        self.assertIsNone(s._multi_module_attn_metadata)

    def test_lmhead_and_capture_wrapper_forward_head_index(self):
        class Base:
            def sample_draft(self, *args, **kwargs):
                return kwargs

        for enabled in (False, True):
            cls = methods(
                MIXIN,
                "LmheadTPDraftSamplingMixin",
                ["sample_draft"],
                Base,
                {
                    "lmhead_tp_enable": lambda enabled=enabled: enabled,
                    "lmhead_tp_pad_rows": lambda x, *a: x,
                },
            )
            s = cls()
            s._lmhead_tp_max_num_logits = lambda: 4
            if enabled:
                # Padding branch trims tensor outputs, so capture kwargs via fake.
                Base.sample_draft = lambda self, *args, **kwargs: torch.tensor([kwargs["spec_step_idx"]])
            result = s.sample_draft(torch.zeros(1, 2), None, None, None, None, None, None, spec_step_idx=2)
            self.assertEqual(
                result.tolist() if enabled else result,
                [2] if enabled else {"spec_step_idx": 2},
            )
        cls = methods(
            WRAPPER,
            "ModelWithContext",
            ["compute_logits", "get_top_tokens"],
            object,
            {},
        )
        wrapper = cls()
        wrapper.original_model = SimpleNamespace(
            compute_logits=Mock(return_value=1), get_top_tokens=Mock(return_value=2)
        )
        for head in range(3):
            self.assertEqual(wrapper.compute_logits(None, spec_step_idx=head), 1)
            wrapper.original_model.compute_logits.assert_called_with(None, spec_step_idx=head)
            self.assertEqual(wrapper.get_top_tokens(None, spec_step_idx=head), 2)
            wrapper.original_model.get_top_tokens.assert_called_with(None, spec_step_idx=head)


class ARRejectionContract(unittest.TestCase):
    def test_once_per_proposal_snapshot_corrects_all_steps_in_batch_order(self):
        path = ROOT / "vllm_ascend/worker/v2/spec_decode/autoregressive/speculator.py"
        seen = []

        class Base:
            def propose(self, *args, **kwargs):
                for step in (1, 2):
                    seen.append(self._calc_next_seq_lens_cpu(torch.tensor([10, 20, 98, 66]), 3, 4, step))
                return "drafts"

        cls = methods(
            path,
            "AscendAutoRegressiveSpeculator",
            ["propose", "_calc_next_seq_lens_cpu"],
            Base,
            {
                "build_attn_metadata_wrapper": nullcontext,
                "torch_gather_wrapper": nullcontext,
                "disable_target_pcp_for_replicated_draft": lambda *a: nullcontext(),
            },
        )
        s = cls()
        s.replicated_pcp = False
        s.max_model_len = 100
        s.idx_mapping = torch.tensor([9, 1, 7])
        rejection = SimpleNamespace()
        rejection.to = Mock(return_value=torch.tensor([0, 1, 3]))

        class Device:
            def __getitem__(self, key):
                return rejection

        result = s.propose(SimpleNamespace(num_reqs=3), {}, {}, None, None, None, Device(), None, None, None, None)
        self.assertEqual(result, "drafts")
        rejection.to.assert_called_once_with(device="cpu", copy=True)
        self.assertEqual([x.tolist() for x in seen], [[11, 20, 96, 0], [12, 21, 97, 0]])
        self.assertIsNone(s._num_rejected_cpu)
        # Capture/dummy with no cache treats rejection as zero, with clamping.
        self.assertEqual(s._calc_next_seq_lens_cpu(torch.tensor([99, -8, 0]), 2, 3, 3).tolist(), [100, 0, 0])

    def test_failed_proposal_clears_rejection_snapshot(self):
        path = ROOT / "vllm_ascend/worker/v2/spec_decode/autoregressive/speculator.py"

        class Base:
            def propose(self, *args, **kwargs):
                raise RuntimeError("device failure")

        cls = methods(
            path,
            "AscendAutoRegressiveSpeculator",
            ["propose"],
            Base,
            {
                "build_attn_metadata_wrapper": nullcontext,
                "torch_gather_wrapper": nullcontext,
                "disable_target_pcp_for_replicated_draft": lambda *a: nullcontext(),
            },
        )
        s = cls()
        s.replicated_pcp = False
        with self.assertRaisesRegex(RuntimeError, "device failure"):
            s.propose(SimpleNamespace(num_reqs=1), {}, {}, None, None, None, torch.tensor([3]), None, None, None, None)
        self.assertIsNone(s._num_rejected_cpu)


if __name__ == "__main__":
    unittest.main(verbosity=2)


# ========================================================================
# from tests/ut/models/step5/test_mtp_quantization_contract.py
# ========================================================================

def _vllm_source(relative):
    spec = importlib.util.find_spec("vllm")
    assert spec is not None and spec.submodule_search_locations
    for root in spec.submodule_search_locations:
        path = Path(root) / relative
        if path.is_file():
            return path
    raise AssertionError(f"Missing installed vLLM source: {relative}")


def _extract(path, names, namespace):
    tree = ast.parse(path.read_text())
    body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names:
            body.append(node)
        elif isinstance(node, ast.ClassDef):
            body.extend(child for child in node.body if isinstance(child, ast.FunctionDef) and child.name in names)
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])),
            str(path),
            "exec",
        ),
        namespace,
    )


def runtime():
    namespace = {
        "dataclass": dataclass,
        "field": field,
        "logger": logging.getLogger(__name__),
        "deepcopy": deepcopy,
        "replace": replace,
        "nn": nn,
    }
    _extract(_vllm_source("model_executor/models/utils.py"), {"WeightsMapper"}, namespace)
    _extract(
        _vllm_source("models/step5/nvidia/mtp.py"),
        {"_rewrite_spec_layer_name"},
        namespace,
    )
    namespace["Step5MTP"] = type(
        "Step5MTP",
        (nn.Module,),
        {"_rewrite_spec_layer_name": namespace["_rewrite_spec_layer_name"]},
    )
    config_path = ASCEND_ROOT / "vllm_ascend/quantization/configs/modelslim_config.py"
    _extract(
        config_path,
        {"apply_vllm_mapper", "get_quant_type_for_layer", "_is_missing_v_shard"},
        namespace,
    )
    namespace["QuantConfig"] = type(
        "QuantConfig",
        (),
        {
            "apply_vllm_mapper": namespace["apply_vllm_mapper"],
            "_add_kvcache_quant_metadata": lambda self: None,
        },
    )
    namespace["_get_mtp_config"] = lambda config: config.model_config.hf_text_config
    namespace["maybe_prefix"] = lambda prefix, suffix: f"{prefix}.{suffix}" if prefix else suffix
    namespace["AscendStep5MultiTokenPredictor"] = lambda **kwargs: SimpleNamespace(
        mtp_start_layer_idx=kwargs["vllm_config"].model_config.hf_text_config.num_hidden_layers,
        layers={
            str(kwargs["vllm_config"].model_config.hf_text_config.num_hidden_layers): SimpleNamespace(
                shared_head=SimpleNamespace(head=nn.Identity())
            )
        },
    )
    namespace["_get_step5_mtp_moe_blocks"] = lambda model: []
    namespace["_set_step5_moe_protocol_metadata"] = lambda *args: None
    _extract(ASCEND_ROOT / "vllm_ascend/models/step5/mtp.py", {"AscendStep5MTP"}, namespace)
    return SimpleNamespace(**namespace)


class Config:
    quant_config: object
    model_config: object


def _config(runtime, description, *, start=92, count=3):
    quant = runtime.QuantConfig()
    quant.quant_description = deepcopy(description)
    quant._mapper_applied = False
    quant.hf_to_vllm_mapper = None
    return Config(
        quant,
        SimpleNamespace(hf_text_config=SimpleNamespace(num_hidden_layers=start, num_nextn_predict_layers=count)),
    )


def test_three_heads_map_real_packed_lookup_and_shared_head(runtime, prefix):
    description = {
        "optional": {"model.layers.92.self_attn.q_proj.weight": "metadata"},
        "group_size": 32,
    }
    for layer in range(92, 95):
        for name in (
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "moe.gate_proj",
            "moe.up_proj",
        ):
            description[f"{prefix}model.layers.{layer}.{name}.weight"] = "FLOAT"
        for name in (
            "enorm",
            "hnorm",
            "eh_proj",
            "transformer.shared_head.output",
            "transformer.shared_head.norm",
        ):
            description[f"{prefix}model.layers.{layer}.{name}.weight"] = "FLOAT"
    config = _config(runtime, description)
    model = runtime.AscendStep5MTP(vllm_config=config)
    mapped = model.vllm_config.quant_config.quant_description
    for layer in range(92, 95):
        for projection, shards in (
            ("qkv_proj", ["q_proj", "k_proj", "v_proj"]),
            ("gate_up_proj", ["gate_proj", "up_proj"]),
        ):
            block = "self_attn" if projection == "qkv_proj" else "moe"
            assert (
                runtime.get_quant_type_for_layer(
                    mapped,
                    f"model.layers.{layer}.mtp_block.{block}.{projection}",
                    {projection: shards},
                )
                is None
            )
        for name in (
            "enorm",
            "hnorm",
            "eh_proj",
            "shared_head.head",
            "shared_head.norm",
        ):
            assert mapped[f"model.layers.{layer}.{name}.weight"] == "FLOAT"
            assert f"model.layers.{layer}.mtp_block.{name}.weight" not in mapped
    assert mapped["optional"] == description["optional"]
    assert mapped["group_size"] == 32
    assert config.quant_config.quant_description == description
    assert model.vllm_config is not config
    assert model.vllm_config.quant_config is not config.quant_config


def test_quantized_shards_preserve_type_and_missing_still_fails(runtime):
    description = {
        f"model.layers.92.self_attn.{shard}.weight": "W8A8_DYNAMIC" for shard in ("q_proj", "k_proj", "v_proj")
    }
    config = _config(runtime, description)
    model = runtime.AscendStep5MTP(vllm_config=config)
    mapped = model.vllm_config.quant_config.quant_description
    packed = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}
    assert (
        runtime.get_quant_type_for_layer(mapped, "model.layers.92.mtp_block.self_attn.qkv_proj", packed)
        == "W8A8_DYNAMIC"
    )
    del mapped["model.layers.92.mtp_block.self_attn.k_proj.weight"]
    with pytest.raises(KeyError, match="k_proj"):
        runtime.get_quant_type_for_layer(mapped, "model.layers.92.mtp_block.self_attn.qkv_proj", packed)


def test_checkpoint_51_float_entries_match_weight_loader_namespace(runtime):
    # Exact 17-name schema of each of the checkpoint's three MTP layers.
    names = (
        "mlp.up_proj",
        "mlp.down_proj",
        "mlp.gate_proj",
        "input_layernorm",
        "self_attn.o_proj",
        "self_attn.k_norm",
        "self_attn.q_proj",
        "self_attn.q_norm",
        "self_attn.g_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "transformer.shared_head.output",
        "transformer.shared_head.norm",
        "hnorm",
        "eh_proj",
        "enorm",
        "post_attention_layernorm",
    )
    description = {f"model.layers.{layer}.{name}.weight": "FLOAT" for layer in range(92, 95) for name in names}
    model = runtime.AscendStep5MTP(vllm_config=_config(runtime, description))
    mapped = model.vllm_config.quant_config.quant_description
    assert len(mapped) == 51 and set(mapped.values()) == {"FLOAT"}
    for name in description:
        layer = int(name.split(".")[2])
        expected = model._rewrite_spec_layer_name(layer, name)
        expected = expected.replace(".transformer.", ".").replace("shared_head.output", "shared_head.head")
        assert mapped[expected] == "FLOAT"
    for layer in range(92, 95):
        for block, projection, shards in (
            ("self_attn", "qkv_proj", ["q_proj", "k_proj", "v_proj"]),
            ("mlp", "gate_up_proj", ["gate_proj", "up_proj"]),
        ):
            assert (
                runtime.get_quant_type_for_layer(
                    mapped,
                    f"model.layers.{layer}.mtp_block.{block}.{projection}",
                    {projection: shards},
                )
                is None
            )


def test_only_actual_mtp_range_maps_and_mapping_is_idempotent(runtime):
    description = {
        "model.layers.6.self_attn.q_proj.weight": "W8A8_DYNAMIC",
        "layers.7.self_attn.q_proj.weight": "FLOAT",
        "model.layers.8.mtp_block.self_attn.q_proj.weight": "FLOAT",
        "model.layers.9.self_attn.q_proj.weight": "W8A8_DYNAMIC",
    }
    model = runtime.AscendStep5MTP(vllm_config=_config(runtime, description, start=7, count=2))
    mapped = model.vllm_config.quant_config.quant_description
    assert mapped["model.layers.6.self_attn.q_proj.weight"] == "W8A8_DYNAMIC"
    assert mapped["model.layers.9.self_attn.q_proj.weight"] == "W8A8_DYNAMIC"
    assert mapped["model.layers.7.mtp_block.self_attn.q_proj.weight"] == "FLOAT"
    assert mapped["model.layers.8.mtp_block.self_attn.q_proj.weight"] == "FLOAT"
    assert model._get_mtp_quantization_mapper(mapped).apply_dict(mapped) == mapped


def test_no_quantization_and_generic_quant_config_remain_supported(runtime):
    config = _config(runtime, {})
    config.quant_config = None
    assert runtime.AscendStep5MTP(vllm_config=config).vllm_config is config
    generic = SimpleNamespace(apply_vllm_mapper=lambda mapper: None)
    config.quant_config = generic
    assert runtime.AscendStep5MTP(vllm_config=config).vllm_config.quant_config is not generic
