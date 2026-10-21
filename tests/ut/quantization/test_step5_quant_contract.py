"""Step5 quantization contracts: fp8 MoE, ModelSlim/VLM loading, mapping.

Merged from 5 source files per design spec section 5.2. AST
isolation is preserved: class definitions are compiled from source so these
run on CPU-only hosts as well as in the NPU unit-test suite.

Path note: the source files used ``Path(__file__).resolve().parents[N]`` to
reach the repo root, and N differs with directory depth. Every such
expression is rewritten to REPO_ROOT below, so this file is depth
independent.
"""



from pathlib import Path

# Repo root of vllm-ascend-private.
REPO_ROOT = Path(__file__).resolve().parents[3]

import ast
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock
import pytest
import torch
from types import SimpleNamespace
from unittest.mock import patch
from vllm.model_executor.model_loader.mtp_validation import (
    disable_mtp_completeness_check,
    is_mtp_completeness_check_enabled,
)
from vllm.models.step5.nvidia.model import Step5ForCausalLM, Step5FusedQKVIndexerLinear
from vllm_ascend.models.step5.vl import AscendStep5ForConditionalGeneration
from vllm.models.step5.nvidia.model import Step5Model
from vllm_ascend.models.step5.model import AscendStep5Model
from vllm.model_executor.models.step3p7 import Step3p7ForConditionalGeneration
from vllm.models.step5.nvidia.model import Step5ForCausalLM
from vllm_ascend.quantization.configs.modelslim_config import AscendModelSlimConfig, get_quant_type_for_layer
from unittest.mock import MagicMock
from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader.dummy_loader import DummyModelLoader
from vllm_ascend.quantization.method_adapters import AscendFusedMoEMethod
from vllm_ascend.quantization.methods.base import AscendMoEScheme, QuantType

ROOT = REPO_ROOT
OPS = ROOT / "vllm_ascend/ops/fused_moe"
SCHEME = ROOT / "vllm_ascend/quantization/methods/w8a8/fp8_block.py"


# ========================================================================
# from tests/ut/models/step5/test_fp8_moe_contract.py
# ========================================================================

def _definitions(path, names, namespace, methods=None, strip_decorators=False):
    tree = ast.parse(path.read_text())
    selected = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    assert len(selected) == len(names), f"Missing definitions in {path}"
    for node in selected:
        if strip_decorators:
            node.decorator_list = []
        if methods is not None and isinstance(node, ast.ClassDef):
            node.bases = []
            node.body = [
                n
                for n in node.body
                if isinstance(n, (ast.Assign, ast.AnnAssign)) or isinstance(n, ast.FunctionDef) and n.name in methods
            ]
    tree.body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *selected]
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), namespace)


def _quantize(x):
    # Same per-token/per-128 scale rule as the vendored act_quant kernel.
    assert x.shape[0] > 0, "The scheme must handle an empty batch before launching a kernel"
    scales, blocks = [], []
    for start in range(0, x.shape[-1], 128):
        values = x[:, start : start + 128].float()
        scale = values.abs().amax(dim=-1).clamp(min=1e-10) / 448.0
        blocks.append((values / scale[:, None]).clamp(-448, 448))
        scales.append(scale)
    return torch.cat(blocks, dim=-1).to(torch.float8_e4m3fn), torch.stack(scales, dim=-1)


def _gemm(x, x_scale, weight, weight_scale):
    # Fold scales after each K-block dot, as in the reference FP8 kernel.
    result = torch.zeros(x.shape[0], weight.shape[0])
    for start in range(0, x.shape[-1], 128):
        dot = x[:, start : start + 128].float() @ weight[:, start : start + 128].float().T
        row_scale = weight_scale[:, start // 128].repeat_interleave(128)[: weight.shape[0]]
        result += dot * x_scale[:, start // 128, None] * row_scale[None, :]
    return result.to(torch.bfloat16)


def _activation(x, limit=7.0):
    gate, up = x.chunk(2, dim=-1)
    return (torch.nn.functional.silu(gate.float()).clamp(max=limit) * up.float().clamp(-limit, limit)).to(x.dtype)


@pytest.fixture
def api(monkeypatch):
    class Activation(Enum):
        SILU = "silu"
        SITU = "situ"
        SWIGLUOAI = "swigluoai"
        SWIGLUOAI_UNINTERLEAVE = "swigluoai_uninterleave"
        SWIGLUSTEP = "swiglustep"
        GELU = "gelu"
        GELU_TANH = "gelu_tanh"

    class CommType(Enum):
        ALLGATHER = 0
        MC2 = 1
        ALLTOALL = 2

    events = []

    def record_event():
        event = object()
        events.append(event)
        return event

    monkeypatch.setattr(
        torch, "npu", SimpleNamespace(current_stream=lambda: SimpleNamespace(record_event=record_event)), raising=False
    )
    config = SimpleNamespace(load_config=SimpleNamespace(load_format="safetensors"))

    def replace_parameter(layer, name, value):
        setattr(layer, name, torch.nn.Parameter(value, requires_grad=False))

    namespace = dict(
        __name__=__name__,
        torch=torch,
        Enum=Enum,
        dataclass=dataclass,
        MoEActivation=Activation,
        logger=Mock(),
        get_current_vllm_config=lambda: config,
        _aclnn_block_fp8_available=lambda: False,
        replace_parameter=replace_parameter,
        cdiv=lambda x, y: (x + y - 1) // y,
        BLOCK_FP8_WEIGHT_DTYPE=torch.float8_e4m3fn,
        AscendSwigluStepAndMul=SimpleNamespace(swiglustep_forward=_activation),
        build_token_dispatch_input=lambda **kwargs: kwargs["fused_experts_input"],
        enable_fusion_gmmswigluquant=lambda: False,
    )
    _definitions(ROOT / "vllm_ascend/quantization/quant_type.py", {"QuantType"}, namespace)
    for filename, names in (
        ("moe_quant", {"MoEMxfpParams", "MoEQuantParams", "_build_mxfp_params", "build_quant_params"}),
        ("router_input", {"MoeRouterInput"}),
        ("fused_experts", {"MoEWeights", "MoEFusedExpertsInput", "build_fused_experts_input"}),
        ("moe_mlp", {"MoEMlpComputeInput", "build_mlp_compute_input"}),
    ):
        _definitions(OPS / f"dataclass/{filename}.py", names, namespace)
    _definitions(OPS / "moe_mlp.py", {"apply_moe_mlp", "_unified_apply_activation"}, namespace)
    _definitions(OPS / "token_dispatcher.py", {"TokenDispatcherWithAllGather"}, namespace, methods={"token_combine"})
    _definitions(
        OPS / "moe_comm_method.py",
        {"MoECommMethod", "FusedExpertsResult"},
        namespace,
        methods={"fused_experts", "_apply_mlp"},
    )
    _definitions(
        ROOT / "vllm_ascend/quantization/methods/base.py",
        {"AscendMoEScheme"},
        namespace,
        methods={"supports_fused_activation"},
    )
    _definitions(SCHEME, {"Step5Fp8BlockFusedMoEMethod", "_to_kmajor"}, namespace, strip_decorators=True)

    def install(name, **attrs):
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)

    install("vllm_ascend.models.step5.fp8_kernels", act_quant=_quantize, fp8_gemm=_gemm)
    install(
        "vllm_ascend.ops.fused_moe.dataclass.fused_experts",
        build_fused_experts_input=namespace["build_fused_experts_input"],
    )
    grouped_calls = []

    def grouped_gemm(xs, weights, scales, x_scales, prefix):
        assert len(xs) == len(weights) == len(scales) == len(x_scales) == 1
        assert prefix.dtype == torch.int64
        x, weight, weight_scale, x_scale = xs[0], weights[0], scales[0], x_scales[0]
        assert x.dtype == weight.dtype == torch.float8_e4m3fn
        assert x_scale.dtype == weight_scale.dtype == torch.float32
        grouped_calls.append((weight, weight_scale, prefix.clone()))
        result = torch.empty(x.shape[0], weight.shape[2], dtype=torch.bfloat16)
        start = 0
        for expert, end in enumerate(prefix.tolist()):
            if end > start:
                result[start:end] = _gemm(x[start:end], x_scale[start:end], weight[expert].T, weight_scale[expert].T)
            start = end
        return [result]

    install("vllm_ascend.ops.grouped_matmul_v5", grouped_matmul_block_fp8=grouped_gemm)

    def unpermute(*, permuted_tokens, sorted_indices, probs):
        assert permuted_tokens.dtype == probs.dtype == torch.float32
        token_ids = torch.arange(probs.shape[0]).repeat_interleave(probs.shape[1])[sorted_indices]
        result = torch.zeros(probs.shape[0], permuted_tokens.shape[1], dtype=permuted_tokens.dtype)
        result.index_add_(0, token_ids, permuted_tokens * probs.flatten()[sorted_indices, None])
        return result

    namespace["DeviceOperator"] = SimpleNamespace(npu_moe_token_unpermute=unpermute)
    context = SimpleNamespace(moe_comm_method=None, moe_comm_type=CommType.MC2)
    install("vllm_ascend.ascend_forward_context", _EXTRA_CTX=context, MoECommType=CommType)
    namespace["MoECommType"] = CommType
    namespace.update(config=config, events=events, grouped_calls=grouped_calls, context=context, _EXTRA_CTX=context)
    return SimpleNamespace(**namespace)


def _layer(api):
    generator = torch.Generator().manual_seed(312)
    layer = torch.nn.Module()
    for name, shape in (("w13_weight", (3, 256, 256)), ("w2_weight", (3, 256, 128))):
        weight = (torch.randn(shape, generator=generator) * 4).to(torch.float8_e4m3fn)
        scale_shape = (shape[0], (shape[1] + 127) // 128, (shape[2] + 127) // 128)
        scale = torch.rand(scale_shape, generator=generator) * 0.1 + 0.1
        layer.register_parameter(name, torch.nn.Parameter(weight, requires_grad=False))
        layer.register_parameter(name + "_scale_inv", torch.nn.Parameter(scale, requires_grad=False))
    layer.ep_rank = 0
    layer.ascend_expert_map = None
    layer.global_redundant_expert_num = 0
    layer.ascend_mc2_mask = None
    layer.apply_router_weight_on_input = False
    layer.activation = api.MoEActivation.SWIGLUSTEP
    return layer


def _reference(layer, x, ids, weights):
    result = torch.zeros_like(x, dtype=torch.float32)
    for token in range(x.shape[0]):
        for slot in range(ids.shape[1]):
            expert = int(ids[token, slot])
            q, scale = _quantize(x[token : token + 1])
            gate_up = _gemm(q, scale, layer.w13_weight[expert], layer.w13_weight_scale_inv[expert])
            q, scale = _quantize(_activation(gate_up))
            out = _gemm(q, scale, layer.w2_weight[expert], layer.w2_weight_scale_inv[expert])
            result[token] += out[0].float() * weights[token, slot]
    return result


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("group_type", [0, 1])
def test_current_communication_pipeline_owns_weights_and_preserves_fp32_combine(api, native, group_type):
    layer = _layer(api)
    method = api.Step5Fp8BlockFusedMoEMethod((128, 128), None)
    x = torch.linspace(-6, 6, 3 * 256).reshape(3, 256).to(torch.bfloat16)
    ids = torch.tensor([[0, 2], [2, 0], [0, 0]])
    weights = torch.tensor([[0.10001, 0.89999], [0.70001, 0.29999], [0.33331, 0.66669]])
    expected = _reference(layer, x, ids, weights)
    api.Step5Fp8BlockFusedMoEMethod.process_weights_after_loading.__globals__["_aclnn_block_fp8_available"] = (
        lambda: native
    )
    method.process_weights_after_loading(layer)

    class CpuDispatcher(api.TokenDispatcherWithAllGather):
        def token_dispatch(self, token_dispatch_input):
            self.input = token_dispatch_input
            assert self.input.layer is layer
            assert self.input.quant.is_quant and not self.input.quant.dispatch_with_quant
            assert self.input.topk_weights is weights
            assert self.input.weights.w1 is layer.w13_weight
            assert self.input.weights.w1_scale is layer.w13_weight_scale_inv
            order = ids.flatten().argsort(stable=True)
            self.token_ids = torch.arange(x.shape[0]).repeat_interleave(ids.shape[1])[order]
            self.slot_weights = weights.flatten()[order]
            groups = torch.bincount(ids.flatten(), minlength=3)
            if group_type == 0:
                groups = groups.cumsum(0)
            return SimpleNamespace(
                hidden_states=x[self.token_ids],
                group_list=groups,
                group_list_type=group_type,
                dynamic_scale=None,
                topk_scales=None,
                combine_metadata=SimpleNamespace(topk_weights=weights, expanded_row_idx=order, restore_shape=x.shape),
            )

    comm = api.MoECommMethod()
    comm.moe_config = None
    comm.token_dispatcher = CpuDispatcher()
    api.context.moe_comm_method = comm
    api.context.moe_comm_type = api.MoECommType.ALLGATHER
    result = method.apply(layer, x, weights, ids)
    torch.testing.assert_close(result.routed_out, expected, rtol=0, atol=0)
    assert method.keep_fp32_topk_weights
    assert len(api.events) == 3
    assert result.before_gmm2_evt is api.events[1]
    assert result.shared_activation_overlap_start_evt is result.before_gmm2_evt
    if native:
        assert len(api.grouped_calls) == 2
        assert api.grouped_calls[0][0] is layer.w13_weight
        assert api.grouped_calls[1][0] is layer.w2_weight
    else:
        assert not api.grouped_calls


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("dummy", [False, True])
def test_post_load_preserves_checkpoint_bytes_and_only_repairs_dummy_scales(api, native, dummy):
    layer = _layer(api)
    original = {name: value.detach().clone() for name, value in layer.named_parameters()}
    method = api.Step5Fp8BlockFusedMoEMethod((128, 128), None)
    api.Step5Fp8BlockFusedMoEMethod.process_weights_after_loading.__globals__["_aclnn_block_fp8_available"] = (
        lambda: native
    )
    api.config.load_config.load_format = "dummy" if dummy else "safetensors"
    method.process_weights_after_loading(layer)
    for name, parameter in layer.named_parameters():
        expected = original[name]
        if "scale" in name and dummy:
            expected = torch.ones_like(expected)
        if native:
            expected = expected.transpose(1, 2).contiguous()
        assert parameter.dtype == expected.dtype
        assert not parameter.requires_grad
        assert parameter.is_contiguous()
        assert torch.equal(parameter.view(torch.uint8), expected.view(torch.uint8))
    assert method.use_aclnn_gmm == native


@pytest.mark.parametrize("group_type", [0, 1])
def test_global_expert_groups_are_windowed_to_owning_ep_rank(api, group_type):
    method = api.Step5Fp8BlockFusedMoEMethod((128, 128), None)
    counts = torch.tensor([8, 9, 10, 2, 0, 3], dtype=torch.int32)
    groups = counts.cumsum(0) if group_type == 0 else counts
    inputs = SimpleNamespace(group_list=groups, group_list_type=group_type, layer=SimpleNamespace(ep_rank=1))
    prefix = method._group_prefix(inputs, 3)
    assert prefix.dtype == torch.int64
    assert prefix.tolist() == [2, 2, 5]


@pytest.mark.parametrize("native", [False, True])
def test_empty_dispatch_executes_current_mlp_without_quantizer_or_gemm_launch(api, native):
    layer = _layer(api)
    method = api.Step5Fp8BlockFusedMoEMethod((128, 128), None)
    api.Step5Fp8BlockFusedMoEMethod.process_weights_after_loading.__globals__["_aclnn_block_fp8_available"] = (
        lambda: native
    )
    method.process_weights_after_loading(layer)
    inputs = SimpleNamespace(
        hidden_states=torch.empty(0, 256, dtype=torch.bfloat16),
        group_list=torch.zeros(3, dtype=torch.int32),
        group_list_type=1,
        layer=layer,
        quant=api.MoEQuantParams(quant_type=api.QuantType.FP8_BLOCK),
        lora_context=None,
        activation=api.MoEActivation.SWIGLUSTEP,
    )
    result, event = api.apply_moe_mlp(inputs, method)
    assert result.shape == (0, 256)
    assert result.dtype == torch.bfloat16
    assert event is api.events[0]
    assert not api.grouped_calls


def test_fallback_rejects_capture_instead_of_synchronizing_host_counts(api, monkeypatch):
    monkeypatch.setattr(torch.npu, "is_current_stream_capturing", lambda: True, raising=False)
    layer = _layer(api)
    method = api.Step5Fp8BlockFusedMoEMethod((128, 128), None)
    inputs = SimpleNamespace(
        hidden_states=torch.ones(1, 256, dtype=torch.bfloat16),
        group_list=torch.tensor([1, 0, 0]),
        group_list_type=1,
        layer=layer,
    )
    with pytest.raises(RuntimeError, match="graph capture requires"):
        method.apply_gmm1(inputs)


@pytest.mark.parametrize("comm_type", ["ALLGATHER", "MC2", "ALLTOALL"])
def test_only_allgather_promotes_projection_for_fp32_weighted_combine(api, comm_type):
    layer = _layer(api)
    method = api.Step5Fp8BlockFusedMoEMethod((128, 128), None)
    inputs = SimpleNamespace(group_list=torch.tensor([1, 0, 0]), group_list_type=1, layer=layer)
    quantized, scale = _quantize(torch.ones(1, 128, dtype=torch.bfloat16))
    api.context.moe_comm_type = getattr(api.MoECommType, comm_type)
    expected = _gemm(quantized, scale, layer.w2_weight[0], layer.w2_weight_scale_inv[0])
    output = method.apply_gmm2(inputs, quantized, scale)
    expected_dtype = torch.float32 if comm_type == "ALLGATHER" else torch.bfloat16
    assert output.dtype == expected_dtype
    torch.testing.assert_close(output.float(), expected.float(), rtol=0, atol=0)


def test_invalid_group_contract_fails_explicitly(api):
    method = api.Step5Fp8BlockFusedMoEMethod((128, 128), None)
    inputs = SimpleNamespace(group_list=torch.zeros(2), group_list_type=1, layer=SimpleNamespace(ep_rank=0))
    with pytest.raises(ValueError, match="one dispatch group"):
        method._group_prefix(inputs, 3)
    inputs.group_list_type = 2
    with pytest.raises(NotImplementedError, match="group_list_type=2"):
        method._group_prefix(inputs, 3)


@pytest.mark.parametrize("availability", [True, False, None])
def test_native_capability_probe_handles_an_unpackaged_wrapper(availability):
    import builtins

    original_import = builtins.__import__

    def import_optional(name, *args, **kwargs):
        if name == "vllm_ascend.ops.grouped_matmul_v5":
            if availability is None:
                raise ModuleNotFoundError("optional wrapper absent", name=name)
            return SimpleNamespace(block_fp8_gmm_available=lambda: availability)
        return original_import(name, *args, **kwargs)

    namespace = {"__builtins__": {**vars(builtins), "__import__": import_optional}}
    _definitions(SCHEME, {"_aclnn_block_fp8_available"}, namespace)
    assert namespace["_aclnn_block_fp8_available"]() is bool(availability)


def test_native_capability_probe_does_not_hide_a_broken_packaged_wrapper():
    import builtins

    original_import = builtins.__import__

    def broken_import(name, *args, **kwargs):
        if name == "vllm_ascend.ops.grouped_matmul_v5":
            raise ModuleNotFoundError("wrapper dependency absent", name="broken_native_dependency")
        return original_import(name, *args, **kwargs)

    namespace = {"__builtins__": {**vars(builtins), "__import__": broken_import}}
    _definitions(SCHEME, {"_aclnn_block_fp8_available"}, namespace)
    with pytest.raises(ModuleNotFoundError, match="wrapper dependency absent"):
        namespace["_aclnn_block_fp8_available"]()


# ========================================================================
# from tests/ut/quantization/test_step5_vlm_loading.py
# ========================================================================

class _TextModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Keep the real tracking/reset/assert methods without creating an NPU
        # projection. The stream consumer stands in for tensor copies only.
        self.projection = object.__new__(Step5FusedQKVIndexerLinear)
        torch.nn.Module.__init__(self.projection)
        self.projection._loaded_shards = set()
        self.calls = 0

    def load_weights(self, weights):
        self.calls += 1
        for name, _ in weights:
            self.projection._loaded_shards.add(name.removesuffix(".weight"))
        return {"projection.weight"}


def _vlm():
    model = object.__new__(AscendStep5ForConditionalGeneration)
    torch.nn.Module.__init__(model)
    language = object.__new__(Step5ForCausalLM)
    torch.nn.Module.__init__(language)
    language.config = SimpleNamespace(tie_word_embeddings=False)
    language.model = _TextModel()
    language.lm_head = torch.nn.Linear(1, 1, bias=False)
    model.language_model = language
    model.vision_model = torch.nn.Linear(1, 1, bias=False)
    return model


def _stream(*, split_projection=False, omitted=None):
    shards = ("q", "k", "v", "index_q", "index_k", "index_z", "index_g")
    for index, shard in enumerate(shards):
        if shard != omitted:
            yield f"model.{shard}.weight", torch.ones(1, 1)
        if split_projection and index == 2:
            yield "vision_model.weight", torch.full((1, 1), 2.0)
    yield "vision_model.weight", torch.full((1, 1), 2.0)
    # This later language group caused a second reset and an all-shards-missing
    # error even when the first group contained the complete projection.
    yield "lm_head.weight", torch.full((1, 1), 3.0)


@pytest.mark.parametrize("split_projection", [False, True])
def test_interleaved_vlm_stream_validates_once_after_all_language_groups(split_projection):
    model = _vlm()
    loaded = model.load_weights(_stream(split_projection=split_projection))
    assert "language_model.model.projection.weight" in loaded
    assert "language_model.lm_head.weight" in loaded
    assert "vision_model.weight" in loaded
    assert model.language_model.model.calls == (2 if split_projection else 1)
    torch.testing.assert_close(model.language_model.lm_head.weight, torch.full((1, 1), 3.0))
    assert not model.language_model.model.projection._loaded_shards
    assert is_mtp_completeness_check_enabled()


@pytest.mark.parametrize("omitted", ["q", "index_q", "index_g"])
def test_incomplete_reload_still_fails_and_cannot_reuse_previous_shards(omitted):
    model = _vlm()
    model.load_weights(_stream())
    with pytest.raises(ValueError, match=f"missing shards .*{omitted}"):
        model.load_weights(_stream(split_projection=True, omitted=omitted))
    assert not model.language_model.model.projection._loaded_shards
    assert is_mtp_completeness_check_enabled()
    model.load_weights(_stream())


def test_stream_exception_clears_tracking_and_restores_validation_scope():
    model = _vlm()

    def broken_stream():
        yield "model.q.weight", torch.ones(1, 1)
        raise RuntimeError("checkpoint read failed")

    with pytest.raises(RuntimeError, match="checkpoint read failed"):
        model.load_weights(broken_stream())
    assert not model.language_model.model.projection._loaded_shards
    assert is_mtp_completeness_check_enabled()
    with pytest.raises(ValueError, match="missing shards"):
        model.load_weights(_stream(omitted="q"))


def test_explicitly_disabled_caller_keeps_its_scope_and_original_behavior():
    model = _vlm()
    with disable_mtp_completeness_check():
        model.load_weights(_stream(omitted="q"))
        assert not is_mtp_completeness_check_enabled()
        assert model.language_model.model.projection._loaded_shards
    assert is_mtp_completeness_check_enabled()
    # A following ordinary transaction resets that record and still rejects
    # a missing shard; externally deferred work cannot donate completeness.
    with pytest.raises(ValueError, match="missing shards .*q"):
        model.load_weights(_stream(omitted="q"))
    assert not model.language_model.model.projection._loaded_shards


def test_real_projection_copy_still_records_shards_while_validation_is_deferred():
    projection = _vlm().language_model.model.projection
    parameter = torch.nn.Parameter(torch.zeros(1, 2), requires_grad=False)
    source = torch.tensor([[2.0, 3.0]])
    with (
        disable_mtp_completeness_check(),
        patch.object(Step5FusedQKVIndexerLinear, "_indexer_source_offset", return_value=(0, 1)),
        patch.object(Step5FusedQKVIndexerLinear, "_indexer_dest_offset", return_value=0),
    ):
        projection.weight_loader_v2(parameter, source, "index_q")
    torch.testing.assert_close(parameter, source)
    assert projection._loaded_shards == {"index_q"}
    assert is_mtp_completeness_check_enabled()


# ========================================================================
# from tests/ut/quantization/test_step5_modelslim_loading.py
# ========================================================================

def _model(quant_name="ascend", recipe="W8A8_MXFP8", group_size=32):
    model = object.__new__(AscendStep5Model)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(hidden_size=64, moe_intermediate_size=32)
    model.moe_num_experts = 4
    description = {
        f"language_model.model.layers.3.moe.{proj}.weight": recipe for proj in ("gate_proj", "up_proj", "down_proj")
    }
    description["group_size"] = group_size
    model.vllm_config = SimpleNamespace(
        quant_config=SimpleNamespace(get_name=lambda: quant_name, quant_description=description)
    )
    return model


def _capture(model, weights):
    # Consume the generator just as the inherited loader does, without an NPU.
    with patch.object(Step5Model, "load_weights", autospec=True, side_effect=lambda self, values: list(values)):
        return model.load_weights(iter(weights))


@pytest.mark.parametrize("projection", ["gate_proj", "up_proj", "down_proj"])
@pytest.mark.parametrize("scale", [False, True])
def test_flat_modelslim_tensors_are_unpacked_without_copy(projection, scale):
    out_size, in_size = (64, 32) if projection == "down_proj" else (32, 64)
    if scale:
        in_size //= 32
    dtype = torch.uint8 if scale else torch.float8_e4m3fn
    tensor = torch.empty((4 * out_size, in_size), dtype=dtype, device="meta")
    name = f"layers.3.moe.{projection}.weight" + ("_scale" if scale else "")
    result = _capture(_model(), [(name, tensor)])
    assert result[0][0] == name
    assert result[0][1].shape == (4, out_size, in_size)
    assert result[0][1].dtype == dtype
    assert result[0][1]._base is tensor


@pytest.mark.parametrize(
    "shape,dtype",
    [
        ((127, 64), torch.float8_e4m3fn),
        ((128, 63), torch.float8_e4m3fn),
        ((4, 31, 64), torch.float8_e4m3fn),
        ((128, 64), torch.bfloat16),
    ],
)
def test_modelslim_loader_rejects_wrong_weight_dimensions_or_dtype(shape, dtype):
    tensor = torch.empty(shape, dtype=dtype, device="meta")
    with pytest.raises(ValueError, match="Invalid Step5 ModelSlim expert tensor"):
        _capture(_model(), [("model.layers.3.moe.gate_proj.weight", tensor)])


@pytest.mark.parametrize("shape,dtype", [((128, 1), torch.uint8), ((128, 2), torch.float32)])
def test_modelslim_loader_rejects_wrong_scale_dimensions_or_dtype(shape, dtype):
    tensor = torch.empty(shape, dtype=dtype, device="meta")
    with pytest.raises(ValueError, match="Invalid Step5 ModelSlim expert tensor"):
        _capture(_model(), [("layers.3.moe.gate_proj.weight_scale", tensor)])


@pytest.mark.parametrize("group_size", [0, 3, "32"])
def test_modelslim_loader_rejects_invalid_scale_groups(group_size):
    tensor = torch.empty((128, 2), dtype=torch.uint8, device="meta")
    with pytest.raises(ValueError, match="Invalid Step5 ModelSlim group_size"):
        _capture(_model(group_size=group_size), [("layers.3.moe.gate_proj.weight_scale", tensor)])


@pytest.mark.parametrize(
    "quant_name,recipe,name",
    [
        ("fp8", "W8A8_MXFP8", "layers.3.moe.gate_proj.weight"),
        ("ascend", "FLOAT", "layers.3.moe.gate_proj.weight"),
        ("ascend", "W8A8_MXFP8", "layers.3.share_expert.gate_proj.weight"),
        ("ascend", "W8A8_MXFP8", "layers.3.moe.gate_proj.weight_scale_inv"),
    ],
)
def test_original_hf_float_and_unrelated_tensors_are_unchanged(quant_name, recipe, name):
    tensor = torch.empty((4, 32, 64), dtype=torch.bfloat16, device="meta")
    assert _capture(_model(quant_name=quant_name, recipe=recipe), [(name, tensor)])[0][1] is tensor


def test_valid_3d_modelslim_tensor_and_no_quant_config_are_unchanged():
    tensor = torch.empty((4, 32, 64), dtype=torch.float8_e4m3fn, device="meta")
    model = _model()
    assert _capture(model, [("layers.3.moe.gate_proj.weight", tensor)])[0][1] is tensor
    model.vllm_config.quant_config = None
    assert _capture(model, [("layers.3.moe.gate_proj.weight", tensor)])[0][1] is tensor


def test_flat_scale_unpack_preserves_expert_order_and_bytes():
    tensor = torch.arange(4 * 32 * 2).remainder(255).to(torch.uint8).reshape(128, 2)
    unpacked = _capture(_model(), [("layers.3.moe.gate_proj.weight_scale", tensor)])[0][1]
    assert unpacked.data_ptr() == tensor.data_ptr()
    for expert_id in range(4):
        torch.testing.assert_close(unpacked[expert_id], tensor[expert_id * 32 : (expert_id + 1) * 32])


# ========================================================================
# from tests/ut/quantization/configs/test_step5_modelslim_mapping.py
# ========================================================================

@pytest.fixture
def description():
    # A quantized routed layer, a FLOAT routed layer, and a FLOAT shared expert.
    return {
        **{f"model.layers.3.moe.{proj}.weight": "W8A8_MXFP8" for proj in ("gate_proj", "up_proj", "down_proj")},
        **{f"model.layers.88.moe.{proj}.weight": "FLOAT" for proj in ("gate_proj", "up_proj", "down_proj")},
        **{f"model.layers.3.share_expert.{proj}.weight": "FLOAT" for proj in ("gate_proj", "up_proj", "down_proj")},
    }


@pytest.mark.parametrize("model_type", ["step5", "step5_vl"])
def test_step5_packed_experts_preserve_quantized_and_float_layers(description, model_type):
    config = AscendModelSlimConfig(description)
    config.packed_modules_mapping = {}
    config._update_packed_modules_mapping(model_type)
    for index, expected in ((3, "W8A8_MXFP8"), (88, None)):
        assert (
            get_quant_type_for_layer(
                config.quant_description, f"model.layers.{index}.moe.experts", config.packed_modules_mapping
            )
            == expected
        )


@pytest.mark.parametrize("model_type", ["step5", "step5_vl"])
@pytest.mark.parametrize("malformation,exception", [("mixed", ValueError), ("missing", KeyError)])
def test_step5_packed_experts_reject_incomplete_or_mixed_recipes(description, model_type, malformation, exception):
    if malformation == "mixed":
        description["model.layers.3.moe.up_proj.weight"] = "FLOAT"
    else:
        del description["model.layers.3.moe.down_proj.weight"]
    config = AscendModelSlimConfig(description)
    config.packed_modules_mapping = {}
    config._update_packed_modules_mapping(model_type)
    with pytest.raises(exception):
        get_quant_type_for_layer(config.quant_description, "model.layers.3.moe.experts", config.packed_modules_mapping)


def test_step5_vlm_outer_and_inner_mappers_preserve_routed_and_shared_recipes(description):
    config = AscendModelSlimConfig(description)
    # initialize_model configures the VLM first, then the nested causal model.
    config.apply_vllm_mapper(Step3p7ForConditionalGeneration.hf_to_vllm_mapper.get_rename_mapper())
    config.apply_vllm_mapper(Step5ForCausalLM.hf_to_vllm_mapper.get_rename_mapper())
    config.packed_modules_mapping = {
        name: list(shards) for name, shards in Step5ForCausalLM.packed_modules_mapping.items()
    }
    config._update_packed_modules_mapping("step5")
    assert (
        get_quant_type_for_layer(
            config.quant_description, "language_model.model.layers.3.moe.experts", config.packed_modules_mapping
        )
        == "W8A8_MXFP8"
    )
    assert (
        get_quant_type_for_layer(
            config.quant_description, "language_model.model.layers.88.moe.experts", config.packed_modules_mapping
        )
        is None
    )
    assert "language_model.model.layers.3.moe.share_expert.gate_proj.weight" in config.quant_description
    assert (
        get_quant_type_for_layer(
            config.quant_description,
            "language_model.model.layers.3.moe.share_expert.gate_up_proj",
            config.packed_modules_mapping,
        )
        is None
    )


# ========================================================================
# from tests/ut/quantization/test_mxfp8_dummy_scales.py
# ========================================================================

def _create_layer(quant_type=QuantType.W8A8MXFP, dtype=torch.uint8):
    scheme = MagicMock(spec=AscendMoEScheme)
    scheme.quant_type = quant_type
    scheme.group_size = 32
    scheme.get_weight.return_value = {}
    scheme.get_dynamic_quant_param.return_value = {
        name: torch.full((2, 64, 2), 3, dtype=dtype)
        for name in ("w13_weight_scale", "w2_weight_scale", "unrelated_scale")
    }
    method = AscendFusedMoEMethod(scheme, MagicMock())
    layer = torch.nn.Module()
    method.create_weights(layer, 2, 64, 64, torch.bfloat16)
    return layer


def test_mxfp8_dummy_loader_initializes_finite_neutral_scale_bytes():
    layer = _create_layer()
    for name in ("w13_weight_scale", "w2_weight_scale"):
        assert getattr(layer, name).dummy_weight_value == 127
    assert not hasattr(layer.unrelated_scale, "dummy_weight_value")

    # Call the actual loader: a state_dict-based imitation would lose metadata.
    DummyModelLoader(LoadConfig(load_format="dummy")).load_weights(layer, None)
    for name in ("w13_weight_scale", "w2_weight_scale"):
        torch.testing.assert_close(getattr(layer, name), torch.full((2, 64, 2), 127, dtype=torch.uint8))

    # Real checkpoint assignment still replaces the dummy default byte-for-byte.
    checkpoint = {name: torch.full_like(param, 120) for name, param in layer.named_parameters()}
    layer.load_state_dict(checkpoint)
    for param in layer.parameters():
        torch.testing.assert_close(param, torch.full_like(param, 120))


@pytest.mark.parametrize("quant_type", [QuantType.NONE, QuantType.W8A8, QuantType.W4A8MXFP, QuantType.FP8_BLOCK])
def test_non_mxfp8_schemes_do_not_receive_dummy_scale_metadata(quant_type):
    layer = _create_layer(quant_type=quant_type)
    assert all(not hasattr(param, "dummy_weight_value") for param in layer.parameters())


def test_floating_scales_do_not_receive_byte_encoded_dummy_value():
    layer = _create_layer(dtype=torch.float32)
    assert all(not hasattr(param, "dummy_weight_value") for param in layer.parameters())
