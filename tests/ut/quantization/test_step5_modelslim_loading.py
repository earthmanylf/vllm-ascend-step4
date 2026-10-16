# SPDX-License-Identifier: Apache-2.0
"""Flat ModelSlim tensors must retain expert boundaries before upstream loading."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.models.step5.nvidia.model import Step5Model

from vllm_ascend.models.step5.model import AscendStep5Model


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
