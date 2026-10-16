# SPDX-License-Identifier: Apache-2.0
"""Exercise the real dummy loader with ModelSlim's byte-encoded E8M0 scales."""

from unittest.mock import MagicMock

import pytest
import torch
from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader.dummy_loader import DummyModelLoader

from vllm_ascend.quantization.method_adapters import AscendFusedMoEMethod
from vllm_ascend.quantization.methods.base import AscendMoEScheme, QuantType


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
