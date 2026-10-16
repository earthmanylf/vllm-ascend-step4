# SPDX-License-Identifier: Apache-2.0
"""Packed Step5 ModelSlim expert recipes must survive VLM namespace mapping."""

import pytest
from vllm.model_executor.models.step3p7 import Step3p7ForConditionalGeneration
from vllm.models.step5.nvidia.model import Step5ForCausalLM

from vllm_ascend.quantization.configs.modelslim_config import AscendModelSlimConfig, get_quant_type_for_layer


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
