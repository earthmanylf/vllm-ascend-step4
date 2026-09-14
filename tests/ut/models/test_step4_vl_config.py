#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
"""CPU UT: Step4 VL wrapper config (``model_type="step4_vl"``) registration.

The Step-4-new-2609 checkpoints ship a nested vision-language wrapper config
whose ``text_config`` holds the flat Step4 text parameters. These tests guard
the patch that teaches vLLM's ``_CONFIG_REGISTRY`` to resolve ``step4_vl`` by
flattening ``text_config`` into an otherwise plain ``Step4Config``.
"""

import json

import vllm_ascend.patch.platform.patch_step4_vl_config  # noqa: F401
from vllm.transformers_utils.config import _CONFIG_REGISTRY, get_config
from vllm_ascend.patch.platform.patch_step4_vl_config import Step4VLConfig

_TEXT_CONFIG = {
    "architectures": ["Step4ForCausalLM"],
    "model_type": "step4",
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_attention_heads": 4,
    "num_attention_groups": 2,
    "num_hidden_layers": 2,
    "num_nextn_predict_layers": 1,
    "vocab_size": 128,
    "max_position_embeddings": 4096,
    "layer_types": ["sliding_attention", "full_attention", "full_attention"],
}

_WRAPPER_CONFIG = {
    "architectures": ["Step4ForConditionalGeneration"],
    "model_type": "step4_vl",
    "num_hidden_layers": 999,
    "vision_config": {"model_type": "perception_encoder", "layers": 47},
    "text_config": _TEXT_CONFIG,
}


def test_step4_vl_registered_in_vllm_config_registry():
    assert _CONFIG_REGISTRY["step4_vl"] is Step4VLConfig


def test_text_config_is_flattened_into_step4_config():
    config = Step4VLConfig(**_WRAPPER_CONFIG)

    assert config.model_type == "step4"
    assert config.architectures == ["Step4ForCausalLM"]
    assert config.hidden_size == 64
    assert config.num_hidden_layers == 2
    assert config.num_nextn_predict_layers == 1
    assert config.max_position_embeddings == 4096
    # transformers validates len(layer_types) == num_hidden_layers, so only
    # the dense slice is exposed while the MTP-inclusive copy is kept aside.
    assert config.layer_types == ["sliding_attention", "full_attention"]
    assert config.layer_types_with_mtp == [
        "sliding_attention",
        "full_attention",
        "full_attention",
    ]


def test_wrapper_only_fields_do_not_shadow_text_config():
    config = Step4VLConfig(**_WRAPPER_CONFIG)

    # Conflicting wrapper-level fields lose to ``text_config``.
    assert config.num_hidden_layers == 2
    # Wrapper-only fields survive as plain attributes.
    assert config.vision_config == {"model_type": "perception_encoder", "layers": 47}


def test_get_config_resolves_step4_vl_checkpoint(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(_WRAPPER_CONFIG))

    config = get_config(str(tmp_path), trust_remote_code=True)

    assert isinstance(config, Step4VLConfig)
    assert config.model_type == "step4"
    assert config.architectures == ["Step4ForCausalLM"]
