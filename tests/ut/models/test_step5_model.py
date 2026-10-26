# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import pytest
import torch

from vllm_ascend.models.step5.model import _reshape_step5_fused_expert_weights

NUM_EXPERTS = 3
HIDDEN_SIZE = 5
MOE_INTERMEDIATE_SIZE = 4


class TestReshapeStep5FusedExpertWeights:
    def test_2d_fused_expert_tensors_are_reshaped_to_3d(self):
        # ModelSlim MXFP8 layout: [num_experts * out_features, in_features].
        weights = [
            (
                "model.layers.3.moe.gate_proj.weight",
                torch.zeros(NUM_EXPERTS * MOE_INTERMEDIATE_SIZE, HIDDEN_SIZE),
            ),
            (
                "model.layers.3.moe.up_proj.weight",
                torch.zeros(NUM_EXPERTS * MOE_INTERMEDIATE_SIZE, HIDDEN_SIZE),
            ),
            (
                "model.layers.3.moe.down_proj.weight",
                torch.zeros(NUM_EXPERTS * HIDDEN_SIZE, MOE_INTERMEDIATE_SIZE),
            ),
            (
                "model.layers.3.moe.gate_proj.weight_scale",
                torch.zeros(NUM_EXPERTS * MOE_INTERMEDIATE_SIZE, 128, dtype=torch.uint8),
            ),
        ]

        reshaped = dict(_reshape_step5_fused_expert_weights(iter(weights), NUM_EXPERTS))

        assert tuple(reshaped["model.layers.3.moe.gate_proj.weight"].shape) == (
            NUM_EXPERTS,
            MOE_INTERMEDIATE_SIZE,
            HIDDEN_SIZE,
        )
        assert tuple(reshaped["model.layers.3.moe.up_proj.weight"].shape) == (
            NUM_EXPERTS,
            MOE_INTERMEDIATE_SIZE,
            HIDDEN_SIZE,
        )
        assert tuple(reshaped["model.layers.3.moe.down_proj.weight"].shape) == (
            NUM_EXPERTS,
            HIDDEN_SIZE,
            MOE_INTERMEDIATE_SIZE,
        )
        assert tuple(reshaped["model.layers.3.moe.gate_proj.weight_scale"].shape) == (
            NUM_EXPERTS,
            MOE_INTERMEDIATE_SIZE,
            128,
        )

    def test_reshape_preserves_element_order(self):
        fused = torch.arange(2 * NUM_EXPERTS * 4, dtype=torch.float32).reshape(NUM_EXPERTS * 2, 4)

        (name, reshaped), *_ = _reshape_step5_fused_expert_weights(
            iter([("model.layers.3.moe.gate_proj.weight", fused)]),
            NUM_EXPERTS,
        )

        assert name == "model.layers.3.moe.gate_proj.weight"
        assert torch.equal(reshaped[0], fused[:2])
        assert torch.equal(reshaped[-1], fused[-2:])

    def test_non_expert_tensors_pass_through_untouched(self):
        weights = [
            # Router gate: leading dim is num_experts already.
            ("model.layers.3.moe.gate.weight", torch.zeros(NUM_EXPERTS, HIDDEN_SIZE)),
            # Shared expert / dense MLP projections stay 2D.
            ("model.layers.3.moe.share_expert.gate_proj.weight", torch.zeros(MOE_INTERMEDIATE_SIZE, HIDDEN_SIZE)),
            ("model.layers.3.share_expert.gate_proj.weight", torch.zeros(MOE_INTERMEDIATE_SIZE, HIDDEN_SIZE)),
            ("model.layers.0.mlp.gate_proj.weight", torch.zeros(12, HIDDEN_SIZE)),
            # Attention projections must never be reshaped.
            ("model.layers.3.self_attn.q_proj.weight", torch.zeros(HIDDEN_SIZE, HIDDEN_SIZE)),
            # 1-D and already-3D tensors pass through.
            ("model.layers.3.moe.router_bias", torch.zeros(NUM_EXPERTS)),
            (
                "model.layers.3.moe.gate_proj.weight",
                torch.zeros(NUM_EXPERTS, MOE_INTERMEDIATE_SIZE, HIDDEN_SIZE),
            ),
        ]
        original_shapes = {name: tuple(tensor.shape) for name, tensor in weights}

        reshaped = dict(_reshape_step5_fused_expert_weights(iter(weights), NUM_EXPERTS))

        for name, tensor in reshaped.items():
            assert tuple(tensor.shape) == original_shapes[name], name

    def test_malformed_expert_tensor_is_left_for_upstream_error(self):
        # Leading dim neither equals nor divides num_experts: pass through so
        # the upstream loader raises its own informative error.
        weights = [
            ("model.layers.3.moe.gate_proj.weight", torch.zeros(1000, HIDDEN_SIZE)),
        ]

        reshaped = dict(_reshape_step5_fused_expert_weights(iter(weights), NUM_EXPERTS))

        assert tuple(reshaped["model.layers.3.moe.gate_proj.weight"].shape) == (1000, HIDDEN_SIZE)


@pytest.mark.parametrize("projection", ["gate_proj", "up_proj", "down_proj"])
@pytest.mark.parametrize("suffix", ["weight", "weight_scale", "weight_scale_2", "input_scale"])
def test_step5_fused_expert_reshape(projection, suffix):
    rows, cols = (
        (HIDDEN_SIZE, MOE_INTERMEDIATE_SIZE) if projection == "down_proj" else (MOE_INTERMEDIATE_SIZE, HIDDEN_SIZE)
    )
    expected = torch.arange(NUM_EXPERTS * rows * cols).reshape(NUM_EXPERTS, rows, cols).float()
    fused = expected.reshape(NUM_EXPERTS * rows, cols)
    name = f"language_model.model.layers.3.moe.{projection}.{suffix}"
    [(actual_name, actual)] = _reshape_step5_fused_expert_weights(iter([(name, fused)]), NUM_EXPERTS)
    assert actual_name == name
    assert actual.dtype == fused.dtype
    assert actual.shape == expected.shape
    assert torch.equal(actual, expected)
    # A second load of an already stacked tensor must not reshape it again.
    [(again_name, again)] = _reshape_step5_fused_expert_weights(iter([(name, actual)]), NUM_EXPERTS)
    assert again_name == name
    assert again is actual
