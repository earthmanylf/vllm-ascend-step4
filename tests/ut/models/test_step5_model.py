# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import torch

from vllm_ascend.models.step5.model import _reshape_step5_fused_expert_weights

NUM_EXPERTS = 352
HIDDEN_SIZE = 4096
MOE_INTERMEDIATE_SIZE = 1536


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
            ("model.layers.0.mlp.gate_proj.weight", torch.zeros(13824, HIDDEN_SIZE)),
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
