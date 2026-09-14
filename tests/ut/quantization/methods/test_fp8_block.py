#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import types
import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

from tests.ut.base import TestBase
from tests.ut.quantization.conftest_quantization import create_mock_vllm_config
from vllm_ascend.quantization.methods.w8a8.fp8_block import (
    AscendFp8BlockFusedMoEMethod,
    AscendFp8BlockLinearMethod,
    Step4Fp8BlockFusedMoEMethod,
    resolve_block_scales,
)
from vllm_ascend.quantization.quant_type import QuantType

MODULE = "vllm_ascend.quantization.methods.w8a8.fp8_block"

# Elements per shared exponent produced by npu_dynamic_mx_quant.
MX_GROUP_SIZE = 32


def reference_resolve(weight, scale_inv, block_n, block_k, out_dtype):
    """Straightforward per-element expansion used as the numerical oracle."""
    out_features, in_features = weight.shape
    weight_fp32 = weight.to(torch.float32)
    resolved = torch.empty((out_features, in_features), dtype=out_dtype)
    for row in range(out_features):
        for col in range(in_features):
            resolved[row, col] = weight_fp32[row, col] * scale_inv[row // block_n, col // block_k]
    return resolved


def make_block_weight(out_features, in_features, block_n, block_k, seed=0):
    generator = torch.Generator().manual_seed(seed)
    weight = torch.randn(out_features, in_features, generator=generator).to(torch.float8_e4m3fn)
    scale_inv = torch.rand((out_features + block_n - 1) // block_n, (in_features + block_k - 1) // block_k) + 0.5
    return weight, scale_inv


class TestResolveBlockScales(TestBase):
    def test_matches_per_element_expansion(self):
        weight, scale_inv = make_block_weight(16, 12, 4, 3)
        resolved = resolve_block_scales(weight, scale_inv, 4, 3, torch.float32)
        expected = reference_resolve(weight, scale_inv, 4, 3, torch.float32)
        self.assertTrue(torch.equal(resolved, expected))

    def test_handles_blocks_that_do_not_divide_the_shape(self):
        weight, scale_inv = make_block_weight(10, 7, 4, 3, seed=1)
        resolved = resolve_block_scales(weight, scale_inv, 4, 3, torch.float32)
        expected = reference_resolve(weight, scale_inv, 4, 3, torch.float32)
        self.assertEqual(resolved.shape, (10, 7))
        self.assertTrue(torch.equal(resolved, expected))

    def test_row_chunking_does_not_change_the_result(self):
        weight, scale_inv = make_block_weight(10, 7, 2, 3, seed=2)
        expected = reference_resolve(weight, scale_inv, 2, 3, torch.float32)
        with patch(f"{MODULE}._ROWS_PER_DEQUANT_STEP", 4):
            resolved = resolve_block_scales(weight, scale_inv, 2, 3, torch.float32)
        self.assertTrue(torch.equal(resolved, expected))

    def test_scale_is_applied_in_float32(self):
        # A block's shared scale is applied before the result is rounded to the
        # model dtype, so a scale bfloat16 cannot represent is not itself lossy.
        weight = torch.ones(1, 1).to(torch.float8_e4m3fn)
        scale_inv = torch.tensor([[1.001953125]])
        resolved = resolve_block_scales(weight, scale_inv, 128, 128, torch.float32)
        self.assertEqual(resolved.item(), 1.001953125)

    def test_unpaired_scale_shape_is_rejected(self):
        weight = torch.randn(256, 256).to(torch.float8_e4m3fn)
        with self.assertRaisesRegex(ValueError, "unpaired"):
            resolve_block_scales(weight, torch.ones(2, 3), 128, 128, torch.bfloat16)

    def test_non_2d_weight_is_rejected(self):
        weight = torch.randn(2, 256, 256).to(torch.float8_e4m3fn)
        with self.assertRaisesRegex(ValueError, "2D"):
            resolve_block_scales(weight, torch.ones(2, 2, 2), 128, 128, torch.bfloat16)


class TestAscendFp8BlockLinearMethod(TestBase):
    def build_scheme(self, is_950=False, block_size=(128, 128)):
        with (
            patch(f"{MODULE}.get_current_vllm_config", return_value=create_mock_vllm_config()),
            patch(f"{MODULE}.is_950", return_value=is_950),
            patch(f"{MODULE}.AscendW8A8MXFP8DynamicLinearMethod") as mock_mxfp8,
        ):
            mock_mxfp8.return_value.dynamic_mx_quant_scale_alg = 0
            mock_mxfp8.return_value.group_size = MX_GROUP_SIZE
            scheme = AscendFp8BlockLinearMethod(block_size)
        return scheme

    def test_get_weight_is_fp8(self):
        scheme = self.build_scheme()
        weight = scheme.get_weight(512, 256, torch.bfloat16)["weight"]
        self.assertEqual(weight.shape, (256, 512))
        self.assertEqual(weight.dtype, torch.float8_e4m3fn)

    def test_pergroup_param_declares_block_scale_and_loader_hints(self):
        scheme = self.build_scheme(block_size=(128, 64))
        params = scheme.get_pergroup_param(512, 256, torch.bfloat16)
        self.assertEqual(params["weight_scale_inv"].shape, (2, 8))
        self.assertEqual(params["weight_scale_inv"].dtype, torch.float32)
        # Loaders offset shards in weight elements and convert to scale entries,
        # so they need the block size to round that conversion up.
        self.assertEqual(params["_block_quant_scale"], (128, 64))
        # The bit-packing hints must stay absent: they make the loaders truncate.
        self.assertNotIn("_packed_dim", params)
        self.assertNotIn("_packed_factor", params)

    def test_pergroup_param_rounds_up_partial_blocks(self):
        scheme = self.build_scheme()
        params = scheme.get_pergroup_param(300, 200, torch.bfloat16)
        self.assertEqual(params["weight_scale_inv"].shape, (2, 3))

    def _make_layer(self, out_features=8, in_features=64, block=(4, 32)):
        weight, scale_inv = make_block_weight(out_features, in_features, block[0], block[1], seed=3)
        layer = nn.Module()
        layer.prefix = "model.layers.0.mlp.down_proj"
        layer.weight = nn.Parameter(weight, requires_grad=False)
        layer.weight_scale_inv = nn.Parameter(scale_inv, requires_grad=False)
        return layer, weight, scale_inv

    def test_resolves_to_model_dtype_off_950(self):
        scheme = self.build_scheme(is_950=False, block_size=(4, 32))
        layer, weight, scale_inv = self._make_layer()

        with patch(f"{MODULE}.maybe_trans_nz", side_effect=lambda tensor: tensor):
            scheme.process_weights_after_loading(layer)

        self.assertEqual(layer.weight.dtype, torch.bfloat16)
        self.assertFalse(hasattr(layer, "weight_scale_inv"))
        expected = reference_resolve(weight, scale_inv, 4, 32, torch.bfloat16)
        self.assertTrue(torch.equal(layer.weight.data, expected))

    def test_requantizes_to_mxfp8_on_950(self):
        scheme = self.build_scheme(is_950=True, block_size=(4, 32))
        layer, _, _ = self._make_layer()
        num_groups = 64 // MX_GROUP_SIZE
        quantized = torch.zeros(8, 64).to(torch.float8_e4m3fn)
        # npu_dynamic_mx_quant hands back the group axis split in two.
        mx_scale = torch.zeros(8, num_groups // 2, 2, dtype=torch.uint8)

        with patch(f"{MODULE}.torch_npu") as mock_npu:
            mock_npu.npu_dynamic_mx_quant.return_value = (quantized, mx_scale)
            scheme.process_weights_after_loading(layer)

        mock_npu.npu_dynamic_mx_quant.assert_called_once()
        self.assertEqual(layer.weight.dtype, torch.float8_e4m3fn)
        self.assertEqual(layer.weight_scale.dtype, torch.uint8)
        # The schemes consume one entry per group, not the split the operator emits.
        self.assertEqual(layer.weight_scale.shape, (8, num_groups))
        self.assertFalse(hasattr(layer, "weight_scale_inv"))
        scheme.mxfp8_method.process_weights_after_loading.assert_called_once_with(layer)

    def test_kv_b_proj_is_left_dense_for_the_attention_backend(self):
        # SFA splits kv_b_proj into W_UK/W_UV and disposes of the layer, so it
        # never runs a matmul. Requantizing it would only destroy the dense
        # matrix the attention backend has to absorb.
        scheme = self.build_scheme(is_950=True, block_size=(4, 32))
        layer, weight, scale_inv = self._make_layer()
        layer.prefix = "model.layers.0.self_attn.kv_b_proj"

        with (
            patch(f"{MODULE}.maybe_trans_nz", side_effect=lambda tensor: tensor),
            patch(f"{MODULE}.torch_npu") as mock_npu,
        ):
            scheme.process_weights_after_loading(layer)

        mock_npu.npu_dynamic_mx_quant.assert_not_called()
        self.assertEqual(layer.weight.dtype, torch.bfloat16)
        expected = reference_resolve(weight, scale_inv, 4, 32, torch.bfloat16)
        self.assertTrue(torch.equal(layer.weight.data, expected))

    def test_falls_back_when_reduction_dim_is_not_mx_aligned(self):
        scheme = self.build_scheme(is_950=True, block_size=(4, 32))
        layer, weight, scale_inv = self._make_layer(out_features=8, in_features=48, block=(4, 32))

        with patch(f"{MODULE}.maybe_trans_nz", side_effect=lambda tensor: tensor):
            scheme.process_weights_after_loading(layer)

        self.assertIsNone(scheme.mxfp8_method)
        self.assertEqual(layer.weight.dtype, torch.bfloat16)
        expected = reference_resolve(weight, scale_inv, 4, 32, torch.bfloat16)
        self.assertTrue(torch.equal(layer.weight.data, expected))

    def test_mx_fallback_does_not_require_layer_prefix(self):
        # The 950 MX-aligned path never logs this warning; this only covers the
        # fallback log so a missing prefix cannot raise AttributeError.
        scheme = self.build_scheme(is_950=True, block_size=(4, 32))
        layer, _, _ = self._make_layer(out_features=8, in_features=48, block=(4, 32))
        delattr(layer, "prefix")

        with patch(f"{MODULE}.maybe_trans_nz", side_effect=lambda tensor: tensor):
            scheme.process_weights_after_loading(layer)

        self.assertIsNone(scheme.mxfp8_method)
        self.assertEqual(layer.weight.dtype, torch.bfloat16)

    def test_apply_uses_unquantized_gemm_when_resolved_to_bf16(self):
        scheme = self.build_scheme(is_950=False)
        scheme.mxfp8_method = None
        layer = nn.Module()
        layer.weight = nn.Parameter(torch.randn(4, 8), requires_grad=False)
        x = torch.randn(2, 8)

        with patch("torch.ops.vllm.unquantized_gemm", return_value="gemm") as mock_gemm:
            self.assertEqual(scheme.apply(layer, x), "gemm")
        mock_gemm.assert_called_once_with(x, layer.weight, None)

    def test_apply_delegates_to_mxfp8_when_requantized(self):
        scheme = self.build_scheme(is_950=True)
        scheme.mxfp8_method.apply.return_value = "mxfp8"
        layer = nn.Module()
        x = torch.randn(2, 8)

        self.assertEqual(scheme.apply(layer, x, None, 0), "mxfp8")
        scheme.mxfp8_method.apply.assert_called_once_with(layer, x, None, 0)


class TestAscendFp8BlockFusedMoEMethod(TestBase):
    def build_scheme(self, is_950=False, block_size=(4, 32)):
        with (
            patch(f"{MODULE}.get_current_vllm_config", return_value=create_mock_vllm_config()),
            patch(f"{MODULE}.is_950", return_value=is_950),
            patch(f"{MODULE}.AscendW8A8MXFP8DynamicFusedMoEMethod") as mock_mxfp8,
        ):
            mock_mxfp8.return_value.dynamic_mx_quant_scale_alg = 0
            mock_mxfp8.return_value.group_size = MX_GROUP_SIZE
            scheme = AscendFp8BlockFusedMoEMethod(block_size, MagicMock())
        return scheme

    def test_weights_are_fp8_and_scales_are_block_shaped(self):
        scheme = self.build_scheme(block_size=(128, 128))
        weights = scheme.get_weight(4, 256, 512, torch.bfloat16)
        self.assertEqual(weights["w13_weight"].shape, (4, 512, 512))
        self.assertEqual(weights["w2_weight"].shape, (4, 512, 256))
        self.assertEqual(weights["w13_weight"].dtype, torch.float8_e4m3fn)

        scales = scheme.get_dynamic_quant_param(4, 256, 512, torch.bfloat16)
        self.assertEqual(scales["w13_weight_scale_inv"].shape, (4, 4, 4))
        self.assertEqual(scales["w2_weight_scale_inv"].shape, (4, 4, 2))
        self.assertEqual(scales["w13_weight_scale_inv"].dtype, torch.float32)

    def test_group_size_makes_the_loader_shard_the_scales(self):
        # AscendFusedMoEMethod only tags a scale as group-wise, and therefore
        # narrows it per expert shard, when the scheme reports a group size.
        scheme = self.build_scheme(block_size=(128, 64))
        self.assertEqual(scheme.group_size, 64)

    def _make_moe_layer(self, num_experts=2, intermediate=8, hidden=64, block=(4, 32)):
        layer = nn.Module()
        layer.prefix = "model.layers.0.mlp.experts"
        w13, w13_scale = make_block_weight(num_experts * 2 * intermediate, hidden, block[0], block[1], seed=4)
        w2, w2_scale = make_block_weight(num_experts * hidden, intermediate, block[0], block[1], seed=5)
        layer.w13_weight = nn.Parameter(w13.reshape(num_experts, 2 * intermediate, hidden), requires_grad=False)
        layer.w2_weight = nn.Parameter(w2.reshape(num_experts, hidden, intermediate), requires_grad=False)
        layer.w13_weight_scale_inv = nn.Parameter(
            torch.rand(num_experts, 2 * intermediate // block[0], hidden // block[1]) + 0.5, requires_grad=False
        )
        layer.w2_weight_scale_inv = nn.Parameter(
            torch.rand(num_experts, hidden // block[0], intermediate // block[1]) + 0.5, requires_grad=False
        )
        return layer

    def test_resolves_every_expert_off_950(self):
        scheme = self.build_scheme(is_950=False)
        layer = self._make_moe_layer(intermediate=32)
        original_w13 = layer.w13_weight.data.clone()
        original_scale = layer.w13_weight_scale_inv.data.clone()
        scheme._bf16_method = MagicMock()

        scheme.process_weights_after_loading(layer)

        self.assertEqual(layer.w13_weight.dtype, torch.bfloat16)
        self.assertEqual(layer.w2_weight.dtype, torch.bfloat16)
        self.assertFalse(hasattr(layer, "w13_weight_scale_inv"))
        self.assertFalse(hasattr(layer, "w2_weight_scale_inv"))
        scheme._bf16_method.process_weights_after_loading.assert_called_once_with(layer)

        expected = reference_resolve(original_w13[0], original_scale[0], 4, 32, torch.bfloat16)
        self.assertTrue(torch.equal(layer.w13_weight.data[0], expected))

    def test_requantizes_every_expert_on_950(self):
        scheme = self.build_scheme(is_950=True)
        # Both weights need an even group count so the operator's split layout
        # is representable for each of them.
        layer = self._make_moe_layer(intermediate=64)
        num_experts = layer.w13_weight.shape[0]

        def fake_mx_quant(tensor, **kwargs):
            rows, cols = tensor.shape
            num_groups = cols // MX_GROUP_SIZE
            return (
                torch.zeros(rows, cols).to(torch.float8_e4m3fn),
                torch.zeros(rows, num_groups // 2, 2, dtype=torch.uint8),
            )

        with patch(f"{MODULE}.torch_npu") as mock_npu:
            mock_npu.npu_dynamic_mx_quant.side_effect = fake_mx_quant
            scheme.process_weights_after_loading(layer)

        self.assertEqual(mock_npu.npu_dynamic_mx_quant.call_count, 2 * num_experts)
        self.assertEqual(layer.w13_weight.dtype, torch.float8_e4m3fn)
        self.assertEqual(layer.w13_weight_scale.shape, (num_experts, 128, 64 // MX_GROUP_SIZE))
        self.assertEqual(layer.w13_weight_scale.dtype, torch.uint8)
        self.assertFalse(hasattr(layer, "w13_weight_scale_inv"))
        scheme.mxfp8_method.process_weights_after_loading.assert_called_once_with(layer)

    def test_apply_delegates_to_the_active_method(self):
        scheme = self.build_scheme(is_950=False)
        scheme._bf16_method = MagicMock()
        scheme._bf16_method.apply.return_value = "bf16"
        layer = nn.Module()

        result = scheme.apply(layer, torch.randn(2, 8), None, None, None, None)

        self.assertEqual(result, "bf16")
        scheme._bf16_method.apply.assert_called_once()


class TestStep4Fp8BlockFusedMoEMethod(TestBase):
    def build_scheme(self, block_size=(128, 128)):
        return Step4Fp8BlockFusedMoEMethod(block_size, MagicMock())

    def test_non_128_blocks_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "128x128"):
            self.build_scheme(block_size=(64, 128))

    def test_weights_stay_fp8_with_checkpoint_block_scales(self):
        scheme = self.build_scheme()
        weights = scheme.get_weight(4, 256, 512, torch.bfloat16)
        self.assertEqual(weights["w13_weight"].shape, (4, 512, 512))
        self.assertEqual(weights["w13_weight"].dtype, torch.float8_e4m3fn)
        self.assertEqual(weights["w2_weight"].shape, (4, 512, 256))

        scales = scheme.get_dynamic_quant_param(4, 256, 512, torch.bfloat16)
        self.assertEqual(scales["w13_weight_scale_inv"].shape, (4, 4, 4))
        self.assertEqual(scales["w2_weight_scale_inv"].shape, (4, 4, 2))
        self.assertEqual(scales["w13_weight_scale_inv"].dtype, torch.float32)

    def test_group_size_makes_the_loader_shard_the_scales(self):
        scheme = self.build_scheme()
        self.assertEqual(scheme.group_size, 128)

    def test_process_weights_after_loading_is_a_noop_without_aclnn(self):
        scheme = self.build_scheme()
        layer = self._make_step4_layer()
        w13_before = layer.w13_weight.data.clone()
        scale_before = layer.w13_weight_scale_inv.data.clone()

        with patch(f"{MODULE}._aclnn_block_fp8_available", return_value=False):
            scheme.process_weights_after_loading(layer)

        self.assertTrue(torch.equal(layer.w13_weight.data, w13_before))
        self.assertTrue(torch.equal(layer.w13_weight_scale_inv.data, scale_before))
        self.assertEqual(layer.w13_weight.dtype, torch.float8_e4m3fn)

    def test_process_weights_after_loading_transposes_to_kmajor_with_aclnn(self):
        scheme = self.build_scheme()
        layer = self._make_step4_layer(num_experts=2, intermediate=128, hidden=512)
        w13_before = layer.w13_weight.data.clone()
        w13_scale_before = layer.w13_weight_scale_inv.data.clone()

        with patch(f"{MODULE}._aclnn_block_fp8_available", return_value=True):
            scheme.process_weights_after_loading(layer)

        # Pure layout change: the fp8 payload and scales are bitwise preserved
        # under the [g, N, K] -> [g, K, N] (and matching scale) transpose that
        # aclnnGroupedMatmulV5 expects.
        self.assertEqual(tuple(layer.w13_weight.shape), (2, 512, 256))
        self.assertTrue(torch.equal(layer.w13_weight.data, w13_before.transpose(1, 2)))
        self.assertTrue(torch.equal(layer.w13_weight_scale_inv.data, w13_scale_before.transpose(1, 2)))
        self.assertEqual(tuple(layer.w2_weight.shape), (2, 128, 512))
        self.assertEqual(layer.w13_weight.dtype, torch.float8_e4m3fn)

    def _make_step4_layer(self, num_experts=2, intermediate=256, hidden=512):
        layer = nn.Module()
        layer.prefix = "model.layers.0.moe.experts"
        layer.w13_weight = nn.Parameter(
            torch.randn(num_experts, 2 * intermediate, hidden).to(torch.float8_e4m3fn), requires_grad=False
        )
        layer.w2_weight = nn.Parameter(
            torch.randn(num_experts, hidden, intermediate).to(torch.float8_e4m3fn), requires_grad=False
        )
        layer.w13_weight_scale_inv = nn.Parameter(
            torch.rand(num_experts, 2 * intermediate // 128, hidden // 128) + 0.5, requires_grad=False
        )
        layer.w2_weight_scale_inv = nn.Parameter(
            torch.rand(num_experts, hidden // 128, intermediate // 128) + 0.5, requires_grad=False
        )
        layer.activation = "swiglustep"
        layer.ascend_expert_map = None
        layer.global_redundant_expert_num = 0
        layer.ascend_mc2_mask = None
        layer.apply_router_weight_on_input = False
        return layer

    def test_apply_routes_to_the_fp8_block_pipeline(self):
        scheme = self.build_scheme()
        layer = self._make_step4_layer()
        x = torch.randn(3, 512, dtype=torch.bfloat16)
        topk_ids = torch.tensor([[0, 1], [0, 1], [1, 0]], dtype=torch.int32)
        topk_weights = torch.rand(3, 2, dtype=torch.bfloat16)

        with patch("vllm_ascend.ascend_forward_context._EXTRA_CTX") as mock_ctx:
            mock_ctx.moe_comm_method.fused_experts.return_value = "routed"
            result = scheme.apply(layer, x, topk_weights, topk_ids)

        self.assertEqual(result, "routed")
        fused_input = mock_ctx.moe_comm_method.fused_experts.call_args.kwargs["fused_experts_input"]
        self.assertIs(fused_input.weights.w1, layer.w13_weight)
        self.assertIs(fused_input.weights.w2, layer.w2_weight)
        self.assertIs(fused_input.weights.w1_scale, layer.w13_weight_scale_inv)
        self.assertIs(fused_input.weights.w2_scale, layer.w2_weight_scale_inv)
        self.assertEqual(fused_input.quant.quant_type, QuantType.FP8_BLOCK)
        self.assertEqual(fused_input.activation, "swiglustep")

    def test_apply_passes_router_weights_through_in_fp32(self):
        # topk_weights come straight from npu_moe_gating_top_k (fp32 out with
        # fp32 logits in); apply must hand them to the pipeline as fp32, so a
        # router fallback that narrows them to bf16 is only ever a cast away.
        scheme = self.build_scheme()
        layer = self._make_step4_layer()
        x = torch.randn(3, 512, dtype=torch.bfloat16)
        topk_ids = torch.tensor([[0, 1]] * 3, dtype=torch.int32)
        topk_weights = torch.rand(3, 2, dtype=torch.bfloat16)

        with patch("vllm_ascend.ascend_forward_context._EXTRA_CTX") as mock_ctx:
            mock_ctx.moe_comm_method.fused_experts.return_value = "routed"
            scheme.apply(layer, x, topk_weights, topk_ids)

        fused_input = mock_ctx.moe_comm_method.fused_experts.call_args.kwargs["fused_experts_input"]
        self.assertEqual(fused_input.topk_weights.dtype, torch.float32)
        self.assertTrue(torch.equal(fused_input.topk_weights, topk_weights.float()))

        fp32_weights = torch.rand(3, 2, dtype=torch.float32)
        with patch("vllm_ascend.ascend_forward_context._EXTRA_CTX") as mock_ctx:
            mock_ctx.moe_comm_method.fused_experts.return_value = "routed"
            scheme.apply(layer, x, fp32_weights, topk_ids)
        fused_input = mock_ctx.moe_comm_method.fused_experts.call_args.kwargs["fused_experts_input"]
        self.assertIs(fused_input.topk_weights, fp32_weights)

    def test_select_experts_keeps_fp32_weights_for_opted_in_schemes(self):
        # AscendRoutedExperts._select_experts narrows topk_weights to the
        # hidden dtype before the quant method sees them; the step4 scheme
        # opts out via keep_fp32_topk_weights so the gating op's fp32 output
        # reaches the routed combine untouched.
        from vllm_ascend.ops.fused_moe.routed_experts import AscendRoutedExperts

        experts = AscendRoutedExperts.__new__(AscendRoutedExperts)
        experts.log2phy = None
        experts.n_shared_experts = 0
        experts.moe_config = MagicMock()
        experts.moe_config.num_experts = 4
        experts.global_redundant_expert_num = 0
        experts.mix_placement = False
        experts.router = MagicMock()
        fp32_weights = torch.rand(2, 3, dtype=torch.float32)
        experts.router._select_experts.return_value = (fp32_weights, torch.zeros(2, 3, dtype=torch.int32))
        hidden = torch.randn(2, 8, dtype=torch.bfloat16)

        experts.quant_method = MagicMock()
        experts.quant_method.quant_method = self.build_scheme()
        with (
            patch("vllm_ascend.ops.fused_moe.routed_experts.get_moe_num_logical_experts", return_value=4),
            patch("vllm_ascend.ops.fused_moe.routed_experts.get_ascend_config") as mock_cfg,
        ):
            mock_cfg.return_value.enable_force_eplb = False
            weights, _ = experts._select_experts(
                hidden_states=hidden, router_logits=None, enable_force_load_balance=False
            )
        self.assertEqual(weights.dtype, torch.float32)
        self.assertIs(weights, fp32_weights)

        # A plain quant method without the opt-in flag (no auto-attrs, unlike
        # MagicMock) goes through the hidden-dtype cast.
        experts.quant_method = types.SimpleNamespace()
        experts.router._select_experts.return_value = (fp32_weights.clone(), torch.zeros(2, 3, dtype=torch.int32))
        with (
            patch("vllm_ascend.ops.fused_moe.routed_experts.get_moe_num_logical_experts", return_value=4),
            patch("vllm_ascend.ops.fused_moe.routed_experts.get_ascend_config") as mock_cfg,
        ):
            mock_cfg.return_value.enable_force_eplb = False
            weights, _ = experts._select_experts(
                hidden_states=hidden, router_logits=None, enable_force_load_balance=False
            )
        self.assertEqual(weights.dtype, torch.bfloat16)


class TestStep4Fp8BlockConfig(TestBase):
    def _make_config(self, weight_block_size=(128, 128)):
        from vllm_ascend.quantization.configs.fp8_config import Step4Fp8BlockConfig

        return Step4Fp8BlockConfig(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            ignored_layers=[],
            weight_block_size=list(weight_block_size),
        )

    def _make_moe_layer(self):
        from vllm.model_executor.layers.fused_moe import RoutedExperts

        layer = RoutedExperts.__new__(RoutedExperts)
        layer.moe_config = MagicMock()
        return layer

    def test_moe_layers_route_to_the_step4_scheme(self):
        from vllm_ascend.quantization.method_adapters import AscendFusedMoEMethod

        quant_method = self._make_config().get_quant_method(self._make_moe_layer(), "model.layers.0.moe.experts")
        self.assertIsInstance(quant_method, AscendFusedMoEMethod)
        self.assertIsInstance(quant_method.quant_method, Step4Fp8BlockFusedMoEMethod)

    def test_linear_layers_keep_the_inherited_route(self):
        from vllm.model_executor.layers.linear import LinearBase

        linear = LinearBase.__new__(LinearBase)
        with patch(
            "vllm_ascend.quantization.configs.fp8_config.AscendFp8Config.get_quant_method",
            return_value="inherited",
        ):
            self.assertEqual(self._make_config().get_quant_method(linear, "model.layers.0.mlp.gate_proj"), "inherited")


class TestStep4Fp8ExpertLayout(TestBase):
    def _layout(self, quantization_config):
        from vllm_ascend.models.step4.model import _step4_fp8_expert_layout

        config = MagicMock()
        config.quantization_config = quantization_config
        return _step4_fp8_expert_layout(config)

    def test_fp8_checkpoint_reports_bf16_moe_layers(self):
        bf16_layers, block = self._layout(
            {
                "quant_method": "fp8",
                "weight_block_size": [128, 128],
                "modules_to_not_convert": [
                    "model.layers.3.moe.experts.w13_weight",
                    "model.layers.3.moe.experts.w2_weight",
                    "model.layers.4.mlp.gate_proj",
                ],
            }
        )
        self.assertEqual(bf16_layers, {3})
        self.assertEqual(block, 128)

    def test_unquantized_checkpoint_has_no_fp8_layers(self):
        self.assertEqual(self._layout({}), (set(), 0))
        self.assertEqual(self._layout(None), (set(), 0))

    def test_non_fp8_quant_method_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "quant_method=fp8"):
            self._layout({"quant_method": "mxfp8"})

    def test_non_128_blocks_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "128x128"):
            self._layout({"quant_method": "fp8", "weight_block_size": [64, 128]})


if __name__ == "__main__":
    unittest.main()
