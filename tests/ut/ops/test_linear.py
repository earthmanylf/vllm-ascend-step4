import unittest
from unittest import mock
from unittest.mock import MagicMock, patch

import torch

from tests.ut.base import TestBase
from vllm_ascend import ascend_config
from vllm_ascend.distributed import parallel_state
from vllm_ascend.ops.linear import (
    AscendMergedColumnParallelLinear,
    AscendReplicatedLinear,
    AscendRowParallelLinear,
    AscendUnquantizedLinearMethod,
)


class BaseLinearTest(unittest.TestCase):
    def setUp(self):
        self.mock_group = mock.MagicMock()
        self.mock_group.world_size = 2
        self.mock_group.rank_in_group = 0

        parallel_state._MLP_TP = self.mock_group
        parallel_state._OTP = self.mock_group

        self.mock_ascend_config = MagicMock()
        self.mock_ascend_config.finegrained_tp_config.oproj_tensor_parallel_size = 2
        self.mock_ascend_config.finegrained_tp_config.mlp_tensor_parallel_size = 2

        self.patches = [
            patch("vllm_ascend.ascend_config.get_ascend_config", return_value=self.mock_ascend_config),
            patch("vllm_ascend.distributed.parallel_state.get_otp_group", return_value=self.mock_group),
            patch("vllm_ascend.distributed.parallel_state.get_mlp_tp_group", return_value=self.mock_group),
            patch("vllm_ascend.ops.linear_op.get_tp_group", return_value=self.mock_group),
            patch(
                "vllm.distributed.parallel_state.get_tp_group",
                return_value=self.mock_group,
            ),
            patch("vllm_ascend.utils.mlp_tp_enable", return_value=True),
            patch("vllm_ascend.utils.oproj_tp_enable", return_value=True),
            patch("vllm_ascend.ops.linear_op.enable_dsa_cp", return_value=False),
        ]

        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()


class TestAscendUnquantizedLinearMethod(TestBase):
    def setUp(self):
        self.method = AscendUnquantizedLinearMethod()
        self.layer = mock.MagicMock()
        mock_dtype = mock.PropertyMock(return_value=torch.float16)
        type(self.layer.weight.data).dtype = mock_dtype
        mock_is_meta = mock.PropertyMock(return_value=False)
        type(self.layer.weight.data).is_meta = mock_is_meta
        self.layer.precast_fp32_weight = False

    @patch("vllm_ascend.utils.get_ascend_config")
    @mock.patch("torch_npu.npu_format_cast")
    def test_process_weights_after_loading_with_nz0(self, mock_format_cast, mock_get_config):
        mock_config = MagicMock()
        mock_config.weight_nz_mode = 0
        mock_get_config.return_value = mock_config
        self.method.process_weights_after_loading(self.layer)
        mock_format_cast.assert_not_called()

    @patch("vllm_ascend.utils.get_ascend_config")
    @mock.patch("torch_npu.npu_format_cast")
    def test_process_weights_after_loading_with_nz1(self, mock_format_cast, mock_get_config):
        mock_config = MagicMock()
        mock_config.weight_nz_mode = 1
        mock_get_config.return_value = mock_config
        self.method.process_weights_after_loading(self.layer)
        mock_format_cast.assert_not_called()

    @patch("vllm_ascend.utils.get_ascend_config")
    @mock.patch("torch_npu.npu_format_cast")
    def test_process_weights_after_loading_with_nz2(self, mock_format_cast, mock_get_config):
        mock_config = MagicMock()
        mock_config.weight_nz_mode = 2
        mock_get_config.return_value = mock_config
        self.method.process_weights_after_loading(self.layer)
        mock_format_cast.assert_called_once()


class TestAscendRowParallelLinear(BaseLinearTest):
    @patch("vllm_ascend.ops.linear.get_current_vllm_config", return_value=MagicMock())
    @patch(
        "vllm_ascend.ops.linear.AscendUnquantizedLinearMethod.apply",
        new=lambda self, layer, x, bias=None: torch.nn.functional.linear(x, layer.weight, bias),
    )
    def test_mlp_optimize(self, mock_get_current_vllm_config):
        ascend_config._ASCEND_CONFIG = MagicMock()
        ascend_config._ASCEND_CONFIG.scheduler_config.recompute_scheduler_enable = False
        ascend_config._ASCEND_CONFIG.finegrained_tp_config.mlp_tensor_parallel_size = 2
        ascend_config._ASCEND_CONFIG.ascend_scheduler_config.enabled = False

        linear = AscendRowParallelLinear(
            input_size=16,
            output_size=8,
            prefix="down_proj",
        )
        self.assertEqual(linear.custom_op.comm_group, parallel_state._MLP_TP)

        input_tensor = torch.randn(16, 8)
        linear(input_tensor)

    @patch("vllm_ascend.ops.linear.get_current_vllm_config", return_value=MagicMock())
    @patch(
        "vllm_ascend.ops.linear.AscendUnquantizedLinearMethod.apply",
        new=lambda self, layer, x, bias=None: torch.nn.functional.linear(x, layer.weight, bias),
    )
    def test_oproj_tp(self, mock_get_current_vllm_config):
        ascend_config._ASCEND_CONFIG = MagicMock()
        ascend_config._ASCEND_CONFIG.scheduler_config.recompute_scheduler_enable = False
        ascend_config._ASCEND_CONFIG.finegrained_tp_config.oproj_tensor_parallel_size = 2
        ascend_config._ASCEND_CONFIG.ascend_scheduler_config.enabled = False

        linear = AscendRowParallelLinear(
            input_size=16,
            output_size=8,
            prefix="o_proj",
        )
        self.assertEqual(linear.custom_op.comm_group, parallel_state._OTP)

        input_tensor = torch.randn(16, 8)
        linear(input_tensor)


class TestAscendMergedColumnParallelLinear(BaseLinearTest):
    def test_merged_mlp_tp_init(self):
        ascend_config._ASCEND_CONFIG = MagicMock()
        ascend_config._ASCEND_CONFIG.scheduler_config.recompute_scheduler_enable = False
        ascend_config._ASCEND_CONFIG.finegrained_tp_config.mlp_tensor_parallel_size = 2
        ascend_config._ASCEND_CONFIG.ascend_scheduler_config.enabled = False

        linear = AscendMergedColumnParallelLinear(
            input_size=16,
            output_sizes=[8, 8],
            prefix="gate_up_proj",
        )
        self.assertEqual(linear.custom_op.comm_group, parallel_state._MLP_TP)


class TestAscendReplicatedLinear(BaseLinearTest):
    def test_init_disable_tp(self):
        linear = AscendReplicatedLinear(
            input_size=16,
            output_size=8,
        )
        self.assertTrue(isinstance(linear.quant_method, AscendUnquantizedLinearMethod))

    def test_init_without_disable_tp(self):
        linear = AscendReplicatedLinear(
            input_size=16,
            output_size=8,
        )
        self.assertTrue(isinstance(linear.quant_method, AscendUnquantizedLinearMethod))


class TestColumnParallelOpDispatch(unittest.TestCase):
    """Tests for _get_column_parallel_op factory — share_expert, g_proj."""

    def setUp(self):
        self.mock_layer = MagicMock()
        self._patches = [
            patch("vllm_ascend.ops.linear_op.mlp_tp_enable", return_value=False),
            patch("vllm_ascend.ops.linear_op.oproj_tp_enable", return_value=False),
            patch("vllm_ascend.ops.linear_op.enable_dsa_cp", return_value=False),
            patch("vllm_ascend.ops.linear_op.is_moe_layer", return_value=False),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()

    def _get_column_op(self, prefix: str):
        from vllm_ascend.ops.linear_op import _get_column_parallel_op

        return _get_column_parallel_op(prefix, self.mock_layer)

    def test_share_expert_disabled_with_sp_column(self):
        """share_expert / shared_expert prefix → None when SP enabled."""
        self.assertIsNone(self._get_column_op("model.layers.0.mlp.share_expert.gate_up_proj"))
        self.assertIsNone(self._get_column_op("model.layers.0.mlp.shared_expert.gate_up_proj"))

    def test_g_proj_does_not_use_removed_sp_column_path(self):
        """g_proj (Step3p5 attention gate) is included in SP column prefixes."""
        self.assertIsNone(self._get_column_op("model.layers.0.self_attn.g_proj"))

    def test_multimodal_encoder_prefix_skips_sp_column(self):
        """Multimodal encoder variants should not enter the SP column path."""
        self.assertIsNone(self._get_column_op("model.vision_model_proj.indexer_proj"))
        self.assertIsNone(self._get_column_op("model.vision_tower_encoder.qkv_proj"))


class TestRowParallelOpDispatch(unittest.TestCase):
    """Tests for _get_row_parallel_op — mtp_block, share_expert."""

    def setUp(self):
        self.mock_layer = MagicMock()
        self._patches = [
            patch("vllm_ascend.ops.linear_op.mlp_tp_enable", return_value=False),
            patch("vllm_ascend.ops.linear_op.oproj_tp_enable", return_value=False),
            patch("vllm_ascend.ops.linear_op.enable_dsa_cp", return_value=False),
            patch("vllm_ascend.ops.linear_op.is_moe_layer", return_value=False),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()

    def _op(self, prefix: str):
        from vllm_ascend.ops.linear_op import _get_row_parallel_op

        return _get_row_parallel_op(prefix, self.mock_layer)

    def test_share_expert_disabled_with_sp_row(self):
        """share_expert / shared_expert prefix → None when SP enabled."""
        self.assertIsNone(self._op("model.layers.0.mlp.share_expert.down_proj"))
        self.assertIsNone(self._op("model.layers.0.mlp.shared_expert.down_proj"))

    def test_multimodal_encoder_prefix_skips_sp_row(self):
        """Multimodal encoder variants should not enter the SP row path."""
        self.assertIsNone(self._op("model.multi_modal_projector.down_proj"))
        self.assertIsNone(self._op("model.patch_merge_mlp.out_proj"))


class TestGetParallelOpShareExpert(unittest.TestCase):
    """Tests for get_parallel_op — share_expert/shared_expert disables TP."""

    def setUp(self):
        self.mock_layer = MagicMock()
        self.mock_group = MagicMock()
        self.mock_group.rank_in_group = 1
        self.mock_group.world_size = 2
        self._patches = [
            patch("vllm_ascend.ops.linear_op.get_tp_group", return_value=self.mock_group),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()

    def _call(self, prefix: str):
        from vllm_ascend.ops.linear_op import get_parallel_op

        return get_parallel_op(False, prefix, self.mock_layer, False)

    def test_share_expert_disables_tp(self):
        """share_expert / shared_expert / shared_experts → (None, 0, 1)."""
        with patch("vllm_ascend.ops.linear_op.shared_expert_dp_enabled", return_value=True):
            for prefix in (
                "model.layers.0.mlp.share_expert.gate_up_proj",
                "model.layers.0.mlp.shared_expert.gate_up_proj",
                "model.layers.0.mlp.shared_experts.gate_up_proj",
            ):
                custom_op, tp_rank, tp_size = self._call(prefix)
                self.assertIsNone(custom_op)
                self.assertEqual(tp_rank, 0)
                self.assertEqual(tp_size, 1)

    def test_sequence_parallel_does_not_replicate_shared_expert_weights(self):
        """After decoupling, SP alone keeps TP (weights not replicated)."""
        with patch("vllm_ascend.ops.linear_op.shared_expert_dp_enabled", return_value=False):
            custom_op, tp_rank, tp_size = self._call("model.layers.0.mlp.shared_experts.gate_up_proj")

        self.assertIsNone(custom_op)
        self.assertEqual(tp_rank, 1)
        self.assertEqual(tp_size, 2)

    def test_shared_expert_keeps_tp_without_dp_or_sequence_parallel(self):
        with patch("vllm_ascend.ops.linear_op.shared_expert_dp_enabled", return_value=False):
            custom_op, tp_rank, tp_size = self._call("model.layers.0.mlp.shared_experts.gate_up_proj")

        self.assertIsNone(custom_op)
        self.assertEqual(tp_rank, 1)
        self.assertEqual(tp_size, 2)

    def test_shared_expert_ignores_disable_tp_from_model(self):
        """Models pass disable_tp=is_sequence_parallel for shared experts; after
        decoupling, only the shared-expert DP switch replicates weights."""
        from vllm_ascend.ops.linear_op import get_parallel_op

        prefix = "model.layers.0.mlp.shared_experts.gate_up_proj"
        with patch("vllm_ascend.ops.linear_op.shared_expert_dp_enabled", return_value=False):
            custom_op, tp_rank, tp_size = get_parallel_op(True, prefix, self.mock_layer, False)

        self.assertIsNone(custom_op)
        self.assertEqual(tp_rank, 1)
        self.assertEqual(tp_size, 2)

        with patch("vllm_ascend.ops.linear_op.shared_expert_dp_enabled", return_value=True):
            custom_op, tp_rank, tp_size = get_parallel_op(True, prefix, self.mock_layer, False)

        self.assertIsNone(custom_op)
        self.assertEqual(tp_rank, 0)
        self.assertEqual(tp_size, 1)


class TestRowParallelLinearMmrsFusion(unittest.TestCase):
    """Tests for the fused mm + reduce-scatter (npu_mm_reduce_scatter_base)
    path in AscendRowParallelLinear.forward(reduce_scatter_results=True)."""

    def setUp(self):
        self.mock_group = MagicMock()
        self.mock_group.world_size = 2
        self.mock_group.rank_in_group = 0
        backend = MagicMock()
        backend.get_hccl_comm_name.return_value = "hcom-test"
        self.mock_group.device_group._get_backend.return_value = backend

        self._patches = [
            patch("vllm_ascend.ops.linear_op.get_tp_group", return_value=self.mock_group),
            patch("vllm.distributed.parallel_state.get_tp_group", return_value=self.mock_group),
            patch("vllm.distributed.get_tp_group", return_value=self.mock_group),
            # Pin the hardware profile so the tests do not depend on the
            # machine running them.
            patch(
                "vllm_ascend.ops.linear.get_current_hardware_profile",
                return_value=MagicMock(),
            ),
            patch("vllm_ascend.ops.linear_op.mlp_tp_enable", return_value=False),
            patch("vllm_ascend.ops.linear_op.oproj_tp_enable", return_value=False),
            patch("vllm_ascend.ops.linear_op.enable_dsa_cp", return_value=False),
            patch("vllm_ascend.ops.linear_op.is_moe_layer", return_value=False),
            patch("vllm_ascend.ops.linear.get_current_vllm_config", return_value=MagicMock()),
            patch("torch.distributed.get_rank", return_value=0),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()

    def _make_linear(self, **kwargs):
        kwargs.setdefault("input_size", 16)
        kwargs.setdefault("output_size", 8)
        kwargs.setdefault("bias", False)
        kwargs.setdefault("params_dtype", torch.bfloat16)
        kwargs.setdefault("prefix", "o_proj")
        return AscendRowParallelLinear(**kwargs)

    def _rs_forward(self, linear, rows=16):
        input_tensor = torch.randn(rows, linear.input_size_per_partition, dtype=torch.bfloat16)
        return linear(input_tensor, reduce_scatter_results=True, reduce_scatter_dim=0)

    @patch("vllm_ascend.ops.linear.get_forward_context")
    @patch("vllm_ascend.ops.linear.torch_npu.get_npu_format", return_value=2)
    @patch("vllm_ascend.device.device_op.DeviceOperator")
    def test_fused_path_taken_when_eligible(self, mock_device_op, _mock_fmt, mock_get_ctx):
        mock_get_ctx.return_value = MagicMock(mmrs_fusion=True)
        expected = torch.randn(8, 8, dtype=torch.bfloat16)
        mock_device_op.npu_mm_reduce_scatter_base.return_value = expected

        linear = self._make_linear()
        output, bias = self._rs_forward(linear)

        mock_device_op.npu_mm_reduce_scatter_base.assert_called_once()
        args = mock_device_op.npu_mm_reduce_scatter_base.call_args.args
        self.assertEqual(tuple(args[0].shape), (16, 8))  # (m, k) shard input
        self.assertEqual(tuple(args[1].shape), (8, 8))  # ND (k, n) weight copy
        self.assertEqual(args[2], "hcom-test")
        self.assertEqual(args[3], 2)
        self.assertIs(output, expected)
        self.assertIsNone(bias)

    @patch("vllm_ascend.ops.linear.get_forward_context")
    @patch("vllm_ascend.device.device_op.DeviceOperator")
    @patch("vllm.distributed.tensor_model_parallel_reduce_scatter")
    def test_fallback_when_context_switch_off(self, mock_rs, mock_device_op, mock_get_ctx):
        mock_get_ctx.return_value = MagicMock(mmrs_fusion=False)
        partial = torch.randn(16, 8, dtype=torch.bfloat16)
        shard = torch.randn(8, 8, dtype=torch.bfloat16)
        mock_rs.return_value = shard

        linear = self._make_linear()
        with patch.object(linear.quant_method, "apply", return_value=partial):
            output, _ = self._rs_forward(linear)

        mock_device_op.npu_mm_reduce_scatter_base.assert_not_called()
        mock_rs.assert_called_once_with(partial, 0)
        self.assertIs(output, shard)

    @patch("vllm_ascend.ops.linear.get_forward_context")
    @patch("vllm_ascend.device.device_op.DeviceOperator")
    @patch("vllm.distributed.tensor_model_parallel_reduce_scatter")
    def test_fallback_when_rows_not_divisible(self, mock_rs, mock_device_op, mock_get_ctx):
        mock_get_ctx.return_value = MagicMock(mmrs_fusion=True)
        mock_rs.return_value = MagicMock()

        linear = self._make_linear()
        with patch.object(linear.quant_method, "apply", return_value=torch.randn(15, 8)):
            self._rs_forward(linear, rows=15)

        mock_device_op.npu_mm_reduce_scatter_base.assert_not_called()
        mock_rs.assert_called_once()

    @patch("vllm_ascend.ops.linear.get_forward_context")
    @patch("vllm_ascend.ops.linear.torch_npu.get_npu_format", side_effect=RuntimeError("no npu"))
    @patch("vllm_ascend.device.device_op.DeviceOperator")
    @patch("vllm.distributed.tensor_model_parallel_reduce_scatter")
    def test_init_failure_latches_fallback(self, mock_rs, mock_device_op, _mock_fmt, mock_get_ctx):
        mock_get_ctx.return_value = MagicMock(mmrs_fusion=True)
        mock_rs.return_value = MagicMock()

        linear = self._make_linear()
        with patch.object(linear.quant_method, "apply", return_value=torch.randn(16, 8)):
            self._rs_forward(linear)
            self._rs_forward(linear)

        mock_device_op.npu_mm_reduce_scatter_base.assert_not_called()
        self.assertTrue(linear._mmrs_unavailable)
        self.assertEqual(mock_rs.call_count, 2)


if __name__ == "__main__":
    unittest.main()
