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

        # maybe_trans_nz reads ndim/shape to screen out k=1/n=1 weights, so the
        # mock must expose a realistic 2D (non-singleton) shape, not a MagicMock.
        type(self.layer.weight.data).ndim = mock.PropertyMock(return_value=2)
        type(self.layer.weight.data).shape = mock.PropertyMock(return_value=torch.Size([64, 32]))
        self.layer.precast_fp32_weight = False
        self.layer.skip_weight_nz_conversion = False

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

    @patch("vllm_ascend.utils.get_ascend_config")
    @mock.patch("torch_npu.npu_format_cast")
    def test_process_weights_after_loading_skips_nz_for_marked_layer(self, mock_format_cast, mock_get_config):
        mock_config = MagicMock()
        mock_config.weight_nz_mode = 2
        mock_get_config.return_value = mock_config
        self.layer.skip_weight_nz_conversion = True
        self.layer.precast_fp32_weight = True
        # Real tensor so precast can materialize weight_fp32 alongside the NZ skip.
        weight = torch.randn(8, 4, dtype=torch.float16)
        self.layer.weight.data = weight
        self.layer.prefix = "model.layers.0.mlp.gate"

        self.method.process_weights_after_loading(self.layer)

        mock_format_cast.assert_not_called()
        self.assertEqual(self.layer.weight_fp32.dtype, torch.float32)
        torch.testing.assert_close(self.layer.weight_fp32, weight.to(torch.float32))

    @patch("vllm_ascend.utils.get_ascend_config")
    @mock.patch("vllm_ascend.ops.linear.maybe_trans_nz", side_effect=lambda x: x)
    @mock.patch("torch_npu.npu_format_cast")
    def test_process_weights_after_loading_precasts_fp32_weight(
        self, mock_format_cast, mock_maybe_trans_nz, mock_get_config
    ):
        mock_config = MagicMock()
        mock_config.weight_nz_mode = 0
        mock_get_config.return_value = mock_config

        weight = torch.randn(8, 4, dtype=torch.float16)
        layer = mock.MagicMock()
        layer.weight.data = weight
        layer.prefix = "model.layers.0.mlp.gate"
        layer.precast_fp32_weight = True
        layer.skip_weight_nz_conversion = True

        self.method.process_weights_after_loading(layer)

        self.assertEqual(layer.weight_fp32.dtype, torch.float32)
        torch.testing.assert_close(layer.weight_fp32, weight.to(torch.float32))
        mock_format_cast.assert_not_called()


class TestShouldReshapeWoATo3d(unittest.TestCase):
    """Tests for _should_reshape_wo_a_to_3d — DSV4 wo_a 3D layout decision."""

    @staticmethod
    def _profile_supporting(mx_quant_fusion: bool):
        profile = MagicMock()
        profile.supports.return_value = mx_quant_fusion
        return profile

    def _reshape_decision(self, prefix, dtype, mx_quant_fusion):
        with patch(
            "vllm_ascend.ops.linear.get_current_hardware_profile",
            return_value=self._profile_supporting(mx_quant_fusion),
        ):
            from vllm_ascend.ops.linear import _should_reshape_wo_a_to_3d

            return _should_reshape_wo_a_to_3d(prefix, dtype)

    def test_bf16_wo_a_reshapes_on_a5(self):
        """Regression: ModelSlim keeps unquantized bf16 wo_a while the model
        carries a global quant config; it still needs the 3D reshape in
        weight_loader (previously gated on quant_config is None)."""
        self.assertTrue(self._reshape_decision("model.layers.0.self_attn.wo_a", torch.bfloat16, mx_quant_fusion=True))

    def test_bf16_wo_a_reshapes_without_mx_quant_fusion(self):
        self.assertTrue(self._reshape_decision("model.layers.0.self_attn.wo_a", torch.bfloat16, mx_quant_fusion=False))

    def test_quantized_wo_a_not_reshaped_by_weight_loader(self):
        """fp8 wo_a is left to the quantization path (process_weights_after_loading)."""
        self.assertFalse(
            self._reshape_decision("model.layers.0.self_attn.wo_a", torch.float8_e4m3fn, mx_quant_fusion=True)
        )

    def test_non_wo_a_layer_not_reshaped(self):
        self.assertFalse(
            self._reshape_decision("model.layers.0.self_attn.o_proj", torch.bfloat16, mx_quant_fusion=True)
        )


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


class TestAscendRowParallelLinearReduceScatter(BaseLinearTest):
    """Sequence-parallel o_proj contract (audit item fcfbd6c0a8).

    Step5's SP-MoE layers call ``o_proj(x, reduce_scatter_results=True,
    reduce_scatter_dim=0)`` and then hand the result to the MoE with
    ``input_is_sequence_parallel=True``, i.e. they expect a TP *shard* back.
    Swallowing those kwargs silently returns the full sequence instead, which
    makes ``_forward_ffn`` skip its chunking and feeds an unsharded tensor into
    the MoE -- the shape the MoE MC2 dispatch then rejects.
    """

    @patch("vllm_ascend.ops.linear.get_current_vllm_config", return_value=MagicMock())
    @patch(
        "vllm.model_executor.layers.linear.RowParallelLinear.forward",
        new=lambda self, input_: torch.nn.functional.linear(input_, self.weight, self.bias),
    )
    @patch("vllm.distributed.tensor_model_parallel_reduce_scatter")
    def test_reduce_scatter_results_returns_tp_shard(self, mock_rs, mock_get_current_vllm_config):
        ascend_config._ASCEND_CONFIG = MagicMock()
        ascend_config._ASCEND_CONFIG.scheduler_config.recompute_scheduler_enable = False
        ascend_config._ASCEND_CONFIG.finegrained_tp_config.oproj_tensor_parallel_size = 0
        ascend_config._ASCEND_CONFIG.ascend_scheduler_config.enabled = False

        tp_size = 2
        num_tokens = 8
        # prefix matches neither "o_proj" nor "down_proj", so custom_op stays
        # None and the plain row-parallel branch is the one under test.
        linear = AscendRowParallelLinear(
            input_size=16,
            output_size=8,
            prefix="",
        )
        self.assertIsNone(linear.custom_op)
        self.assertEqual(linear.tp_size, tp_size)

        # No forward context in a UT, so the fused mm+reduce-scatter path must
        # stay off and the two-step GEMM + reduce-scatter must be taken.
        self.assertFalse(linear._use_mmrs_fusion(torch.randn(num_tokens, 8)))

        # Stand in for the TP reduce-scatter returning this rank's shard.
        mock_rs.side_effect = lambda out, dim: out[: out.shape[0] // tp_size]

        with patch.object(
            linear.quant_method,
            "apply",
            side_effect=lambda layer, inputs, bias: torch.nn.functional.linear(inputs, layer.weight, bias),
        ):
            out = linear(
                torch.randn(num_tokens, 8),
                reduce_scatter_results=True,
                reduce_scatter_dim=0,
            )
        if linear.return_bias:
            out, output_bias = out
            self.assertIsNone(output_bias)

        self.assertTrue(
            mock_rs.called,
            "reduce_scatter_results=True must reach the reduce-scatter, not be swallowed",
        )
        self.assertEqual(mock_rs.call_args[0][1], 0, "reduce_scatter_dim must be forwarded")
        self.assertEqual(
            out.shape[0],
            num_tokens // tp_size,
            "output must be a TP shard, not the full sequence",
        )

    @patch("vllm_ascend.ops.linear.get_current_vllm_config", return_value=MagicMock())
    @patch(
        "vllm.model_executor.layers.linear.RowParallelLinear.forward",
        new=lambda self, input_: torch.nn.functional.linear(input_, self.weight, self.bias),
    )
    def test_default_path_still_all_reduces(self, mock_get_current_vllm_config):
        ascend_config._ASCEND_CONFIG = MagicMock()
        ascend_config._ASCEND_CONFIG.scheduler_config.recompute_scheduler_enable = False
        ascend_config._ASCEND_CONFIG.finegrained_tp_config.oproj_tensor_parallel_size = 0
        ascend_config._ASCEND_CONFIG.ascend_scheduler_config.enabled = False

        num_tokens = 8
        linear = AscendRowParallelLinear(input_size=16, output_size=8, prefix="")
        self.assertIsNone(linear.custom_op)

        out = linear(torch.randn(num_tokens, 8))

        self.assertEqual(
            out.shape[0],
            num_tokens,
            "the default path must stay full-length so non-SP callers are unaffected",
        )


class TestMMRSWeightFormat(unittest.TestCase):
    def setUp(self):
        self.layer = object.__new__(AscendRowParallelLinear)
        torch.nn.Module.__init__(self.layer)
        self.layer.weight = torch.nn.Parameter(torch.ones(4, 8, dtype=torch.bfloat16))
        self.layer.prefix = "test.mmrs"
        self.layer._mmrs_hcom = None
        self.layer._mmrs_weight = None
        self.layer._mmrs_unavailable = False

    @patch("vllm_ascend.ops.linear.torch.device", return_value=torch.device("cpu"))
    @patch("vllm_ascend.ops.linear.torch.distributed.get_rank", return_value=0)
    @patch("vllm.distributed.get_tp_group")
    @patch("vllm_ascend.ops.linear.torch_npu.get_npu_format", create=True)
    def test_rejects_non_nd_before_hcom_init(self, get_format, get_tp_group, get_rank, device):
        for weight_format in (29, 0):
            with self.subTest(weight_format=weight_format):
                self.layer._mmrs_unavailable = False
                get_format.reset_mock()
                get_format.return_value = weight_format
                self.assertFalse(self.layer._mmrs_fusion_ready())
                get_format.assert_called_once_with(self.layer.weight)
                self.assertTrue(self.layer._mmrs_unavailable)
                self.assertIsNone(self.layer._mmrs_weight)
                self.assertIsNone(self.layer._mmrs_hcom)
                get_tp_group.assert_not_called()
                get_format.reset_mock()
                self.assertFalse(self.layer._mmrs_fusion_ready())
                get_format.assert_not_called()

    @patch("vllm.distributed.get_tp_group")
    @patch(
        "vllm_ascend.ops.linear.torch_npu.get_npu_format",
        side_effect=RuntimeError("format unavailable"),
        create=True,
    )
    def test_format_query_failure_falls_back_before_hcom_init(self, get_format, get_tp_group):
        self.assertFalse(self.layer._mmrs_fusion_ready())
        get_format.assert_called_once_with(self.layer.weight)
        self.assertTrue(self.layer._mmrs_unavailable)
        self.assertIsNone(self.layer._mmrs_weight)
        self.assertIsNone(self.layer._mmrs_hcom)
        get_tp_group.assert_not_called()

    @patch("vllm_ascend.ops.linear.torch.device", return_value=torch.device("cpu"))
    @patch("vllm_ascend.ops.linear.torch.distributed.get_rank", return_value=0)
    @patch("vllm.distributed.get_tp_group")
    @patch("vllm_ascend.ops.linear.torch_npu.get_npu_format", return_value=2, create=True)
    def test_nd_weight_initializes_and_caches_transpose(self, get_format, get_tp_group, get_rank, device):
        backend = get_tp_group.return_value.device_group._get_backend.return_value
        backend.get_hccl_comm_name.return_value = "test_hcom"
        self.assertTrue(self.layer._mmrs_fusion_ready())
        get_format.assert_called_once_with(self.layer.weight)
        backend.get_hccl_comm_name.assert_called_once_with(0)
        self.assertEqual(self.layer._mmrs_hcom, "test_hcom")
        torch.testing.assert_close(self.layer._mmrs_weight, self.layer.weight.data.t())
        self.assertEqual(self.layer._mmrs_weight.data_ptr(), self.layer.weight.data_ptr())
        self.assertFalse(self.layer._mmrs_unavailable)
        self.assertTrue(self.layer._mmrs_fusion_ready())
        get_format.assert_called_once()
        get_tp_group.assert_called_once()


class TestMMRSFusionEligibility(unittest.TestCase):
    @patch("vllm_ascend.ops.linear.get_forward_context")
    def test_raw_mmrs_rejects_quantized_bias_and_dtype_mismatch(self, context):
        context.return_value.mmrs_fusion = True
        layer = object.__new__(AscendRowParallelLinear)
        torch.nn.Module.__init__(layer)
        layer.tp_size = 2
        layer.skip_bias_add = False
        layer.weight = torch.nn.Parameter(torch.ones(4, 8, dtype=torch.bfloat16))
        layer.bias = None
        layer.quant_method = AscendUnquantizedLinearMethod()
        inputs = torch.ones(1024, 8, dtype=torch.bfloat16)
        with patch.object(layer, "_mmrs_fusion_ready", return_value=True) as ready:
            self.assertTrue(layer._use_mmrs_fusion(inputs))
            ready.reset_mock()
            layer.bias = torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
            self.assertFalse(layer._use_mmrs_fusion(inputs))
            layer.skip_bias_add = True
            self.assertTrue(layer._use_mmrs_fusion(inputs))
            ready.reset_mock()
            layer.quant_method = MagicMock()
            self.assertFalse(layer._use_mmrs_fusion(inputs))
            ready.assert_not_called()
            layer.quant_method = AscendUnquantizedLinearMethod()
            self.assertFalse(layer._use_mmrs_fusion(inputs.float()))
            layer.tp_size = 1
            self.assertFalse(layer._use_mmrs_fusion(inputs))


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


if __name__ == "__main__":
    unittest.main()
