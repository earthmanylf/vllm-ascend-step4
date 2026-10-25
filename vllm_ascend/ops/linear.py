# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
"""
To customize linear communication groups or forward of classes in this file,
extend new linear operations in linear_op.py.
The classes in this file should not be modified, including AscendQKVParallelLinear,
AscendMergedColumnParallelLinear, AscendMergedColumnParallelLinear,
AscendRowParallelLinear and AscendColumnParallelLinear.
"""

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from vllm.config import get_current_vllm_config
from vllm.distributed import divide
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (  # noqa
    WEIGHT_LOADER_V2_SUPPORTED,
    ColumnParallelLinear,
    LinearBase,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    QuantizeMethodBase,
    ReplicatedLinear,
    RowParallelLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.utils import set_weight_attrs
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_ascend.device.hardware_profile import (
    HardwareCapability,
    WeightLayoutPolicy,
    get_current_hardware_profile,
)
from vllm_ascend.ops.linear_op import get_parallel_op, get_replicated_op
from vllm_ascend.quantization.tp_weight_switch import TPWeightGatherSpec, TPWeightSwitchMixin
from vllm_ascend.utils import maybe_trans_nz

logger = init_logger(__name__)

MMRS_FUSION_ROW_THRESHOLD = 1000


def unquantized_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.nn.functional.linear(x, weight, bias)


def unquantized_gemm_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


direct_register_custom_op(
    op_name="unquantized_gemm",
    op_func=unquantized_gemm,
    fake_impl=unquantized_gemm_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)


def _should_keep_nd_for_compatibility_weight(weight: torch.Tensor) -> bool:
    return (
        get_current_hardware_profile().weight_layout_policy is WeightLayoutPolicy.FORCE_NZ
        and weight.ndim >= 2
        and (weight.shape[-1] == 1 or weight.shape[-2] == 1)
    )


class AscendUnquantizedLinearMethod(TPWeightSwitchMixin, UnquantizedLinearMethod):
    """Linear method without quantization"""

    tp_weight_gather_specs = (TPWeightGatherSpec("weight", gather_dim=1),)
    tp_weight_output_gather_specs = (TPWeightGatherSpec("weight"),)
    supports_tp_weight_switch = True

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer)
        keep_nd_weight = _should_keep_nd_for_compatibility_weight(layer.weight.data)
        # must use fp32 to avoid accuracy degradation in dsv4.
        if getattr(layer, "precast_fp32_weight", False):
            weight_fp32 = layer.weight.data.to(torch.float32)
            layer.weight_fp32 = weight_fp32 if keep_nd_weight else maybe_trans_nz(weight_fp32)
        if "conv1d" not in layer.prefix:
            # 310P torch_npu rejects FRACTAL_NZ matmul when the weight-side
            # matrix has n=1 or k=1. Keep scalar gates such as Qwen MoE's
            # shared_expert_gate in ND format, leaving non-310P policy intact.
            if not keep_nd_weight:
                layer.weight.data = maybe_trans_nz(layer.weight.data)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return torch.ops.vllm.unquantized_gemm(x, layer.weight, bias)


# TODO(realliujiaxu): Remove this class after linear of vllm supports custom comm group
class AscendLinearBase(LinearBase):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
    ):
        nn.Module.__init__(self)

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.skip_bias_add = skip_bias_add
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype
        self.quant_config = quant_config
        self.prefix = prefix
        if quant_config is None:
            self.quant_method: QuantizeMethodBase | None = AscendUnquantizedLinearMethod()
        else:
            self.quant_method = quant_config.get_quant_method(self, prefix=prefix)
        self.return_bias = return_bias
        self.disable_tp = disable_tp


class AscendQKVParallelLinear(QKVParallelLinear):
    """Linear layers for the attention's QKV transformation.

    Linear layers for the linear transformation of the query, key, and value
    vectors in the attention layer. The weight matrix is concatenated along
    the output dimension. The layer is parallelized along the head dimension.
    When the number of key/value heads is smaller than the number of query
    heads (e.g., multi-query/grouped-query attention), the key/value head may
    be replicated while the query heads are partitioned.
    """

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = True,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
        v_head_size: int | None = None,
    ):
        self.v_head_size = v_head_size if v_head_size is not None else head_size
        self.custom_op, _, tp_size = get_parallel_op(disable_tp, prefix, self, "column")
        # TODO(realliujiaxu): Replace the initialization code below with super().__init__ after
        # linear of vllm supports custom comm group
        self.hidden_size = hidden_size
        self.head_size = head_size
        self.total_num_heads = total_num_heads
        if total_num_kv_heads is None:
            total_num_kv_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads
        # Divide the weight matrix along the last dimension.
        self.num_heads = divide(self.total_num_heads, tp_size)
        if tp_size >= self.total_num_kv_heads:
            self.num_kv_heads = 1
            self.num_kv_head_replicas = divide(tp_size, self.total_num_kv_heads)
        else:
            self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)
            self.num_kv_head_replicas = 1
        input_size = self.hidden_size
        output_size = (self.num_heads + 2 * self.num_kv_heads) * tp_size * self.head_size
        self.output_sizes = [
            self.num_heads * self.head_size * tp_size,  # q_proj
            self.num_kv_heads * self.head_size * tp_size,  # k_proj
            self.num_kv_heads * self.head_size * tp_size,  # v_proj
        ]
        AscendColumnParallelLinear.__init__(
            self,
            input_size=input_size,
            output_size=output_size,
            bias=bias,
            gather_output=False,
            skip_bias_add=skip_bias_add,
            params_dtype=params_dtype,
            quant_config=quant_config,
            prefix=prefix,
            return_bias=return_bias,
            disable_tp=disable_tp,
        )

    def forward(
        self,
        input_,
    ) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
        if self.custom_op is not None:
            return self.custom_op.apply(input_)

        return super().forward(input_)


class AscendMergedColumnParallelLinear(MergedColumnParallelLinear):
    """Packed linear layers with column parallelism.

    Similar to ColumnParallelLinear, but the weight matrix is concatenated
    along the output dimension. When the weight matrix is loaded, the
    different partitions are sharded separately.

    Use the MLP tensor parallelism group in the MLP module,
    and the original TP group in other modules.
    """

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = True,
        gather_output: bool = False,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
    ):
        self.custom_op, self.tp_rank, self.tp_size = get_parallel_op(disable_tp, prefix, self, "column")
        # TODO(realliujiaxu): Replace the initialization code below with super().__init__ after
        # linear of vllm supports custom comm group
        self.output_sizes = output_sizes
        assert all(output_size % self.tp_size == 0 for output_size in output_sizes)
        AscendColumnParallelLinear.__init__(
            self,
            input_size=input_size,
            output_size=sum(output_sizes),
            bias=bias,
            gather_output=gather_output,
            skip_bias_add=skip_bias_add,
            params_dtype=params_dtype,
            quant_config=quant_config,
            prefix=prefix,
            return_bias=return_bias,
            disable_tp=disable_tp,
        )

    def forward(
        self,
        input_,
    ) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
        if self.custom_op is not None:
            return self.custom_op.apply(input_)

        return super().forward(input_)


class AscendRowParallelLinear(RowParallelLinear):
    """Linear layer with row parallelism.
    Use the MLP tensor parallelism group in the MLP module,
    and the original TP group in other modules.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        input_is_parallel: bool = True,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        out_dtype: torch.dtype | None = None,
        reduce_results: bool = True,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
    ):
        self.custom_op, self.tp_rank, self.tp_size = get_parallel_op(disable_tp, prefix, self, "row")
        # TODO(realliujiaxu): Replace the initialization code below with super().__init__ after
        # linear of vllm supports custom comm group
        # Divide the weight matrix along the first dimension.
        self.input_size_per_partition = divide(input_size, self.tp_size)
        self.output_size_per_partition = output_size
        self.output_partition_sizes = [output_size]
        self.out_dtype = out_dtype

        AscendLinearBase.__init__(
            self,
            input_size,
            output_size,
            skip_bias_add,
            params_dtype,
            quant_config,
            prefix,
            return_bias=return_bias,
            disable_tp=disable_tp,
        )

        self.input_is_parallel = input_is_parallel
        self.reduce_results = reduce_results

        assert self.quant_method is not None
        self.quant_method.create_weights(
            layer=self,
            input_size_per_partition=self.input_size_per_partition,
            output_partition_sizes=self.output_partition_sizes,
            input_size=self.input_size,
            output_size=self.output_size,
            params_dtype=self.params_dtype,
            weight_loader=(
                self.weight_loader_v2
                if self.quant_method.__class__.__name__ in WEIGHT_LOADER_V2_SUPPORTED
                else self.weight_loader
            ),
        )
        if not reduce_results and (bias and not skip_bias_add):
            raise ValueError("When not reduce the results, adding bias to the results can lead to incorrect results")

        if bias:
            self.bias = Parameter(torch.empty(self.output_size, dtype=params_dtype))
            set_weight_attrs(
                self.bias,
                {
                    "output_dim": 0,
                    "weight_loader": self.weight_loader,
                },
            )
        else:
            self.register_parameter("bias", None)

        if self.custom_op is not None:
            self.custom_op.update_attrs()

        # Lazy state for the fused mm + reduce-scatter path
        # (npu_mm_reduce_scatter_base), used by the sequence-parallel
        # reduce_scatter_results branch below. _mmrs_weight holds an ND
        # (k, n) view of the weight: the fused kernel's comm_mode="ai_cpu"
        # (A5) rejects NZ, and passing ND keeps both adaptor families on one
        # code path. None until first use; _mmrs_unavailable latches a
        # permanent fallback to the two-step GEMM + reduce-scatter.
        self._mmrs_hcom: str | None = None
        self._mmrs_weight: torch.Tensor | None = None
        self._mmrs_unavailable = False

    def _mmrs_fusion_ready(self) -> bool:
        """Lazily resolve the TP-group hcom name and the ND transposed weight.

        Returns True when the fused kernel can serve this layer; any failure
        latches _mmrs_unavailable so the caller falls back to the two-step
        GEMM + reduce-scatter permanently.
        """
        if self._mmrs_unavailable:
            return False
        if self._mmrs_weight is not None:
            return True
        try:
            from vllm.distributed import get_tp_group

            device_group = get_tp_group().device_group
            rank = torch.distributed.get_rank(group=device_group)
            backend = device_group._get_backend(torch.device("npu"))
            self._mmrs_hcom = backend.get_hccl_comm_name(rank)
            weight = self.weight.data
            self._mmrs_weight = weight.t()
        except Exception as e:  # noqa: BLE001 - degrade to two-step path
            logger.warning_once(
                "mm_reduce_scatter fusion unavailable for %s (%s); falling back to GEMM + reduce_scatter.",
                self.prefix,
                e,
            )
            self._mmrs_unavailable = True
            return False
        return True

    def _use_mmrs_fusion(self, input_parallel: torch.Tensor) -> bool:
        """Fuse only above the row threshold and with a compatible input shape."""
        try:
            # get_forward_context() asserts outside set_forward_context (UTs);
            # no context means the mmrs_fusion switch is unavailable.
            forward_context = get_forward_context()
        except AssertionError:
            return False
        return (
            getattr(forward_context, "mmrs_fusion", False)
            and input_parallel.dim() == 2
            and input_parallel.shape[0] > MMRS_FUSION_ROW_THRESHOLD
            and input_parallel.shape[0] % self.tp_size == 0
            and self._mmrs_fusion_ready()
        )

    def forward(
        self,
        input_,
        reduce_scatter_results: bool = False,
        reduce_scatter_dim: int = 0,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
        if reduce_scatter_results:
            # Sequence-parallel contract: compute the row-parallel partial and
            # reduce-scatter it across TP instead of all-reducing (half the
            # communication volume). Callers pad rows to a TP multiple.
            if self.custom_op is not None:
                raise NotImplementedError("reduce_scatter_results is not supported with custom ops")
            if not self.reduce_results:
                raise ValueError("reduce_scatter_results requires reduce_results=True")
            from vllm.distributed import split_tensor_along_last_dim, tensor_model_parallel_reduce_scatter

            if self.input_is_parallel:
                input_parallel = input_
            else:
                input_parallel = split_tensor_along_last_dim(input_, num_partitions=self.tp_size)[
                    self.tp_rank
                ].contiguous()
            if reduce_scatter_dim == 0 and self._use_mmrs_fusion(input_parallel):
                # Fused mm + reduce-scatter: the kernel overlaps GEMM with the
                # communication internally and returns the (m/tp, n) shard.
                from vllm_ascend.device.device_op import DeviceOperator

                output = DeviceOperator.npu_mm_reduce_scatter_base(
                    input_parallel.contiguous(), self._mmrs_weight, self._mmrs_hcom, self.tp_size
                )
            else:
                bias_ = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias
                output_parallel = self.quant_method.apply(self, input_parallel, bias_)
                if self.tp_size > 1:
                    output = tensor_model_parallel_reduce_scatter(output_parallel, reduce_scatter_dim)
                else:
                    output = output_parallel
            output_bias = self.bias if self.skip_bias_add else None
            if not self.return_bias:
                return output
            return output, output_bias
        if self.custom_op is not None:
            return self.custom_op.apply(input_)

        return super().forward(input_)


class AscendColumnParallelLinear(ColumnParallelLinear):
    """Linear layer with column parallelism.

    Use the MLP tensor parallelism group in the MLP module,
    and the original TP group in other modules.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        gather_output: bool = False,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        quant_config: QuantizationConfig | None = None,
        output_sizes: list[int] | None = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
    ):
        #
        self.custom_op, self.tp_rank, self.tp_size = get_parallel_op(disable_tp, prefix, self, "column")
        # TODO(realliujiaxu): Replace the initialization code below with super().__init__ after
        # linear of vllm supports custom comm group
        self.input_size_per_partition = input_size
        self.output_size_per_partition = divide(output_size, self.tp_size)
        self.output_partition_sizes = [self.output_size_per_partition]
        # If QKV or MergedColumn, use output size of each partition.
        if hasattr(self, "output_sizes"):
            self.output_partition_sizes = [divide(output_size, self.tp_size) for output_size in self.output_sizes]

        AscendLinearBase.__init__(
            self,
            input_size,
            output_size,
            skip_bias_add,
            params_dtype,
            quant_config,
            prefix,
            return_bias=return_bias,
            disable_tp=disable_tp,
        )

        self.gather_output = gather_output

        if output_sizes is None:
            output_sizes = [output_size]

        assert self.quant_method is not None
        self.quant_method.create_weights(
            layer=self,
            input_size_per_partition=self.input_size_per_partition,
            output_partition_sizes=self.output_partition_sizes,
            input_size=self.input_size,
            output_size=self.output_size,
            params_dtype=self.params_dtype,
            weight_loader=(
                self.weight_loader_v2
                if self.quant_method.__class__.__name__ in WEIGHT_LOADER_V2_SUPPORTED
                else self.weight_loader
            ),
        )
        if bias:
            self.bias = Parameter(torch.empty(self.output_size_per_partition, dtype=params_dtype))
            set_weight_attrs(
                self.bias,
                {
                    "output_dim": 0,
                    "weight_loader": self.weight_loader,
                },
            )
        else:
            self.register_parameter("bias", None)

        if self.custom_op is not None:
            self.custom_op.update_attrs()
        self.prefix = prefix
        if "wo_a" in prefix:
            hf_config = get_current_vllm_config().model_config.hf_text_config
            self.n_local_groups = getattr(hf_config, "o_groups", 0) // self.tp_size
            self.o_lora_rank = getattr(hf_config, "o_lora_rank", 0)

    def forward(
        self,
        input_,
    ) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
        if self.custom_op is not None:
            return self.custom_op.apply(input_)

        return super().forward(input_)

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
        supports_dynamic_mx_quant_fusion = get_current_hardware_profile().supports(
            HardwareCapability.DYNAMIC_MX_QUANT_FUSION
        )
        reshape_bf16_wo_a = (
            "wo_a" in self.prefix
            and supports_dynamic_mx_quant_fusion
            and self.quant_config is None
            and loaded_weight.dtype == torch.bfloat16
        )
        if "wo_a" in self.prefix and (not supports_dynamic_mx_quant_fusion or reshape_bf16_wo_a):
            if self.weight.ndim == 2:
                super().weight_loader(param, loaded_weight)
                self.weight.data = (
                    self.weight.data.view(self.n_local_groups, self.o_lora_rank, -1).transpose(2, 1).contiguous()
                )
            else:
                # In RL update flows, wo_a can be loaded again after being
                # transformed into [n_local_groups, hidden_size, o_lora_rank].
                shard_size = self.n_local_groups * self.o_lora_rank
                start_idx = self.tp_rank * shard_size
                if loaded_weight.shape[0] != shard_size:
                    loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
                loaded_weight = (
                    loaded_weight.view(
                        self.n_local_groups,
                        self.o_lora_rank,
                        -1,
                    )
                    .transpose(2, 1)
                    .contiguous()
                )

                if loaded_weight.shape != self.weight.shape:
                    raise ValueError(
                        f"Unexpected wo_a weight shape {tuple(loaded_weight.shape)}, "
                        f"expected {tuple(self.weight.shape)}"
                    )
                self.weight.data.copy_(loaded_weight)
        else:
            super().weight_loader(param, loaded_weight)


class AscendReplicatedLinear(ReplicatedLinear):
    """Ascend Replicated linear layer.

    Args:
        input_size: input dimension of the linear layer.
        output_size: output dimension of the linear layer.
        bias: If true, add bias.
        skip_bias_add: If true, skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        quant_config: Quantization configure.
        prefix: The name of the layer in the state dict, including all parents
                        (e.g. model.layers.0.qkv_proj)
        return_bias: If true, return bias together with outputs in forward pass.
        disable_tp: Take no effect for replicated linear layers.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
    ):
        self.custom_op, self.tp_rank, self.tp_size = get_replicated_op(disable_tp, prefix, self)
        # If MergedReplicatedLinear, use output size of each partition.
        if hasattr(self, "output_sizes"):
            self.output_partition_sizes = self.output_sizes
        else:
            self.output_partition_sizes = [output_size]

        AscendLinearBase.__init__(
            self,
            input_size,
            output_size,
            skip_bias_add,
            params_dtype,
            quant_config,
            prefix=prefix,
            return_bias=return_bias,
            disable_tp=disable_tp,
        )

        # All the linear layer supports quant method.
        assert self.quant_method is not None
        self.quant_method.create_weights(
            self,
            self.input_size,
            [self.output_size],
            self.input_size,
            self.output_size,
            self.params_dtype,
            weight_loader=self.weight_loader,
        )

        if bias:
            self.bias = Parameter(torch.empty(self.output_size, dtype=self.params_dtype))
            set_weight_attrs(
                self.bias,
                {
                    "output_dim": 0,
                    "weight_loader": self.weight_loader,
                },
            )
        else:
            self.register_parameter("bias", None)

        if self.custom_op is not None:
            self.custom_op.update_attrs()

    def forward(
        self,
        input_,
    ) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
        if self.custom_op is not None:
            return self.custom_op.apply(input_)

        return super().forward(input_)
