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

"""Replace Step5RMSNorm on Ascend with NPU fused kernels.

TODO: Once upstream ``Step5RMSNorm`` inherits from ``CustomOp`` and implements
``forward_oot`` for Ascend, this patch can be removed. The current workaround
is needed because ``Step5RMSNorm`` directly inherits ``nn.Module`` and falls
back to a torch-native ``pow + mean + mul + rsqrt + mul`` sequence on NPU.
"""

import torch
from vllm.models.step5.layernorm import Step5RMSNorm

from vllm_ascend.utils import enable_custom_op


def _step5_rms_norm_forward_ascend(
    self,
    x: torch.Tensor,
    residual: torch.Tensor | None = None,
    output: torch.Tensor | None = None,
    fp16_out: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    if fp16_out:
        raise ValueError("Step5RMSNorm does not support fp16_out.")

    import torch_npu

    weight = self.weight
    if self.zero_centered:
        weight = weight + 1.0

    if residual is not None:
        if enable_custom_op():
            x, _, residual = torch.ops._C_ascend.npu_add_rms_norm_bias(
                x, residual, weight, None, self.variance_epsilon
            )
        else:
            x, _, residual = torch_npu.npu_add_rms_norm(
                x, residual, weight, self.variance_epsilon
            )
        return x, residual

    result, _ = torch_npu.npu_rms_norm(x, weight, self.variance_epsilon)
    if output is not None:
        output.copy_(result)
        return output
    return result


Step5RMSNorm.forward = _step5_rms_norm_forward_ascend
