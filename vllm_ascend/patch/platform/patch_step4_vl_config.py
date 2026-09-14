#
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

from typing import Any

from vllm.transformers_utils.config import _CONFIG_REGISTRY
from vllm.transformers_utils.configs.step4 import Step4Config


class Step4VLConfig(Step4Config):
    """Config for Step4 vision-language wrapper checkpoints (``model_type="step4_vl"``).

    The published wrapper config nests the text backbone parameters under
    ``text_config`` while vLLM's Step4 implementation expects the flat
    ``Step4Config`` layout, so the nested mapping is flattened with
    ``text_config`` taking precedence over wrapper-level fields. Wrapper-only
    fields (``vision_config`` etc.) survive as plain attributes and do not
    affect the text inference path. After flattening, the instance reports
    ``model_type == "step4"`` and ``architectures == ["Step4ForCausalLM"]``,
    making it equivalent to the flat Step4 checkpoints.
    """

    model_type = "step4_vl"

    def __init__(self, text_config: dict[str, Any] | None = None, **kwargs: Any) -> None:
        if isinstance(text_config, dict):
            kwargs = {**kwargs, **text_config}
        super().__init__(**kwargs)


# `step4_vl` is not registered in vLLM's `model_type` lookup table, so loading
# a Step4 VL wrapper checkpoint fails to resolve the config. Insert the class
# directly: the registry is a `LazyConfigDict` whose values may be either a
# module attribute name or the class itself.
_CONFIG_REGISTRY["step4_vl"] = Step4VLConfig
