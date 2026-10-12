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
#
"""Ascend Step5 vision-language wrapper.

Upstream vLLM's Step5ForConditionalGeneration (step5_vl.py -> step3p7.py)
implements the complete multimodal pipeline: the PerceptionEncoder vision
tower, the vit_large_projector and the Step3VL processor family. Its
language model is built via init_vllm_registered_model which resolves
text_config.architectures == ["Step5ForCausalLM"] through vLLM's
ModelRegistry — and this plugin already registers the
AscendStep5ForCausalLM override for that architecture. Therefore a thin
subclass is sufficient: the upstream multimodal wrapper runs with the
Ascend-optimized text model as backbone, while the vision tower transparently
picks up the platform op overrides (AscendMMEncoderAttention,
AscendQuickGELU, Ascend parallel linears) registered via
CustomOp.register_oot.

The multimodal processor registration is inherited through the MRO from
Step3VLForConditionalGeneration (_processor_factory is looked up on
the class hierarchy by MultimodalRegistry), so the Step3VL
processor/dummy-input builders serve this class unchanged.
"""

from collections.abc import Iterable

import torch

from vllm.model_executor.models.step5_vl import (
    Step5ForConditionalGeneration as UpstreamStep5ForConditionalGeneration,
)
from vllm.model_executor.models.utils import AutoWeightsLoader


class AscendStep5ForConditionalGeneration(UpstreamStep5ForConditionalGeneration):
    """Upstream Step5 VLM wrapper with the Ascend Step5 text backbone."""

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights with the language model receiving one complete stream.

        Step5ForCausalLM.load_weights resets and validates the fused
        qkv+indexer load state on every invocation, so it must see all of its
        weights (text stack *and* lm_head) in a single call. In VL
        checkpoints the lm_head shard is stored next to the vision
        tensors (after them on disk), so the plain AutoWeightsLoader
        grouping at this level splits the language_model.* stream into
        two batches and the second, lm_head-only batch fails the
        completeness validation after the reset. Buffer the tower and
        projector weights (a few GB) while streaming every
        language_model.* weight through one loader call, then load the
        buffered remainder.
        """
        mapper = self.hf_to_vllm_mapper
        tower_weights: list[tuple[str, torch.Tensor]] = []

        def language_stream(
            source: Iterable[tuple[str, torch.Tensor]],
        ) -> Iterable[tuple[str, torch.Tensor]]:
            for name, tensor in source:
                if mapper._map_name(name).startswith("language_model."):
                    yield name, tensor
                else:
                    tower_weights.append((name, tensor))

        loaded = set()
        loader = AutoWeightsLoader(self)
        loaded.update(loader.load_weights(language_stream(weights), mapper=mapper))
        if tower_weights:
            loaded.update(loader.load_weights(iter(tower_weights), mapper=mapper))
        return loaded
