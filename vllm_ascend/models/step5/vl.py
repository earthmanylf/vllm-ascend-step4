# SPDX-License-Identifier: Apache-2.0
"""Step5 multimodal frontend using the registered Ascend language model."""

from vllm.models.step5.nvidia.vl import Step5ForConditionalGeneration


class AscendStep5ForConditionalGeneration(Step5ForConditionalGeneration):
    @staticmethod
    def get_model_state_cls():
        # Keep multimodal input preparation on the Ascend v2 attention path.
        from vllm_ascend.worker.v2.model_states.default import AscendModelState

        return AscendModelState
