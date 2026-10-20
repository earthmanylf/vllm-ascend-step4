# SPDX-License-Identifier: Apache-2.0
"""Step5 multimodal frontend using the registered Ascend language model."""

from vllm.model_executor.model_loader.mtp_validation import (
    disable_mtp_completeness_check,
    is_mtp_completeness_check_enabled,
)
from vllm.models.step5.nvidia.model import (
    _reset_fused_qkv_indexer_load_state,
    _validate_fused_qkv_indexer_weights,
)
from vllm.models.step5.nvidia.vl import Step5ForConditionalGeneration


class AscendStep5ForConditionalGeneration(Step5ForConditionalGeneration):
    def load_weights(self, weights):
        if not is_mtp_completeness_check_enabled():
            return super().load_weights(weights)

        # AutoWeightsLoader streams adjacent namespace groups, so interleaved
        # vision tensors can invoke the language loader more than once. Its
        # fused-shard completeness check belongs to this whole VLM transaction.
        language_model = self.language_model.model
        _reset_fused_qkv_indexer_load_state(language_model)
        try:
            with disable_mtp_completeness_check():
                loaded_params = super().load_weights(weights)
            loaded_params.update(
                f"language_model.model.{name}" for name in _validate_fused_qkv_indexer_weights(language_model)
            )
            return loaded_params
        finally:
            # Failed loads must not donate shard records to a later reload.
            _reset_fused_qkv_indexer_load_state(language_model)

    @staticmethod
    def get_model_state_cls():
        # Keep multimodal input preparation on the Ascend v2 attention path.
        from vllm_ascend.models.step5.model_state import AscendStep5ModelState

        return AscendStep5ModelState
