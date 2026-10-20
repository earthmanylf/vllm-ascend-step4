# SPDX-License-Identifier: Apache-2.0
"""Ascend MRv2 metadata with persistent Step5 journal request indices."""

from typing import Any

import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.worker.v2.input_batch import AscendInputBatch
from vllm_ascend.worker.v2.model_states.default import AscendModelState


class AscendStep5ModelState(AscendModelState):
    """Keep NPU attention preparation and expose SWF journal coordinates."""

    def __init__(self, vllm_config, model, encoder_cache, device):
        super().__init__(vllm_config, model, encoder_cache, device)
        # FIA can append one padding request to a mixed batch. It never owns
        # a journal row: only the max_num_reqs real state indices are valid.
        self.request_state = torch.full((self.max_num_reqs + 1,), -1, dtype=torch.int64, device=device)

    def prepare_attn(
        self,
        input_batch: AscendInputBatch,
        cudagraph_mode: CUDAGraphMode,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        for_capture: bool = False,
        ubatch_idx: int = 0,
    ) -> dict[str, Any]:
        # Imported here because the backend also participates in model setup.
        from vllm_ascend.models.step5.dsa_attention import AscendStep5DSAMetadataBuilder

        assert ubatch_idx == 0, "DBO is not supported on Ascend"
        num_reqs = input_batch.num_reqs
        if num_reqs > self.max_num_reqs or input_batch.num_reqs_after_padding > self.request_state.numel():
            raise ValueError("Step5 request metadata exceeds the persistent request-state capacity")
        if input_batch.idx_mapping.numel() < num_reqs:
            raise ValueError("Step5 request metadata requires one persistent state index per actual request")
        # The source mapping may be freshly uploaded, while graph replay must
        # always read this same buffer. Clear rows left by a larger batch.
        self.request_state[num_reqs:].fill_(-1)
        self.request_state[:num_reqs].copy_(input_batch.idx_mapping[:num_reqs])
        for groups in attn_groups:
            for group in groups:
                builder = group.get_metadata_builder(ubatch_idx)
                if isinstance(builder, AscendStep5DSAMetadataBuilder):
                    builder.step5_request_state = self.request_state
        return super().prepare_attn(
            input_batch,
            cudagraph_mode,
            block_tables,
            slot_mappings,
            attn_groups,
            kv_cache_config,
            for_capture=for_capture,
            ubatch_idx=ubatch_idx,
        )
