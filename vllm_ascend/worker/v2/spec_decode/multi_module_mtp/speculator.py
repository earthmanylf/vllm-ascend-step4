# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
from typing import Any

import torch
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.dp_utils import DPSyncState
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.spec_decode.multi_module_mtp.speculator import (
    MultiModuleMTPSpeculator,
)

from vllm_ascend.attention.attention_v1 import (
    AscendAttentionBackend,
    AscendAttentionState,
    AscendMetadata,
)
from vllm_ascend.worker.v2.attn_utils import (
    build_attn_metadata_factory,
    build_attn_metadata_wrapper,
)
from vllm_ascend.worker.v2.spec_decode.autoregressive.aclgraph import (
    AutoRegressiveAclGraphManager,
)
from vllm_ascend.worker.v2.spec_decode.autoregressive.speculator import (
    AscendDraftSpeculatorMixin,
    torch_gather_wrapper,
)


class AscendMultiModuleMTPSpeculator(AscendDraftSpeculatorMixin, MultiModuleMTPSpeculator):
    """Run upstream's multi-head MTP algorithm in one Ascend draft graph."""

    draft_attn_state = AscendAttentionState.SpecDecoding

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        parallel_config = vllm_config.parallel_config
        if parallel_config.prefill_context_parallel_size != 1 or parallel_config.decode_context_parallel_size != 1:
            raise NotImplementedError("Ascend multi-module MTP currently requires PCP=1 and DCP=1")
        super().__init__(vllm_config, device)
        num_modules = self.draft_model_config.hf_config.num_nextn_predict_layers
        if self.num_speculative_steps > num_modules:
            raise NotImplementedError(
                "Ascend multi-module MTP requires num_speculative_tokens <= num_nextn_predict_layers"
            )
        self._multi_module_attn_metadata: dict[str, Any] | None = None

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        # Retain upstream's attention-capability and requested-mode selection.
        super().init_cudagraph_manager(cudagraph_mode)
        assert self.cudagraph_manager is not None
        selected_mode = self.cudagraph_manager.cudagraph_mode
        if self.speculative_config.enforce_eager:
            selected_mode = CUDAGraphMode.NONE
        # Compiled/breakable Python-module dispatch needs separate validation.
        # FULL captures all modules, with their distinct spec_step_idx, together.
        selected_mode = CUDAGraphMode.FULL_DECODE_ONLY if selected_mode.has_full_cudagraphs() else CUDAGraphMode.NONE
        self.cudagraph_manager = AutoRegressiveAclGraphManager(
            self.vllm_config, self.device, selected_mode, self.num_speculative_steps + 1
        )
        self.cudagraph_manager.speculator = self
        self.cudagraph_manager.update_stream = self.update_stream

    def capture(self) -> None:
        self.last_token_indices.zero_()
        self.idx_mapping.zero_()
        with build_attn_metadata_wrapper():
            # Upstream captures _generate_drafts, including every module's
            # forward, head-specific sampling, and re-prefill input caching.
            super().capture()

    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        dp_sync_state: DPSyncState | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
        num_speculative_tokens: int | None = None,
    ) -> torch.Tensor:
        self.input_batch = input_batch
        self._multi_module_attn_metadata = None
        try:
            with build_attn_metadata_wrapper(), torch_gather_wrapper():
                return super().propose(
                    input_batch,
                    attn_metadata,
                    slot_mappings,
                    last_hidden_states,
                    aux_hidden_states,
                    num_sampled,
                    num_rejected,
                    last_sampled,
                    next_prefill_tokens,
                    temperature,
                    seeds,
                    dp_sync_state=dp_sync_state,
                    dummy_run=dummy_run,
                    skip_attn_for_dummy_run=skip_attn_for_dummy_run,
                    mm_inputs=mm_inputs,
                    is_profile=is_profile or (dummy_run and skip_attn_for_dummy_run),
                    num_speculative_tokens=num_speculative_tokens,
                )
        finally:
            self._multi_module_attn_metadata = None

    def _prepare_inputs(self, *args, **kwargs) -> None:
        super()._prepare_inputs(*args, **kwargs)
        # Multi-module re-prefill rewinds by max(rejected - 1, 0), unlike AR.
        # Snapshot the exact upstream result once per proposal. FIA's host-side
        # length lists require this synchronization; there is no per-head D2H.
        self.input_buffers.seq_lens_cpu.copy_(self.input_buffers.seq_lens.to(device="cpu"))

    def _build_attn_metadata(
        self,
        num_reqs,
        batch_desc,
        query_start_loc_np,
        seq_lens_cpu_upper_bound,
        step,
        **kwargs,
    ):
        if self.input_batch is None:
            raise RuntimeError("Multi-module MTP input batch is missing")
        if batch_desc.cg_mode == CUDAGraphMode.NONE:
            # A FULL target may append dummy request ends to InputBatch's
            # CPU query array. Eager draft metadata must count only real
            # requests, not that target graph's final padded boundary.
            query_start_loc_np = query_start_loc_np[: num_reqs + 1]
            if (
                len(query_start_loc_np) != num_reqs + 1
                or num_reqs < 1
                or int(query_start_loc_np[0]) != 0
                or any(end < start for start, end in zip(query_start_loc_np, query_start_loc_np[1:]))
                or int(query_start_loc_np[-1]) != self.input_batch.num_tokens
                or not 0 <= self.input_batch.num_tokens <= batch_desc.num_tokens
            ):
                raise RuntimeError("Invalid eager MTP real query prefix")
        with build_attn_metadata_factory(
            self.input_buffers.positions,
            batch_desc.num_tokens,
            torch.from_numpy(self.input_batch.is_prefilling_np),
            seq_lens_cpu=self.input_buffers.seq_lens_cpu,
            attn_state=self.draft_attn_state,
            parallel_config=self.draft_vllm_config.parallel_config,
        ):
            attn_metadata = super()._build_attn_metadata(
                num_reqs,
                batch_desc,
                query_start_loc_np,
                seq_lens_cpu_upper_bound,
                step,
                **kwargs,
            )
        if attn_metadata is None:
            raise RuntimeError("Multi-module MTP attention metadata is missing")
        for metadata in attn_metadata.values():
            if metadata is not None:
                metadata.attn_state = self.draft_attn_state
        # Only the exact ordinary FIA backend consumes this host cumulative
        # list. GQA subclasses such as C8-MXFP/QFA use device length buffers.
        if batch_desc.cg_mode == CUDAGraphMode.FULL and self.attn_backend is AscendAttentionBackend:
            num_reqs_padded = batch_desc.num_reqs
            query_width = batch_desc.uniform_token_count
            if (
                num_reqs_padded is None
                or query_width is None
                or query_width <= 0
                or not 0 < num_reqs <= num_reqs_padded
                or num_reqs_padded * query_width != batch_desc.num_tokens
            ):
                raise RuntimeError("Invalid uniform FIA MTP graph descriptor")
            updates = []
            seen = set()
            for metadata in attn_metadata.values():
                if metadata is None or id(metadata) in seen:
                    continue
                seen.add(id(metadata))
                # FIAParamProvider.resolve reads the direct AscendMetadata;
                # nested decode wrappers are not ordinary FIA metadata.
                if type(metadata) is not AscendMetadata:
                    raise RuntimeError("Unsupported FIA MTP metadata type (including nested decode wrappers)")
                query_ends = metadata.actual_seq_lengths_q
                kv_lens = metadata.seq_lens_list
                if (
                    query_ends is None
                    or len(query_ends) != num_reqs_padded
                    or kv_lens is None
                    or len(kv_lens) != num_reqs_padded
                    or list(query_ends[:num_reqs]) != [(i + 1) * query_width for i in range(num_reqs)]
                    or any(length != 0 for length in kv_lens[num_reqs:])
                ):
                    raise RuntimeError("Invalid FIA MTP real query boundaries or nonzero padded KV lengths")
                real_end = query_ends[num_reqs - 1]
                # Accept original collapsed no-op ends or already repaired
                # shared metadata; reject stale/ambiguous padding boundaries.
                fixed_ends = list(query_ends[:num_reqs]) + [
                    (i + 1) * query_width for i in range(num_reqs, num_reqs_padded)
                ]
                if list(query_ends[num_reqs:]) not in (
                    [real_end] * (num_reqs_padded - num_reqs),
                    fixed_ends[num_reqs:],
                ):
                    raise RuntimeError("Invalid FIA MTP padded query boundaries")
                # Validate every layer before modifying any query boundary.
                updates.append((metadata, fixed_ends))
            for metadata, fixed_ends in updates:
                metadata.actual_seq_lengths_q = fixed_ends
        self._multi_module_attn_metadata = attn_metadata
        return attn_metadata

    def build_fia_params(self, num_reqs_padded, draft_attn_metadata, is_draft_model_prefill):
        """Resolve graph updates for every independent MTP attention layer."""
        return [
            {
                "layer_name": layer_name,
                "actual_seq_lengths": metadata.actual_seq_lengths_q,
                "actual_seq_lengths_kv": metadata.seq_lens_list,
                "block_table": metadata.block_tables,
            }
            for layer_name, metadata in draft_attn_metadata.items()
            if metadata is not None
        ]

    def build_draft_attn_metadatas(self, num_reqs_padded, num_tokens_padded, is_draft_model_prefill):
        # Each independent module has its own provider/layer name. All modules
        # consume the same query positions and KV lengths, as upstream intends.
        assert self._multi_module_attn_metadata is not None, "Multi-module MTP metadata was not prepared for replay"
        return [self._multi_module_attn_metadata]
