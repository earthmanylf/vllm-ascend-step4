# SPDX-License-Identifier: Apache-2.0
"""Fixed SWF page accumulators and request journals on Ascend."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from vllm.logger import init_logger
from vllm.models.step5.nvidia.sparse_summary_cache import Step5SparseSummaryCacheLayer

from vllm_ascend.models.step5._step5_compat import (
    Step5DSAScratchWorkspace,
    Step5SparseSummaryCache,
    Step5SparseSummaryCacheConfig,
)

logger = init_logger(__name__)


class AscendStep5SparseSummaryCache(Step5SparseSummaryCache):
    """The SWF cache allocation/lifecycle with explicit NPU update dispatch.

    NPU tensors have ``is_cuda=False``. Override the update entry points so
    they never enter SWF's CPU reference or import CUDA libdevice kernels.
    """

    def __init__(
        self,
        pages: int,
        block_size: int,
        device: torch.device,
        max_num_seqs: int = 1,
        request_journal_width: int = 0,
        *,
        blocks_per_scheduler_block: int = 1,
    ) -> None:
        self.config = Step5SparseSummaryCacheConfig(pages, block_size, 8, 1, 256)
        super().__init__(pages, block_size, device, max_num_seqs, request_journal_width)
        self.blocks_per_scheduler_block = max(1, int(blocks_per_scheduler_block))
        # Shared, profiled transient workspace; excluded from per-layer state.
        self.prefill_state_scratch: torch.Tensor | None = None

    @property
    def mean_cache(self) -> torch.Tensor:
        return self.mean.view(self.pages, self.block_size // 8, 1, 256)

    @property
    def num_pages(self) -> int:
        return self.pages

    @property
    def page_size(self) -> int:
        return self.block_size

    @property
    def region_block_size(self) -> int:
        return 8

    @property
    def summaries_per_page(self) -> int:
        return self.block_size // 8

    @property
    def num_kv_heads(self) -> int:
        return 1

    @property
    def proxy_dim(self) -> int:
        return 256

    def prefill_metadata(self, slots, device):
        from vllm_ascend.models.step5.summary_kernels import make_prefill_metadata_from_slots

        # This is SWF's host metadata API. The DSA hot path builds metadata
        # directly on device and supplies it to update instead.
        if isinstance(slots, torch.Tensor):
            slots = slots.detach().cpu().numpy()
        return make_prefill_metadata_from_slots(np.asarray(slots), self.block_size, device)

    def update(self, key, z, slots, prefill_metadata=None) -> None:
        if key.device.type == "cpu" and prefill_metadata is None:
            # Keep SWF's independent scalar/Torch reference for CPU tests.
            super().update(key, z, slots.tolist() if isinstance(slots, torch.Tensor) else slots)
            return
        from vllm_ascend.models.step5.summary_kernels import update_prefill

        groups = prefill_metadata if prefill_metadata is not None else self.prefill_metadata(slots, key.device)
        if groups is None:
            return
        update_prefill(
            key,
            z,
            groups,
            self.mean,
            self.numerator,
            self.denominator,
            self.maximum,
            self.last_slot,
            self.block_size,
            state=self._prefill_state(groups),
        )

    def _prefill_state(self, metadata):
        if self.prefill_state_scratch is None:
            return None
        return self.prefill_state_scratch[: metadata.shape[0]]

    def update_decode(self, key, z, slots) -> None:
        if key.device.type == "cpu":
            super().update_decode(key, z, slots)
            return
        from vllm_ascend.models.step5.summary_kernels import update_decode

        update_decode(
            key,
            z,
            slots,
            self._mean,
            self.numerator,
            self.denominator,
            self.maximum,
            self.last_slot,
            self.pages,
            self.block_size,
        )

    def update_request_journal(
        self,
        key,
        z,
        slots,
        query_start_loc,
        seq_lens,
        request_state,
        prefix_metadata=None,
    ) -> None:
        from vllm_ascend.models.step5.summary_kernels import update_prefill, update_request_journal

        args = (
            key,
            z,
            slots,
            query_start_loc,
            seq_lens,
            request_state,
            self.request_journal_values,
            self.request_journal_last,
            self.request_journal_tag,
            self.mean,
            self.numerator,
            self.denominator,
            self.maximum,
            self.last_slot,
            self.pages,
            self.block_size,
            self.request_journal_width,
        )
        if prefix_metadata is None:
            update_request_journal(*args)
            return
        update_request_journal(*args, update=False)
        update_prefill(
            key,
            z,
            prefix_metadata,
            self.mean,
            self.numerator,
            self.denominator,
            self.maximum,
            self.last_slot,
            self.block_size,
            state=self._prefill_state(prefix_metadata),
        )
        update_request_journal(*args, restore=False)

    def reset_scheduler_blocks(self, block_ids, *, clear_mean: bool = True) -> None:
        ids = block_ids.tolist() if isinstance(block_ids, torch.Tensor) else list(block_ids)
        ratio = self.blocks_per_scheduler_block
        if ratio != 1:
            ids = [int(block) * ratio + offset for block in ids for offset in range(ratio)]
        # Reuse SWF tag invalidation, tail reset and optional mean preservation.
        self.reset(ids, clear_mean=clear_mean)

    def reset_blocks(self, block_ids, *, clear_mean: bool = True) -> None:
        self.reset_scheduler_blocks(block_ids, clear_mean=clear_mean)


class AscendStep5SparseSummaryCacheLayer(Step5SparseSummaryCacheLayer):
    """SWF lifecycle and budget with Ascend split-KV owner/scratch binding."""

    def __init__(
        self,
        *,
        prefix: str,
        target_impl: Any,
        sparse_config: Any = None,
        main_layer_name: str,
        static_forward_context: dict[str, Any],
    ) -> None:
        super().__init__(
            prefix=prefix,
            target_impl=target_impl,
            main_layer_name=main_layer_name,
            static_forward_context=static_forward_context,
        )
        self._main_layer_name = main_layer_name
        self._sparse_config = sparse_config
        self._kv_cache = self.kv_cache
        self._kv_cache_block_size: int | None = None
        self._dsa_scratch_workspace = None
        self._owns_dsa_scratch_workspace = False
        self._bound_summary_config = None
        self._budget_only_cache = None

    def get_attn_backend(self):
        from vllm_ascend.models.step5.dsa_attention import AscendStep5DSABackend

        return AscendStep5DSABackend

    def _set_summary_cache_budget(self, block_size: int) -> int:
        Step5SparseSummaryCacheConfig(
            1,
            block_size,
            int(self._target_impl.sparse_region_block_size),
            int(getattr(self._target_impl, "summary_cache_num_proxy_kv_heads", 1)),
            int(getattr(self._sparse_config, "proxy_dim", 256)),
        )
        self._kv_cache_block_size = block_size
        return super()._set_summary_cache_budget(block_size)

    def _fixed_runtime_state_budget(self, block_size: int) -> int:
        return super()._fixed_runtime_state_budget(block_size)

    def bind_kv_cache_side_storage(self, forward_context: dict[str, Any]) -> None:
        from vllm.distributed.kv_transfer import has_kv_transfer_group

        decode_bench = False
        if has_kv_transfer_group():
            from vllm.config import get_current_vllm_config

            cfg = get_current_vllm_config()
            transfer_cfg = cfg.kv_transfer_config
            decode_bench = transfer_cfg is not None and transfer_cfg.kv_connector in (
                "DecodeBenchConnector",
                "PreFilledDecodeBenchConnector",
            )
            if not decode_bench:
                raise RuntimeError(
                    "Step5 DSA sparse summary side storage is not compatible with "
                    "v1 KV transfer/offload yet: the summary sidecar must be transferred."
                )
        self._ensure_shared_scratch_workspace(forward_context)
        owner = forward_context.get(self._main_layer_name)
        if owner is None:
            raise RuntimeError(f"Step5 sparse summary cache owner layer is missing: {self._main_layer_name!r}.")
        owner_kv = owner.kv_cache
        if isinstance(owner_kv, (tuple, list)):
            if not owner_kv:
                raise RuntimeError("Step5 sparse summary cache owner KV cache is empty.")
            key_cache = owner_kv[0]
            if not isinstance(key_cache, torch.Tensor) or key_cache.ndim < 2:
                raise RuntimeError("Step5 sparse summary cache owner KV cache has invalid shape.")
            token_capacity = int(key_cache.shape[0]) * int(key_cache.shape[1])
        else:
            key_cache = owner_kv
            if not isinstance(key_cache, torch.Tensor) or key_cache.ndim < 3:
                raise RuntimeError("Step5 sparse summary cache owner KV cache has invalid shape.")
            token_capacity = int(key_cache.shape[1]) * int(key_cache.shape[2])
        block_size = self._kv_cache_block_size
        if block_size is None:
            raise RuntimeError("Step5 sparse summary cache was bound before its KV spec was created.")
        if token_capacity % block_size:
            raise RuntimeError("Step5 sparse summary cache owner KV capacity does not divide into scheduler blocks.")
        # block_table/slot_mapping use scheduler pages even when owning KV is
        # virtually split into smaller kernel blocks.
        cache = AscendStep5SparseSummaryCache(
            token_capacity // block_size,
            block_size,
            key_cache.device,
            self._target_impl.max_num_seqs,
            self._request_journal_width(),
        )
        self._cache = self._budget_only_cache = cache
        self._bound_summary_config = cache.config
        self.kv_cache = self._kv_cache = cache.mean.view(cache.pages, -1)
        self._target_impl.bind_summary_cache(cache)
        if decode_bench:
            fill_mean = float(transfer_cfg.get_from_extra_config("fill_mean", 0.015))
            cache.mean.copy_(
                torch.full_like(cache.mean, fill_mean, dtype=torch.float32).to(torch.float8_e4m3fn).view(torch.uint8)
            )
        logger.info_once(
            "Allocated SWF Step5 summary cache: pages=%d block_size=%d "
            "page_budget=%d fixed_budget=%d journal_width=%d device=%s",
            cache.pages,
            block_size,
            self._set_summary_cache_budget(block_size),
            self._fixed_runtime_state_budget(block_size),
            self._request_journal_width(),
            key_cache.device,
        )

    @property
    def side_kv_cache(self) -> torch.Tensor:
        return self._kv_cache.unsqueeze(0)

    @staticmethod
    def _side_storage_peers(forward_context):
        peers = {
            id(layer): layer
            for layer in forward_context.values()
            if isinstance(layer, AscendStep5SparseSummaryCacheLayer)
        }
        if not peers:
            raise RuntimeError("Step5 sparse summary side storage is missing.")
        return sorted(peers.values(), key=lambda layer: layer.layer_name)

    @classmethod
    def _ensure_shared_scratch_workspace(cls, forward_context):
        peers = cls._side_storage_peers(forward_context)
        owner = peers[0]
        workspace = owner._dsa_scratch_workspace
        if workspace is None:
            workspace = Step5DSAScratchWorkspace()
        for peer in peers:
            if getattr(peer._target_impl, "_dsa_scratch_workspace", None) is not workspace:
                peer._target_impl.bind_scratch_workspace(workspace)
            peer._dsa_scratch_workspace = workspace
            peer._owns_dsa_scratch_workspace = peer is owner
        return owner, workspace, peers

    def kv_cache_side_storage_memory_profile_key(self, forward_context):
        return self._side_storage_peers(forward_context)[0]

    def prepare_kv_cache_side_storage_memory_profile(self, forward_context, device) -> None:
        """Profile/lock only the shared scratch; persistent cache is budgeted separately."""
        _, workspace, peers = self._ensure_shared_scratch_workspace(forward_context)
        if workspace.memory_profiled:
            if not workspace.allocations_locked:
                raise RuntimeError("Step5 DSA scratch workspace is profiled but not locked.")
            return
        workspace.allocations_locked = False
        try:
            for peer in peers:
                block_size = peer._kv_cache_block_size
                if block_size is None:
                    raise RuntimeError("Step5 DSA scratch profiling ran before KV cache specs were created.")
                cache = AscendStep5SparseSummaryCache(1, block_size, device)
                peer._target_impl.bind_summary_cache(cache, initialize_runtime_state=False)
            workspace.memory_profiled = True
            workspace.allocations_locked = True
        except Exception:
            workspace.allocations_locked = False
            raise

    def zero_kv_cache_side_storage(self, block_ids) -> None:
        if self._cache is not None:
            self._cache.reset_scheduler_blocks(block_ids)

    def release_kv_cache_side_storage(self, block_ids) -> None:
        if self._cache is not None:
            self._cache.reset_scheduler_blocks(block_ids, clear_mean=False)

    def reset_kv_cache_side_storage_runtime_state(self) -> tuple[int, int]:
        if self._cache is None:
            return 0, 0
        count, _ = super().reset_kv_cache_side_storage_runtime_state()
        scratch_count = 0
        if self._owns_dsa_scratch_workspace and self._dsa_scratch_workspace is not None:
            scratch_count = self._dsa_scratch_workspace.reset_runtime_state()
        return count, scratch_count

    def copy_kv_cache_side_storage(self, block_copies, num_blocks) -> None:
        del num_blocks
        if self._cache is None or block_copies is None:
            return
        copies = list(block_copies)
        ratio = self._cache.blocks_per_scheduler_block
        if ratio != 1:
            from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy

            copies = [
                KVCacheBlockCopy(
                    src_block_id=int(c.src_block_id) * ratio + offset, dst_block_id=int(c.dst_block_id) * ratio + offset
                )
                for c in copies
                for offset in range(ratio)
            ]
        self._cache.copy(copies)
