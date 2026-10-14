# SPDX-License-Identifier: Apache-2.0
"""Ascend Step-4 sparse summary side storage.

The Ascend kernels use the staged active-slot layout from the original
Step5 implementation. Its cache data contract and platform-neutral lifecycle
hooks are retained locally because the SWF NVIDIA cache now uses a different
per-page accumulator and request-journal layout. Block reset uses the Ascend
Triton kernel; allocation, profiling and COW preserve the original storage
semantics and share one fixed scratch workspace across DSA layers.
"""

from __future__ import annotations

from typing import Any

import torch
from vllm.logger import init_logger
from vllm.models.step5.nvidia.sparse_summary_cache import (
    Step5SparseSummaryCacheLayer,
)

from vllm_ascend.models.step5._step5_compat import (
    Step5DSAScratchWorkspace,
    Step5SparseSummaryCache,
    Step5SparseSummaryCacheConfig,
    step5_sparse_summary_cache_bytes_per_block,
)
from vllm_ascend.models.step5.summary_kernels import csa_reset_blocks

logger = init_logger(__name__)


class AscendStep5SparseSummaryCache(Step5SparseSummaryCache):
    """The staged Step5 summary cache with NPU Triton block reset.

    The upstream class gates the fused reset kernel behind
    ``device.type == "cuda"`` and falls back to a torch implementation that
    pays several device syncs per call (``.any()``/``unique``/``isin``).
    Block reset fires on every scheduler block allocation, so on Ascend the
    Triton kernel is the only path: it clears the fp8 payload, both slot maps
    and the slot's numerator/denominator/max/staging state in one launch.
    """

    def reset_blocks(self, block_ids: torch.Tensor | list[int], *, clear_mean: bool = True) -> None:
        self.reset_scheduler_blocks(block_ids, clear_mean=clear_mean)

    def reset_scheduler_blocks(self, block_ids: torch.Tensor | list[int], *, clear_mean: bool = True) -> None:
        device = self.mean_cache.device
        if isinstance(block_ids, torch.Tensor):
            ids = block_ids
            if ids.device != device or ids.dtype != torch.long:
                ids = ids.to(device=device, dtype=torch.long)
        else:
            if not block_ids:
                return
            ids = torch.tensor(block_ids, device=device, dtype=torch.long)
        if ids.numel() == 0:
            return
        ratio = max(1, int(self.blocks_per_scheduler_block))
        self._invalidate_mtp_transaction_for_reset(ids, blocks_per_scheduler_block=ratio)
        if ratio != 1:
            offsets = torch.arange(ratio, device=device, dtype=torch.long)
            ids = (ids.view(-1, 1) * ratio + offsets.view(1, -1)).flatten()
        state = (
            getattr(self, "_step5_csa_active_region_ids", None),
            getattr(self, "_step5_csa_active_slot_by_region", None),
            getattr(self, "_step5_csa_numerator_cache", None),
            getattr(self, "_step5_csa_denominator_cache", None),
            getattr(self, "_step5_csa_max_cache", None),
            getattr(self, "_step5_csa_active_token_valid", None),
        )
        if not all(isinstance(value, torch.Tensor) for value in state):
            # Runtime state is initialized at bind; a reset that arrives
            # earlier (e.g. profiling) has no slot state to clear, and the
            # freshly allocated backing is already zero.
            return
        (
            active_region_ids,
            active_slot_by_region,
            active_numerator,
            denominator,
            max_logits,
            staging_valid,
        ) = state
        csa_reset_blocks(
            ids,
            self.mean_cache,
            active_region_ids,
            active_slot_by_region,
            active_numerator,
            denominator,
            max_logits,
            staging_valid,
            num_pages=int(self.num_pages),
            summaries_per_page=int(self.summaries_per_page),
            proxy_dim=int(self.proxy_dim),
            region_block_size=int(self.region_block_size),
            clear_mean=clear_mean,
        )

    def warmup_scheduler_block_reset(self) -> None:
        active_region_ids = getattr(self, "_step5_csa_active_region_ids", None)
        if isinstance(active_region_ids, torch.Tensor) and active_region_ids.numel() > 0:
            warmup_block = torch.zeros((1,), device=self.mean_cache.device, dtype=torch.long)
            self.reset_scheduler_blocks(warmup_block)


class AscendStep5SparseSummaryCacheLayer(Step5SparseSummaryCacheLayer):
    """Adapt SWF registration while retaining Ascend summary lifecycle state."""

    def __init__(
        self,
        *,
        prefix: str,
        target_impl: Any,
        sparse_config: Any = None,
        main_layer_name: str,
        static_forward_context: dict[str, Any],
    ) -> None:
        # swf's Step5SparseSummaryCacheLayer.__init__ takes (prefix, target_impl,
        # main_layer_name, static_forward_context): no sparse_config, and it no
        # longer sets the Ascend-private attributes. Adapt here.
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
        self._backing_bytes_per_page: int | None = None
        self._budget_bytes_per_page: int | None = None
        self._bound_summary_config: Step5SparseSummaryCacheConfig | None = None
        self._budget_only_cache = None

    def get_attn_backend(self):
        from vllm_ascend.models.step5.dsa_attention import AscendStep5DSABackend

        return AscendStep5DSABackend

    def _set_summary_cache_budget(self, block_size: int) -> int:
        if block_size % 8:
            raise ValueError("Step5 KV blocks must align to eight-token regions")
        self._kv_cache_block_size = block_size
        proxy_dim = int(getattr(self._sparse_config, "proxy_dim", 0) or 0)
        num_proxy_kv_heads = int(getattr(self._target_impl, "summary_cache_num_proxy_kv_heads", 1) or 1)
        payload_size_bytes = step5_sparse_summary_cache_bytes_per_block(
            block_size=block_size,
            region_block_size=int(self._target_impl.sparse_region_block_size),
            proxy_dim=proxy_dim,
            num_kv_heads=num_proxy_kv_heads,
        )
        storage_head_size = max(1, (int(payload_size_bytes) + block_size - 1) // block_size)
        backing_page_size_bytes = storage_head_size * block_size
        reverse_map_page_size_bytes = (
            Step5SparseSummaryCacheConfig.max_region_fragments_per_block(
                block_size=block_size,
                region_block_size=int(self._target_impl.sparse_region_block_size),
            )
            * torch.tensor([], dtype=torch.int32).element_size()
        )
        self._backing_bytes_per_page = int(backing_page_size_bytes)
        self._budget_bytes_per_page = int(backing_page_size_bytes + reverse_map_page_size_bytes)
        return self._budget_bytes_per_page

    def bind_kv_cache_side_storage(
        self,
        forward_context: dict[str, Any],
    ) -> None:
        """Upstream bind, adapted to the Ascend split (K, V) owner layout.

        Upstream reads ``num_blocks``/``block_size`` off a combined
        ``(2, num_blocks, block_size, ...)`` owner KV tensor; on Ascend the
        owner layer holds ``(k_cache, v_cache)`` with per-tensor shape
        ``(num_blocks, block_size, num_kv_heads, head_size)``.
        """
        from vllm.distributed.kv_transfer import has_kv_transfer_group

        if has_kv_transfer_group():
            from vllm.config import get_current_vllm_config

            _cfg = get_current_vllm_config()
            _is_decode_bench = _cfg.kv_transfer_config is not None and _cfg.kv_transfer_config.kv_connector in (
                "DecodeBenchConnector",
                "PreFilledDecodeBenchConnector",
            )
            if not _is_decode_bench:
                raise RuntimeError(
                    "Step5 DSA sparse summary side storage is not compatible with "
                    "v1 KV transfer/offload yet. Disable kv_transfer_config for "
                    "Step5, or use a connector implementation that explicitly "
                    "transfers the summary sidecar."
                )
        self._ensure_shared_scratch_workspace(forward_context)

        owner_layer = forward_context.get(self._main_layer_name)
        if owner_layer is None:
            raise RuntimeError(f"Step5 sparse summary cache owner layer is missing: {self._main_layer_name!r}.")
        owner_kv_cache = owner_layer.kv_cache
        if isinstance(owner_kv_cache, (list, tuple)):
            if len(owner_kv_cache) == 0:
                raise RuntimeError("Step5 sparse summary cache owner KV cache is empty.")
            owner_k_cache = owner_kv_cache[0]
        else:
            owner_k_cache = owner_kv_cache
        if not isinstance(owner_k_cache, torch.Tensor) or owner_k_cache.ndim < 2:
            raise RuntimeError(
                f"Step5 sparse summary cache owner KV cache has invalid shape: {type(owner_kv_cache)!r}."
            )
        # The summary sidecar lives in scheduler-block space: one page per
        # scheduler block, so the KV block table and slot_mapping address it
        # directly. Ascend may re-slice the owner KV tensor into smaller
        # kernel blocks (virtual block splitting); derive the scheduler-block
        # count from the total token capacity instead of the tensor shape.
        block_size = self._kv_cache_block_size
        if block_size is None:
            raise RuntimeError("Step5 sparse summary cache was bound before its KV spec was created.")
        if isinstance(owner_kv_cache, (list, tuple)):
            token_capacity = int(owner_k_cache.shape[0]) * int(owner_k_cache.shape[1])
        else:
            token_capacity = int(owner_k_cache.shape[1]) * int(owner_k_cache.shape[2])
        if token_capacity % block_size != 0:
            raise RuntimeError(
                "Step5 sparse summary cache owner KV capacity does not divide "
                f"into scheduler blocks: tokens={token_capacity}, "
                f"block_size={block_size}."
            )
        num_blocks = token_capacity // block_size
        summary_dtype = torch.uint8
        proxy_dim = int(getattr(self._sparse_config, "proxy_dim", 0) or 0)
        num_proxy_kv_heads = int(getattr(self._target_impl, "summary_cache_num_proxy_kv_heads", 1) or 1)
        payload_size_bytes = step5_sparse_summary_cache_bytes_per_block(
            block_size=block_size,
            region_block_size=int(self._target_impl.sparse_region_block_size),
            proxy_dim=proxy_dim,
            num_kv_heads=num_proxy_kv_heads,
            sum_dtype=summary_dtype,
            count_dtype=summary_dtype,
        )
        backing_bytes = ((int(payload_size_bytes) + int(block_size) - 1) // int(block_size)) * int(block_size)
        reverse_map_bytes = (
            Step5SparseSummaryCacheConfig.max_region_fragments_per_block(
                block_size=block_size,
                region_block_size=int(self._target_impl.sparse_region_block_size),
            )
            * torch.tensor([], dtype=torch.int32).element_size()
        )
        self._backing_bytes_per_page = int(backing_bytes)
        self._budget_bytes_per_page = int(backing_bytes + reverse_map_bytes)
        backing = torch.zeros(
            (num_blocks, int(backing_bytes)),
            device=owner_k_cache.device,
            dtype=torch.uint8,
        )
        self._kv_cache = backing
        self.kv_cache = backing
        self._bind_summary_cache(backing, summary_dtype=summary_dtype)
        cache = self._target_impl._summary_cache
        if cache is None:
            raise RuntimeError("Step5 sparse summary cache side storage was not bound.")
        cache.blocks_per_scheduler_block = 1
        cache.warmup_scheduler_block_reset()
        self._budget_only_cache = cache
        if has_kv_transfer_group() and _is_decode_bench:
            import torch as _torch

            _fill_mean = 0.015
            try:
                _kt_cfg = _cfg.kv_transfer_config
                _fill_mean = float(_kt_cfg.get_from_extra_config("fill_mean", 0.015))
            except Exception:
                pass
            with _torch.no_grad():
                fp16_vals = _torch.full(
                    cache.mean_cache.shape,
                    _fill_mean,
                    dtype=_torch.float16,
                    device=cache.mean_cache.device,
                ).to(_torch.float8_e4m3fn)
                cache.mean_cache.view(_torch.uint8).copy_(fp16_vals.view(_torch.uint8))
            logger.info_once(
                "DecodeBenchConnector: filled summary mean_cache with dummy fp8 values (mean=%.6f, shape=%s)",
                _fill_mean,
                tuple(cache.mean_cache.shape),
            )
        logger.info_once(
            "Allocated Step5 sparse summary cache side storage: "
            "pages_per_layer=%d payload_bytes_per_page=%d "
            "allocated_bytes_per_page=%d budget_bytes_per_page=%d "
            "sample_layer_allocated=%.2f GiB device=%s",
            num_blocks,
            int(payload_size_bytes),
            int(backing_bytes),
            int(self._budget_bytes_per_page),
            backing.numel() * backing.element_size() / (1 << 30),
            backing.device,
        )

    def _bind_summary_cache(
        self,
        backing: torch.Tensor,
        *,
        summary_dtype: torch.dtype | None = None,
    ) -> None:
        block_size = self._kv_cache_block_size
        if block_size is None:
            raise RuntimeError("Step5 sparse summary cache was bound before its KV spec was created.")
        if backing.dtype is not torch.uint8:
            raise RuntimeError(f"Step5 sparse summary cache backing tensor must be uint8, got {backing.dtype}.")
        if not backing.is_contiguous():
            raise RuntimeError(
                f"Step5 sparse summary cache backing tensor must be contiguous, got stride={backing.stride()}."
            )
        proxy_dim = int(getattr(self._sparse_config, "proxy_dim", 0) or 0)
        num_proxy_kv_heads = int(getattr(self._target_impl, "summary_cache_num_proxy_kv_heads", 1) or 1)
        config = Step5SparseSummaryCacheConfig(
            num_pages=int(backing.shape[0]),
            page_size=int(block_size),
            region_block_size=int(self._target_impl.sparse_region_block_size),
            num_kv_heads=num_proxy_kv_heads,
            proxy_dim=proxy_dim,
        )
        summary_dtype = summary_dtype or torch.uint8
        expected_bytes = step5_sparse_summary_cache_bytes_per_block(
            block_size=config.page_size,
            region_block_size=config.region_block_size,
            proxy_dim=config.proxy_dim,
            num_kv_heads=config.num_kv_heads,
            sum_dtype=torch.uint8,
            count_dtype=torch.uint8,
        )
        if int(backing.shape[1]) < expected_bytes:
            raise RuntimeError(
                "Step5 sparse summary cache backing tensor shape mismatch: "
                f"expected at least bytes_per_page={expected_bytes}, "
                f"got shape={tuple(backing.shape)}."
            )
        sum_elems = int(config.summaries_per_page) * int(config.num_kv_heads) * int(config.proxy_dim)
        if int(backing.shape[1]) < sum_elems:
            raise RuntimeError(
                f"Step5 sparse mean cache backing is too small: need={sum_elems}, got={int(backing.shape[1])}"
            )
        mean_cache = backing[:, :sum_elems].view(config.sum_shape)
        active_capacity = self._target_impl._csa_active_region_capacity()
        sum_cache = self._target_impl._get_dsa_tensor_buffer_at_least(
            "csa_shared_sum_cache",
            (active_capacity, 1, config.num_kv_heads, config.proxy_dim),
            device=backing.device,
            dtype=torch.float32,
        )
        count_cache = self._target_impl._get_dsa_tensor_buffer_at_least(
            "csa_shared_count_cache",
            (active_capacity, 1, config.num_kv_heads),
            device=backing.device,
            dtype=torch.float32,
        )
        summary_cache = AscendStep5SparseSummaryCache(
            config=config,
            sum_cache=sum_cache,
            count_cache=count_cache,
            mean_cache=mean_cache,
        )
        self._target_impl.bind_summary_cache(summary_cache)
        self._cache = summary_cache
        self._bound_summary_config = config

    @property
    def side_kv_cache(self) -> torch.Tensor:
        # Present the (nb, allocated_bytes) backing as a (1, nb, allocated_bytes)
        # region-major tensor so a generic block-id transfer treats it as a
        # single-region layer (shape[0] == 1), 1:1 aligned with the owner KV
        # blocks.  Storage and the summary kernels are unchanged.
        return self._kv_cache.unsqueeze(0)

    @staticmethod
    def _side_storage_peers(
        forward_context: dict[str, Any],
    ) -> list[AscendStep5SparseSummaryCacheLayer]:
        peers: list[AscendStep5SparseSummaryCacheLayer] = []
        seen_layers: set[int] = set()
        for layer in forward_context.values():
            if not isinstance(layer, AscendStep5SparseSummaryCacheLayer):
                continue
            layer_id = id(layer)
            if layer_id in seen_layers:
                continue
            seen_layers.add(layer_id)
            peers.append(layer)
        if not peers:
            raise RuntimeError("Step5 sparse summary side storage is missing.")
        peers.sort(key=lambda layer: layer.layer_name)
        return peers

    @classmethod
    def _ensure_shared_scratch_workspace(
        cls,
        forward_context: dict[str, Any],
    ) -> tuple[
        AscendStep5SparseSummaryCacheLayer,
        Step5DSAScratchWorkspace,
        list[AscendStep5SparseSummaryCacheLayer],
    ]:
        peers = cls._side_storage_peers(forward_context)
        owner = peers[0]
        workspace = owner._dsa_scratch_workspace
        if workspace is None:
            workspace = Step5DSAScratchWorkspace()
            owner._dsa_scratch_workspace = workspace
        for peer in peers:
            if getattr(peer._target_impl, "_dsa_scratch_workspace", None) is not workspace:
                peer._target_impl.bind_scratch_workspace(workspace)
            peer._dsa_scratch_workspace = workspace
            peer._owns_dsa_scratch_workspace = peer is owner
        return owner, workspace, peers

    def kv_cache_side_storage_memory_profile_key(
        self,
        forward_context: dict[str, Any],
    ) -> object:
        """Return the canonical owner of the model-wide DSA workspace."""
        return self._side_storage_peers(forward_context)[0]

    def prepare_kv_cache_side_storage_memory_profile(
        self,
        forward_context: dict[str, Any],
        device: torch.device,
    ) -> None:
        """Allocate the maximum fixed DSA scratch before KV cache planning."""
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
                    raise RuntimeError(
                        f"Step5 DSA scratch profiling ran before KV cache specs were created for {peer.layer_name!r}."
                    )
                target_impl = peer._target_impl
                config = Step5SparseSummaryCacheConfig(
                    # Fixed scratch sizing uses max_model_len. A single page
                    # is sufficient to supply the remaining layout metadata
                    # without allocating the page-lifetime sidecar itself.
                    num_pages=1,
                    page_size=int(block_size),
                    region_block_size=int(target_impl.sparse_region_block_size),
                    num_kv_heads=int(
                        getattr(
                            target_impl,
                            "summary_cache_num_proxy_kv_heads",
                            1,
                        )
                        or 1
                    ),
                    proxy_dim=int(getattr(peer._sparse_config, "proxy_dim", 0) or 0),
                )
                active_capacity = int(target_impl._csa_active_region_capacity())
                sum_cache = target_impl._get_dsa_tensor_buffer_at_least(
                    "csa_shared_sum_cache",
                    (
                        active_capacity,
                        1,
                        config.num_kv_heads,
                        config.proxy_dim,
                    ),
                    device=device,
                    dtype=torch.float32,
                )
                count_cache = target_impl._get_dsa_tensor_buffer_at_least(
                    "csa_shared_count_cache",
                    (active_capacity, 1, config.num_kv_heads),
                    device=device,
                    dtype=torch.float32,
                )
                profiling_cache = Step5SparseSummaryCache(
                    config=config,
                    sum_cache=sum_cache,
                    count_cache=count_cache,
                    mean_cache=torch.empty(
                        config.sum_shape,
                        device=device,
                        dtype=torch.uint8,
                    ),
                )
                target_impl.bind_summary_cache(
                    profiling_cache,
                    initialize_runtime_state=False,
                )
            workspace.memory_profiled = True
            workspace.allocations_locked = True
        except Exception:
            workspace.allocations_locked = False
            raise

    def _fixed_runtime_state_budget(self, block_size: int) -> int:
        del block_size
        return int(self._target_impl.csa_fixed_runtime_state_size_bytes())

    def zero_kv_cache_side_storage(self, block_ids: torch.Tensor | list[int]) -> None:
        budget_only_cache = self._budget_only_cache
        if budget_only_cache is None:
            return
        if any(
            not isinstance(getattr(budget_only_cache, name, None), torch.Tensor)
            for name in (
                "_step5_csa_active_region_ids",
                "_step5_csa_active_slot_by_region",
                "_step5_csa_numerator_cache",
                "_step5_csa_denominator_cache",
                "_step5_csa_max_cache",
            )
        ):
            return
        budget_only_cache.reset_scheduler_blocks(block_ids)

    def release_kv_cache_side_storage(self, block_ids: torch.Tensor | list[int]) -> None:
        """Release completed request state while retaining prefix-cache payloads."""
        cache = self._budget_only_cache
        if cache is not None:
            cache.reset_scheduler_blocks(block_ids, clear_mean=False)

    def reset_kv_cache_side_storage_runtime_state(self) -> tuple[int, int]:
        """Clear startup dummy state through the generic side-storage hook."""
        cache = self._budget_only_cache
        if cache is None:
            return 0, 0
        cache.reset_runtime_state(reset_producer_scratch=True)

        num_scratch_buffers = 0
        workspace = self._dsa_scratch_workspace
        if getattr(self, "_owns_dsa_scratch_workspace", False) and isinstance(workspace, Step5DSAScratchWorkspace):
            num_scratch_buffers = workspace.reset_runtime_state()
        return 1, num_scratch_buffers

    def copy_kv_cache_side_storage(
        self,
        block_copies: Any,
        num_blocks: int,
    ) -> None:
        """Copy side-cache pages for scheduler copy-on-write operations.

        Prefix-cache and COW paths copy the owner KV pages after allocation.
        The summary sidecar is not part of ``GPUModelRunner.kv_caches`` and
        therefore needs the same block mapping explicitly.
        """
        if self._kv_cache.numel() == 0 or block_copies is None:
            return
        # The worker protocol currently supplies a list, but materializing
        # here keeps this layer safe for connector implementations that pass a
        # one-shot iterator.  The mapping is consumed once for runtime-state
        # cloning and once more for the payload copy below.
        block_copies = list(block_copies)
        if not block_copies:
            return
        from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy
        from vllm.v1.worker.utils import copy_kv_cache_blocks_inplace

        # The scheduler addresses logical blocks, while a side cache may hold
        # multiple physical rows per scheduler block (for example when the
        # owner attention backend expands a scheduler page into several
        # kernel pages).  Reset paths already expand these IDs through
        # ``blocks_per_scheduler_block``; COW must use the same mapping or it
        # would copy only the first physical row and leave the remaining rows
        # stale.
        budget_only_cache = self._budget_only_cache
        blocks_per_scheduler_block = max(
            1,
            int(
                getattr(
                    budget_only_cache,
                    "blocks_per_scheduler_block",
                    1,
                )
            ),
        )
        if blocks_per_scheduler_block > 1:
            expanded_block_copies = [
                KVCacheBlockCopy(
                    src_block_id=(int(block_copy.src_block_id) * blocks_per_scheduler_block + offset),
                    dst_block_id=(int(block_copy.dst_block_id) * blocks_per_scheduler_block + offset),
                )
                for block_copy in block_copies
                for offset in range(blocks_per_scheduler_block)
            ]
            block_copies = expanded_block_copies
            # ``_kv_cache`` is the physical side-cache backing.  Prefer its
            # actual row count over the scheduler count supplied by the
            # generic lifecycle hook once the mapping has been expanded.
            num_blocks = int(self._kv_cache.shape[0])

        # A partial prefix block can own an incomplete CSA region.  Its
        # persistent state lives in the fixed-capacity active-slot arrays, not
        # in ``mean_cache`` alone.  Clone that state before copying the page
        # payload; otherwise a COW destination starts with a valid mean but
        # no numerator/denominator/max (or decode-tail staging), and the next
        # MTP/graph update observes a different history from the source page.
        self._copy_csa_runtime_state(block_copies)

        if budget_only_cache is not None and getattr(budget_only_cache, "_step5_mtp_transaction", None) is not None:
            destinations = sorted(
                {int(copy.dst_block_id) for copy in block_copies if copy.src_block_id != copy.dst_block_id}
            )
            if destinations:
                # A copied destination changes owner even when its physical
                # page number stays the same. Unchanged source pages stay live.
                budget_only_cache._invalidate_mtp_transaction_for_reset(
                    torch.tensor(
                        destinations,
                        device=budget_only_cache.mean_cache.device,
                        dtype=torch.long,
                    ),
                    blocks_per_scheduler_block=1,
                )

        copy_kv_cache_blocks_inplace(
            [self._kv_cache],
            num_blocks,
            block_copies,
        )

    def _copy_csa_runtime_state(self, block_copies: Any) -> None:
        """Clone active CSA state for physical page COW mappings.

        ``active_region_ids`` is authoritative; ``active_slot_by_region`` is
        only a reverse lookup hint and can be stale after slot reuse.  We
        snapshot every live source slot before clearing destinations so
        overlapping copy lists remain deterministic.  This hook runs between
        scheduler zeroing and the model forward, outside CUDA-graph capture.
        """
        cache = self._budget_only_cache
        if cache is None or not block_copies:
            return
        state_names = (
            "_step5_csa_active_region_ids",
            "_step5_csa_active_slot_by_region",
            "_step5_csa_numerator_cache",
            "_step5_csa_denominator_cache",
            "_step5_csa_max_cache",
            "_step5_csa_active_token_k",
            "_step5_csa_active_token_z",
            "_step5_csa_active_token_valid",
        )
        states = tuple(getattr(cache, name, None) for name in state_names)
        if not all(isinstance(state, torch.Tensor) for state in states):
            # The side cache may be bound before CSA runtime state is
            # initialized (e.g. during profiling).  There is no state to copy
            # in that phase; the page payload copy remains valid.
            return
        (
            active_region_ids,
            active_slot_by_region,
            active_numerator,
            active_denominator,
            active_max,
            active_token_k,
            active_token_z,
            active_token_valid,
        ) = states
        assert isinstance(active_region_ids, torch.Tensor)
        assert isinstance(active_slot_by_region, torch.Tensor)
        assert isinstance(active_numerator, torch.Tensor)
        assert isinstance(active_denominator, torch.Tensor)
        assert isinstance(active_max, torch.Tensor)
        assert isinstance(active_token_k, torch.Tensor)
        assert isinstance(active_token_z, torch.Tensor)
        assert isinstance(active_token_valid, torch.Tensor)

        num_pages = int(getattr(cache, "num_pages", 0))
        summaries_per_page = int(getattr(cache, "summaries_per_page", 0))
        active_capacity = int(active_region_ids.numel())
        if num_pages <= 0 or summaries_per_page <= 0 or active_capacity <= 0:
            return

        # Normalize and validate the physical page mappings.  The generic
        # payload copier receives KVCacheBlockCopy objects, but this helper is
        # intentionally independent of the scheduler block geometry.
        physical_pairs: list[tuple[int, int]] = []
        seen_destinations: dict[int, int] = {}
        for block_copy in block_copies:
            src_page = int(block_copy.src_block_id)
            dst_page = int(block_copy.dst_block_id)
            if src_page == dst_page:
                continue
            if not (0 <= src_page < num_pages and 0 <= dst_page < num_pages):
                raise ValueError(
                    "Step5 DSA side-storage COW page is out of range: "
                    f"src={src_page}, dst={dst_page}, num_pages={num_pages}"
                )
            previous_src = seen_destinations.get(dst_page)
            if previous_src is not None:
                if previous_src != src_page:
                    raise ValueError(
                        "Step5 DSA side-storage COW has conflicting sources "
                        f"for destination page {dst_page}: "
                        f"{previous_src} and {src_page}"
                    )
                # Duplicate copies are harmless for payload assignment but
                # would otherwise double-count active-slot allocations.
                continue
            seen_destinations[dst_page] = src_page
            physical_pairs.append((src_page, dst_page))
        if not physical_pairs:
            return

        slot_ids = torch.arange(
            active_capacity,
            device=active_region_ids.device,
            dtype=torch.long,
        )
        snapshots: list[tuple[int, torch.Tensor, tuple[torch.Tensor, ...]]] = []
        destination_regions: list[torch.Tensor] = []
        total_slots_needed = 0

        # Snapshot all source state before any destination is cleared.  The
        # source active map is authoritative; this also repairs around a stale
        # reverse-map hint instead of silently dropping a live region.
        for src_page, dst_page in physical_pairs:
            dst_begin = dst_page * summaries_per_page
            dst_regions_for_page = torch.arange(
                dst_begin,
                dst_begin + summaries_per_page,
                device=active_region_ids.device,
                dtype=torch.long,
            )
            invalid_destination_regions = (dst_regions_for_page < 0) | (
                dst_regions_for_page >= active_slot_by_region.numel()
            )
            if bool(invalid_destination_regions.any()):
                raise ValueError(f"Step5 DSA side-storage COW destination region is out of range: dst_page={dst_page}")
            # Include every destination page, even one with no active source
            # tail.  This clears stale destination ownership before the mean
            # payload is copied.
            destination_regions.append(dst_regions_for_page)

            src_begin = src_page * summaries_per_page
            src_end = src_begin + summaries_per_page
            live_mask = (active_region_ids >= src_begin) & (active_region_ids < src_end)
            src_slots = slot_ids[live_mask]
            if src_slots.numel() == 0:
                continue
            src_regions = active_region_ids[src_slots].clone()
            dst_regions = src_regions + ((dst_page - src_page) * summaries_per_page)
            invalid_regions = (dst_regions < 0) | (dst_regions >= active_slot_by_region.numel())
            if bool(invalid_regions.any()):
                raise ValueError(
                    f"Step5 DSA side-storage COW region is out of range: src_page={src_page}, dst_page={dst_page}"
                )
            snapshots.append(
                (
                    dst_page,
                    dst_regions,
                    (
                        active_numerator[src_slots].clone(),
                        active_denominator[src_slots].clone(),
                        active_max[src_slots].clone(),
                        active_token_k[src_slots].clone(),
                        active_token_z[src_slots].clone(),
                        active_token_valid[src_slots].clone(),
                    ),
                )
            )
            total_slots_needed += int(src_slots.numel())

        # Clear any old destination ownership and staging rows.  The generic
        # zeroer normally did this already, but keeping the hook idempotent
        # makes direct connector/COW callers safe as well.
        destination_regions_flat = torch.cat(destination_regions)
        destination_regions_flat = destination_regions_flat.unique()
        # ``active_region_ids`` is authoritative. The reverse map is only a
        # lookup hint and can be stale after a prior interrupted/failed
        # allocation, so using it to find destination owners can leak a live
        # slot or create duplicate ownership during COW. Find every live
        # destination owner directly and clear all duplicates defensively.
        destination_region_ids = destination_regions_flat.to(
            device=active_region_ids.device,
            dtype=active_region_ids.dtype,
        )
        matched_slots = torch.where(torch.isin(active_region_ids, destination_region_ids))[0].to(torch.long)

        # Check capacity before mutating any destination state.  A destination
        # slot that currently owns one of the copied regions will be released
        # below and is therefore available for a new clone.  If this check
        # fails, leave both source and destination state untouched.
        free_slot_count = int((active_region_ids < 0).sum().item())
        available_slots = free_slot_count + int(matched_slots.numel())
        if available_slots < total_slots_needed:
            raise RuntimeError(
                "Step5 DSA active-slot capacity is exhausted during KV COW: "
                f"needed={total_slots_needed}, available={available_slots}"
            )

        if matched_slots.numel() > 0:
            active_region_ids.index_fill_(0, matched_slots, -1)
            active_numerator.index_fill_(0, matched_slots, 0)
            active_denominator.index_fill_(0, matched_slots, 0)
            active_max.index_fill_(0, matched_slots, float("-inf"))
            active_token_k.index_fill_(0, matched_slots, 0)
            active_token_z.index_fill_(0, matched_slots, 0)
            active_token_valid.index_fill_(0, matched_slots, 0)
        active_slot_by_region.index_fill_(0, destination_regions_flat, -1)

        free_slots = torch.where(active_region_ids < 0)[0]
        if int(free_slots.numel()) < total_slots_needed:
            raise RuntimeError(
                "Step5 DSA active-slot capacity is exhausted during KV COW: "
                f"needed={total_slots_needed}, available={int(free_slots.numel())}"
            )

        cursor = 0
        for _, dst_regions, state_snapshot in snapshots:
            count = int(dst_regions.numel())
            dst_slots = free_slots[cursor : cursor + count]
            cursor += count
            (
                numerator,
                denominator,
                max_logits,
                token_k,
                token_z,
                token_valid,
            ) = state_snapshot
            active_region_ids.index_copy_(0, dst_slots, dst_regions)
            active_slot_by_region.index_copy_(
                0,
                dst_regions.to(dtype=torch.long),
                dst_slots.to(dtype=active_slot_by_region.dtype),
            )
            active_numerator.index_copy_(0, dst_slots, numerator)
            active_denominator.index_copy_(0, dst_slots, denominator)
            active_max.index_copy_(0, dst_slots, max_logits)
            active_token_k.index_copy_(0, dst_slots, token_k)
            active_token_z.index_copy_(0, dst_slots, token_z)
            active_token_valid.index_copy_(0, dst_slots, token_valid)
