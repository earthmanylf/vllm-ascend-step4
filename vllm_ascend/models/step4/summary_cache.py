# SPDX-License-Identifier: Apache-2.0
"""Ascend Step-4 sparse summary side storage.

Structure mirrors the upstream CUDA implementation
(``vllm.models.step4.sparse_summary_cache``): a per-layer
``Step4SparseSummaryCacheLayer``-compatible module owns a page-backed fp8
summary sidecar whose lifetime follows the owner KV blocks, plus the
fixed-capacity CSA active-slot runtime state. The cache/layer classes are
subclassed rather than forked; only the device-specific pieces are replaced:

* block reset uses the Ascend Triton kernel (``csa_reset_blocks``) instead of
  the ``device.type == "cuda"`` branch / torch fallback of the upstream class;
* the bound cache object is ``AscendStep4SparseSummaryCache``.

The KV-cache extra-budget accounting, memory-profiling protocol, zeroing
hooks and COW copy hooks are the upstream, platform-agnostic ones.
"""

from __future__ import annotations

from typing import Any

import torch

from vllm.logger import init_logger
from vllm.models.step4.sparse_summary_cache import (
    Step4SparseSummaryCache,
    Step4SparseSummaryCacheConfig,
    Step4SparseSummaryCacheLayer,
    step4_sparse_summary_cache_bytes_per_block,
)

from vllm_ascend.models.step4.summary_kernels import csa_reset_blocks

logger = init_logger(__name__)


class AscendStep4SparseSummaryCache(Step4SparseSummaryCache):
    """Step4SparseSummaryCache with the NPU Triton block-reset path.

    The upstream class gates the fused reset kernel behind
    ``device.type == "cuda"`` and falls back to a torch implementation that
    pays several device syncs per call (``.any()``/``unique``/``isin``).
    Block reset fires on every scheduler block allocation, so on Ascend the
    Triton kernel is the only path: it clears the fp8 payload, both slot maps
    and the slot's numerator/denominator/max/staging state in one launch.
    """

    def reset_blocks(
        self, block_ids: torch.Tensor | list[int], *, clear_mean: bool = True
    ) -> None:
        self.reset_scheduler_blocks(block_ids, clear_mean=clear_mean)

    def reset_scheduler_blocks(
        self, block_ids: torch.Tensor | list[int], *, clear_mean: bool = True
    ) -> None:
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
        if ratio != 1:
            offsets = torch.arange(ratio, device=device, dtype=torch.long)
            ids = (ids.view(-1, 1) * ratio + offsets.view(1, -1)).flatten()
        state = (
            getattr(self, "_step4_csa_active_region_ids", None),
            getattr(self, "_step4_csa_active_slot_by_region", None),
            getattr(self, "_step4_csa_numerator_cache", None),
            getattr(self, "_step4_csa_denominator_cache", None),
            getattr(self, "_step4_csa_max_cache", None),
            getattr(self, "_step4_csa_active_token_valid", None),
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
        active_region_ids = getattr(self, "_step4_csa_active_region_ids", None)
        if isinstance(active_region_ids, torch.Tensor) and active_region_ids.numel() > 0:
            warmup_block = torch.zeros(
                (1,), device=self.mean_cache.device, dtype=torch.long
            )
            self.reset_scheduler_blocks(warmup_block)


class AscendStep4SparseSummaryCacheLayer(Step4SparseSummaryCacheLayer):
    """Binds ``AscendStep4SparseSummaryCache`` instead of the upstream class."""

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
            _is_decode_bench = (
                _cfg.kv_transfer_config is not None
                and _cfg.kv_transfer_config.kv_connector
                in ("DecodeBenchConnector", "PreFilledDecodeBenchConnector")
            )
            if not _is_decode_bench:
                raise RuntimeError(
                    "Step4 DSA sparse summary side storage is not compatible with "
                    "v1 KV transfer/offload yet. Disable kv_transfer_config for "
                    "Step4, or use a connector implementation that explicitly "
                    "transfers the summary sidecar."
                )
        self._ensure_shared_scratch_workspace(forward_context)

        owner_layer = forward_context.get(self._main_layer_name)
        if owner_layer is None:
            raise RuntimeError(
                "Step4 sparse summary cache owner layer is missing: "
                f"{self._main_layer_name!r}."
            )
        owner_kv_cache = owner_layer.kv_cache
        if isinstance(owner_kv_cache, (list, tuple)):
            if len(owner_kv_cache) == 0:
                raise RuntimeError(
                    "Step4 sparse summary cache owner KV cache is empty."
                )
            owner_k_cache = owner_kv_cache[0]
        else:
            owner_k_cache = owner_kv_cache
        if not isinstance(owner_k_cache, torch.Tensor) or owner_k_cache.ndim < 2:
            raise RuntimeError(
                "Step4 sparse summary cache owner KV cache has invalid shape: "
                f"{type(owner_kv_cache)!r}."
            )
        # The summary sidecar lives in scheduler-block space: one page per
        # scheduler block, so the KV block table and slot_mapping address it
        # directly. Ascend may re-slice the owner KV tensor into smaller
        # kernel blocks (virtual block splitting); derive the scheduler-block
        # count from the total token capacity instead of the tensor shape.
        block_size = self._kv_cache_block_size
        if block_size is None:
            raise RuntimeError(
                "Step4 sparse summary cache was bound before its KV spec was "
                "created."
            )
        if isinstance(owner_kv_cache, (list, tuple)):
            token_capacity = int(owner_k_cache.shape[0]) * int(owner_k_cache.shape[1])
        else:
            token_capacity = int(owner_k_cache.shape[1]) * int(owner_k_cache.shape[2])
        if token_capacity % block_size != 0:
            raise RuntimeError(
                "Step4 sparse summary cache owner KV capacity does not divide "
                f"into scheduler blocks: tokens={token_capacity}, "
                f"block_size={block_size}."
            )
        num_blocks = token_capacity // block_size
        summary_dtype = torch.uint8
        proxy_dim = int(getattr(self._sparse_config, "proxy_dim", 0) or 0)
        num_proxy_kv_heads = int(
            getattr(self._target_impl, "summary_cache_num_proxy_kv_heads", 1) or 1
        )
        payload_size_bytes = step4_sparse_summary_cache_bytes_per_block(
            block_size=block_size,
            region_block_size=int(self._target_impl.sparse_region_block_size),
            proxy_dim=proxy_dim,
            num_kv_heads=num_proxy_kv_heads,
            sum_dtype=summary_dtype,
            count_dtype=summary_dtype,
        )
        backing_bytes = (
            (int(payload_size_bytes) + int(block_size) - 1) // int(block_size)
        ) * int(block_size)
        reverse_map_bytes = (
            Step4SparseSummaryCacheConfig.max_region_fragments_per_block(
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
        self._bind_summary_cache(backing, summary_dtype=summary_dtype)
        cache = self._target_impl._summary_cache
        if cache is None:
            raise RuntimeError("Step4 sparse summary cache side storage was not bound.")
        cache.blocks_per_scheduler_block = 1
        cache.warmup_scheduler_block_reset()
        self._budget_only_cache = cache
        if has_kv_transfer_group() and _is_decode_bench:
            import torch as _torch
            _fill_mean = 0.015
            try:
                _kt_cfg = _cfg.kv_transfer_config
                _fill_mean = float(
                    _kt_cfg.get_from_extra_config("fill_mean", 0.015)
                )
            except Exception:
                pass
            with _torch.no_grad():
                fp16_vals = _torch.full(
                    cache.mean_cache.shape,
                    _fill_mean,
                    dtype=_torch.float16,
                    device=cache.mean_cache.device,
                ).to(_torch.float8_e4m3fn)
                cache.mean_cache.view(_torch.uint8).copy_(
                    fp16_vals.view(_torch.uint8)
                )
            logger.info_once(
                "DecodeBenchConnector: filled summary mean_cache with "
                "dummy fp8 values (mean=%.6f, shape=%s)",
                _fill_mean,
                tuple(cache.mean_cache.shape),
            )
        logger.info_once(
            "Allocated Step4 sparse summary cache side storage: "
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
            raise RuntimeError(
                "Step4 sparse summary cache was bound before its KV spec was created."
            )
        if backing.dtype is not torch.uint8:
            raise RuntimeError(
                "Step4 sparse summary cache backing tensor must be uint8, "
                f"got {backing.dtype}."
            )
        if not backing.is_contiguous():
            raise RuntimeError(
                "Step4 sparse summary cache backing tensor must be contiguous, "
                f"got stride={backing.stride()}."
            )
        proxy_dim = int(getattr(self._sparse_config, "proxy_dim", 0) or 0)
        num_proxy_kv_heads = int(
            getattr(self._target_impl, "summary_cache_num_proxy_kv_heads", 1) or 1
        )
        config = Step4SparseSummaryCacheConfig(
            num_pages=int(backing.shape[0]),
            page_size=int(block_size),
            region_block_size=int(self._target_impl.sparse_region_block_size),
            num_kv_heads=num_proxy_kv_heads,
            proxy_dim=proxy_dim,
        )
        summary_dtype = summary_dtype or torch.uint8
        expected_bytes = step4_sparse_summary_cache_bytes_per_block(
            block_size=config.page_size,
            region_block_size=config.region_block_size,
            proxy_dim=config.proxy_dim,
            num_kv_heads=config.num_kv_heads,
            sum_dtype=torch.uint8,
            count_dtype=torch.uint8,
        )
        if int(backing.shape[1]) < expected_bytes:
            raise RuntimeError(
                "Step4 sparse summary cache backing tensor shape mismatch: "
                f"expected at least bytes_per_page={expected_bytes}, "
                f"got shape={tuple(backing.shape)}."
            )
        sum_elems = (
            int(config.summaries_per_page)
            * int(config.num_kv_heads)
            * int(config.proxy_dim)
        )
        if int(backing.shape[1]) < sum_elems:
            raise RuntimeError(
                "Step4 sparse mean cache backing is too small: "
                f"need={sum_elems}, got={int(backing.shape[1])}"
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
        summary_cache = AscendStep4SparseSummaryCache(
            config=config,
            sum_cache=sum_cache,
            count_cache=count_cache,
            mean_cache=mean_cache,
        )
        self._target_impl.bind_summary_cache(summary_cache)
        self._bound_summary_config = config
