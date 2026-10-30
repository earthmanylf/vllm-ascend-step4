# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend port of Step5 TailRing region summaries.

Semantic source: vLLM main ``vllm/models/step5/nvidia/sparse_summary_cache.py``
(commit 8407bc0bb6).

Design note (why this is much smaller than the previous SWF port):
upstream replaced per-page FP32 accumulators plus a per-request undo journal
with a circular buffer of the raw ``[key | z]`` rows. A region summary is
computed once, by the row that closes the region, by replaying the eight rows
from the ring. Because only closed regions are ever scored, a rejected draft
that closed a region needs no rollback -- the next step overwrites its row and
the region is recomputed when it closes again. The cache therefore holds no
per-request state at all; ``mean`` is the only member.

The Ascend updates route to ``summary_kernels.py`` instead of main's CUDA
kernels. Ascend also exposes the summary group's own addresses as metadata;
physical page padding does not imply shared manager block IDs.
"""

from __future__ import annotations

import os
import weakref
from dataclasses import dataclass, replace
from typing import Any

import torch
from vllm.logger import logger
from vllm.models.step5.nvidia.sparse_summary_cache import (
    Step5SparseSummaryBackend,
    Step5SparseSummaryCacheLayer,
    Step5SparseSummaryMetadataBuilder,
)
from vllm.v1.attention.backend import AttentionMetadata, CommonAttentionMetadata

from vllm_ascend.core.step5_kv_cache import uses_step5_blnhc


@dataclass
class AscendStep5SparseSummaryMetadata(AttentionMetadata):
    """Addresses from the summary KV group, in full-batch token/request order."""

    slot_mapping: torch.Tensor
    block_table: torch.Tensor


class AscendStep5SparseSummaryMetadataBuilder(Step5SparseSummaryMetadataBuilder):
    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> AscendStep5SparseSummaryMetadata:
        # Retain the runner's device buffers; graph replay refreshes them in place.
        return AscendStep5SparseSummaryMetadata(
            slot_mapping=common_attn_metadata.slot_mapping,
            block_table=common_attn_metadata.block_table_tensor,
        )


class AscendStep5SparseSummaryBackend(Step5SparseSummaryBackend):
    @staticmethod
    def get_builder_cls() -> type[AscendStep5SparseSummaryMetadataBuilder]:
        return AscendStep5SparseSummaryMetadataBuilder


class AscendStep5SparseSummaryCache:
    """main's TailRing summary cache with explicit NPU update dispatch.

    Intentionally does NOT subclass main's ``Step5SparseSummaryCache``: that
    base binds the CUDA-libdevice kernels at import time of its methods, and
    inheriting would still expose them. This class keeps the same public
    surface (``mean`` plus the three update entry points) so callers are
    unaffected, while every update goes to the Ascend kernels.
    """

    def __init__(self, mean: torch.Tensor) -> None:
        self.mean = mean

    def update_decode(self, key, z, slots, positions, ring) -> None:
        from vllm_ascend.models.step5.summary_kernels import update_decode

        update_decode(
            key,
            z,
            slots,
            positions,
            ring.metadata().base[: slots.shape[0]],
            self.mean,
            ring.rows,
            ring.capacity,
        )

    def update_verify(self, key, z, slots, query_start_loc, seq_lens, ring) -> None:
        from vllm_ascend.models.step5.summary_kernels import update_verify

        update_verify(
            key,
            z,
            slots,
            query_start_loc,
            seq_lens,
            ring.metadata().base[: slots.shape[0]],
            self.mean,
            ring.rows,
            ring.capacity,
        )

    def update_prefill(self, key, z, metadata, positions, ring, *, token_offset=0) -> None:
        from vllm_ascend.models.step5.summary_kernels import update_prefill

        ring_metadata = ring.metadata()
        token_end = token_offset + key.shape[0]
        update_prefill(
            key,
            z,
            metadata,
            positions,
            ring_metadata.base[token_offset:token_end],
            ring_metadata.slot_mapping[token_offset:token_end],
            self.mean,
            ring.rows,
            ring.capacity,
        )


def _dense_summary_enabled() -> bool:
    """Whether a strided summary table may be replaced by a dense mirror."""
    return bool(int(os.environ.get("VLLM_ASCEND_STEP5_SUMMARY_DENSE", "1")))


def _summary_table(page: torch.Tensor, *, blnhc: bool) -> torch.Tensor:
    """Return the summary table the indexer and the summary kernels share.

    ``page`` is the BLNHC view of this layer's summary pages, whose block
    stride is the model-wide block tile rather than the 4 KiB page. The fused
    ``IndexerRegionScore`` operator only accepts a dense paged summary, so a
    strided table makes it gather every visible region into a temporary once
    per layer and step (``aclnnIndexerRegionScore_SliceAiCore_Slice``, about
    1.05 ms per decode step at 128K context / bs96 across 23 layers).

    A dense mirror with the same block indexing keeps the address contract of
    ``summary_kernels.summary_layout`` (block stride == rows * 256 bytes) and
    of the block table, so writers and readers both stay correct while the
    operator runs its single-kernel path. It is only used for the BLNHC
    layout, i.e. when no KV transfer instance aliases these pages; the
    transferable compact profile keeps its paged table untouched.

    ``VLLM_ASCEND_STEP5_SUMMARY_DENSE=0`` restores the paged table for A/B
    runs (and pays the per-layer repack kernel again).
    """
    if not blnhc or not _dense_summary_enabled() or page.is_contiguous():
        return page
    # Zeroed so a never-written region reads as an empty summary, exactly
    # like the freshly allocated paged KV it replaces.
    return torch.zeros(tuple(page.shape), dtype=page.dtype, device=page.device)


class AscendStep5SparseSummaryCacheLayer(Step5SparseSummaryCacheLayer):
    """main's summary layer, installing the Ascend cache.

    main's ``bind_kv_cache`` installs ``Step5SparseSummaryCache``, whose
    update methods bind the CUDA-libdevice kernels. Override it to install
    ``AscendStep5SparseSummaryCache`` so the NPU dispatch is used.

    Note the constructor deliberately drops the SWF-era ``main_layer_name``
    and ``sparse_config`` arguments: main's layer needs neither, and the
    side-storage budget hooks they fed (``_set_summary_cache_budget`` /
    ``_fixed_runtime_state_budget``) do not exist in the TailRing design --
    the V2 BLNHC path keeps the real compressed FullAttentionSpec and the
    upstream planner groups it with main KV. Both use the same manager IDs.
    The older padded profile remains separate for non-V2 execution.
    """

    def __init__(
        self,
        *,
        prefix: str,
        target_impl: Any,
        main_layer: Any,
        static_forward_context: dict[str, Any],
    ) -> None:
        super().__init__(
            prefix=prefix,
            target_impl=target_impl,
            static_forward_context=static_forward_context,
        )
        self._main_layer_ref = weakref.ref(main_layer)
        self._target_impl.summary_layer_name = self.layer_name
        # Set by get_kv_cache_spec. Under BLNHC the 4 KiB summary page sits in
        # a model-wide block tile, so the fused IndexerRegionScore operator
        # gathers every visible region per layer and step; _summary_table
        # replaces that strided table with a dense mirror (see its docstring).
        self._blnhc_summary = False

    def get_attn_backend(self) -> type[AscendStep5SparseSummaryBackend]:
        return AscendStep5SparseSummaryBackend

    def get_kv_cache_spec(self, vllm_config):
        spec = super().get_kv_cache_spec(vllm_config)
        main_layer = self._main_layer_ref()
        if main_layer is None:
            raise RuntimeError("Step5 summary main layer is no longer available")
        main_spec = main_layer.get_kv_cache_spec(vllm_config)
        if spec.block_size != main_spec.block_size:
            raise ValueError("Step5 summary and main KV must use the same logical block size")
        if uses_step5_blnhc(vllm_config):
            # BLNHC packs every layer page into one model-wide block tile, so a
            # 4 KiB summary page is separated from the next one by the whole
            # tile stride. The fused IndexerRegionScore operator needs a dense
            # paged summary, so keep a dense mirror as the table the indexer
            # and the summary kernels share (see _summary_table).
            #
            # A KV transfer instance (PD) moves the real tile pages through its
            # own views, and a mirror would then be written by neither side; in
            # that configuration keep the paged table and pay the repack.
            transfer = getattr(vllm_config, "kv_transfer_config", None)
            self._blnhc_summary = not (transfer is not None and transfer.is_kv_transfer_instance)
            return spec
        return replace(spec, page_size_padded=main_spec.page_size_bytes)

    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        # [blocks, 1, block_size // 8, 256] -> [blocks, block_size // 8, 256]
        self.kv_cache = kv_cache
        page = kv_cache[:, 0]
        table = _summary_table(page, blnhc=getattr(self, "_blnhc_summary", False))
        if table is not page:
            logger.info_once(
                "Step5 summary cache: dense index table %s (elements) for the "
                "indexer; BLNHC page stride %d elements would force a per-layer "
                "repack kernel every step.",
                tuple(table.shape),
                page.stride(0),
            )
            # Publish the mirror as this layer's cache so every writer that
            # reaches the summary pages through the layer (the runner's
            # KVBlockZeroer, which zeroes freshly allocated blocks through
            # `static_forward_context[layer].kv_cache`) writes the mirror and
            # not the dead tile pages. Without this the mirror keeps a previous
            # owner's summaries for a reused block and the indexer reads them.
            self.kv_cache = table.unsqueeze(1)
        self._target_impl.summary_cache = AscendStep5SparseSummaryCache(table)


__all__ = [
    "AscendStep5SparseSummaryBackend",
    "AscendStep5SparseSummaryMetadata",
    "AscendStep5SparseSummaryMetadataBuilder",
    "AscendStep5SparseSummaryCache",
    "AscendStep5SparseSummaryCacheLayer",
]
