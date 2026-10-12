# SPDX-License-Identifier: Apache-2.0
"""aclnn IndexerRegionScore (fused score + topk + sort) adapter.

Wraps ``indexer_region_score`` from ops-transformer, which fuses region
scoring, causal masking (mask_mode=3) and TopK selection with ascending
logical-id sort (``top_k`` + ``sort_mode=1``) into a single operator call.

The operator returns ``scores [G,T,R]`` and fills the caller-allocated
``selected_region_ids [G,T,top_k]`` INT32 (logical region ids, ascending,
-1 padded). It performs **no** physical-page mapping, current-region append
or metadata packing — those stay in the Triton pack kernels
(``_decode_pack_kernel`` / ``_append_current_region_kernel``), whose
``ids [rows, topk]`` input contract is satisfied by a zero-copy reshape of
the operator output.
"""

from __future__ import annotations

import torch

from cann_ops_transformer.ops.attention.indexer_region_score import (
    indexer_region_score,
)


def indexer_score_topk(
    index_q: torch.Tensor,
    weights: torch.Tensor,
    mean_cache: torch.Tensor,
    block_table: torch.Tensor,
    cu_seq_lens: torch.Tensor,
    actual_kv_lengths: torch.Tensor,
    *,
    max_region_len: int,
    topk: int,
    region_size: int = 8,
    # Weights arrive pre-scaled by H**-0.5 (model.py scales them before the
    # attention call), so no extra scaling is applied here.
    head_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused score + topk + sort via aclnnIndexerRegionScore.

    Args:
        index_q:  [T, G, H, D] — quantized to float8_e4m3fn internally.
        weights:  [T, G, H] fp32 — signed per-head weights.
        mean_cache: [P, S, 1, D] float8_e4m3fn — paged summary cache.
        block_table: [N, C] int32 — logical-to-physical page map.
        cu_seq_lens: [N+1] int32 — cumulative packed query lengths.
        actual_kv_lengths: [N] int32 — effective KV sequence length of
            each request including the current queries (``seq_len``;
            mask_mode=3 causal input). The operator restores the first
            query position as ``actual_kv_lengths[n] - q_len`` (``q_len``
            from ``cu_seq_lens``) and selects only from regions strictly
            below ``floor((ctx + j) / region_size)``, i.e. the current
            region never competes for a slot.
        max_region_len: number of logical regions to score.
        topk: history candidate count.
        region_size: tokens per region (8).

    Returns:
        (scores [G*T, max_region_len] fp32, ids [G*T, topk] int32) — both
        group-major 2D views of the operator's [G,T,...] outputs, ready for
        the Triton pack kernels. ``ids`` rows hold logical region ids in
        ascending order with ``-1`` tail padding, the exact contract of
        ``region_topk_ids`` / ``topk_select_new`` outputs.
    """
    tokens, groups = index_q.shape[:2]
    selected_ids = torch.empty(
        (groups, tokens, topk),
        device=index_q.device,
        dtype=torch.int32,
    )
    scores = indexer_region_score(
        index_q.to(torch.float8_e4m3fn),
        weights,
        paged_key_summary=mean_cache,
        block_table=block_table,
        cu_seq_lens=cu_seq_lens,
        actual_kv_lengths=actual_kv_lengths,
        mask_mode=3,
        head_scale=head_scale,
        cache_mode=1,
        max_region_len=max_region_len,
        region_size=region_size,
        top_k=topk,
        sort_mode=1,
        selected_region_ids=selected_ids,
    )
    return (
        scores.reshape(groups * tokens, scores.shape[-1]),
        selected_ids.reshape(groups * tokens, topk),
    )
