# SPDX-License-Identifier: Apache-2.0
"""aclnn Step4BlockSparseAttention operator adapter for Ascend 950 (A5).

The operator (from ``ops-transformer``) computes the Step-4 DSA main attention:
QK → softmax → PV gathered from the flat TGD main KV cache over the regions
selected by the indexer. It consumes the same ``packed_regions`` /
``region_counts`` encoding the local Triton selectors already emit, so both
paths share the metadata stage.

Prefill and Decode share a single aclnn and kernel. Dispatch is by
``packed_regions.dtype``: INT32 → Prefill, INT64 → Decode.
"""

from __future__ import annotations

import torch

from cann_ops_transformer_custom.ops import (
    step4_block_sparse_attention,
)


def step4_block_sparse_attention_prefill(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    packed_regions: torch.Tensor,
    region_counts: torch.Tensor,
    *,
    num_kv_groups: int,
    region_size: int = 8,
    page_size: int = 128,
    softmax_scale: float | None = None,
    return_softmax_lse: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """aclnn operator version of ``sparse_attention_prefill``.

    ``packed_regions`` must be INT32 (Prefill mode).
    """
    num_heads = query.shape[1]
    out, lse = step4_block_sparse_attention(
        query,
        key_cache,
        value_cache,
        packed_regions,
        region_counts,
        num_heads=num_heads,
        num_key_value_heads=num_kv_groups,
        kv_seqlens=None,
        region_size=region_size,
        page_size=page_size,
        scale_value=softmax_scale if softmax_scale is not None else 0.0,
        layout_q="TND",
        layout_kv="TGD",
        return_softmax_lse=return_softmax_lse,
        metadata=None,
    )
    return out, lse


def step4_block_sparse_attention_decode(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    packed_regions: torch.Tensor,
    region_counts: torch.Tensor,
    kv_seqlens: torch.Tensor,
    *,
    num_kv_groups: int,
    region_size: int = 8,
    page_size: int = 128,
    softmax_scale: float | None = None,
    return_softmax_lse: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """aclnn operator version of ``sparse_attention_decode``.

    ``packed_regions`` must be INT64 (Decode mode). ``kv_seqlens`` is required.
    """
    num_heads = query.shape[1]
    out, lse = step4_block_sparse_attention(
        query,
        key_cache,
        value_cache,
        packed_regions,
        region_counts,
        num_heads=num_heads,
        num_key_value_heads=num_kv_groups,
        kv_seqlens=kv_seqlens,
        region_size=region_size,
        page_size=page_size,
        scale_value=softmax_scale if softmax_scale is not None else 0.0,
        layout_q="TND",
        layout_kv="TGD",
        return_softmax_lse=return_softmax_lse,
        metadata=None,
    )
    return out, lse
