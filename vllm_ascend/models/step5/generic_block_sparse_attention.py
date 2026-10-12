# SPDX-License-Identifier: Apache-2.0
"""aclnn GenericBlockSparseAttention adapter.

Wraps ``generic_block_sparse_attention`` (+ its AICPU metadata op) from
ops-transformer as the exact-attention replacement for the Triton
``sparse_attention_decode`` / ``sparse_attention_prefill`` kernels.

Contract parity with the previous pack + Triton chain:

- ``ids`` is the ``indexer_score_topk`` output: ``[G*T, topk]`` int32 logical
  region ids, ascending, ``-1`` tail padding, current region NOT included
  (the fused indexer's mask3 bound keeps it out of the top-k pool).
- The current region ``floor(position / region_size)`` is appended here into
  a ``topk + 1`` wide tensor at slot ``count`` -- the same slot the Triton
  pack kernels (``decode_pack_from_ids`` / ``prefill_pack_from_ids``) used
  for their extra column -- so a full top-k selection plus the current
  region still fits without evicting a selected id.
- ``mask_mode=CAUSAL`` with right-aligned varlen (``cu_seqlens_q`` +
  ``seqused_kv``) reproduces the per-token causal truncation the pack
  kernels expressed through per-region ``valid`` counts: query ``i`` of a
  request sits at kv position ``seqused_kv - q_len + i`` and attends only
  tokens at positions ``<=`` its own.
- K/V stay in the native paged layout: the Step5 KV cache is
  ``[num_blocks, page_size, G, D]``, which is exactly the operator's
  ``PA_BBND`` layout, so the 4-D views are passed through untouched and the
  operator resolves logical region -> physical page through ``block_table``.

``is_consistent_topk`` is fixed to ``False``: live counts vary per row in
serving, and the operator's consistent-count fast path does not apply.
"""

from __future__ import annotations

import torch

from cann_ops_transformer.ops.attention.generic_block_sparse_attention import (
    generic_block_sparse_attention,
    generic_block_sparse_attention_metadata,
)

_SPARSE_PATTERN_PER_GROUP_PER_TOKEN = 4
_MASK_CAUSAL = 1
_QUANT_NONE = 0
_SOFTMAX_PRECISION_FP32 = 1


try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False

_TRITON_DEVICE_TYPES = ("cuda", "npu", "xpu")

if _HAS_TRITON:

    @triton.jit
    def _append_current_region_kernel(
        ids_ptr,
        positions_ptr,
        wide_ptr,
        counts_ptr,
        tokens,
        region_size,
        topk: tl.constexpr,
        block_k: tl.constexpr,
        block_t: tl.constexpr,
    ):
        """Pack ``block_t`` rows of one group, streaming ``topk`` in chunks."""
        group = tl.program_id(0)
        tile = tl.program_id(1)
        row = tile * block_t + tl.arange(0, block_t)
        row_mask = row < tokens
        rows = group * tokens + row

        live = tl.zeros((block_t,), dtype=tl.int32)
        for k0 in range(0, topk, block_k):
            col = k0 + tl.arange(0, block_k)
            mask = row_mask[:, None] & (col < topk)[None, :]
            ids = tl.load(
                ids_ptr + rows[:, None] * topk + col[None, :],
                mask=mask,
                other=-1,
            )
            live += tl.sum((ids != -1).to(tl.int32), axis=1)
            tl.store(
                wide_ptr + rows[:, None] * (topk + 1) + col[None, :],
                ids,
                mask=mask,
            )

        pad = tl.full((block_t,), -1, dtype=tl.int32)
        tl.store(wide_ptr + rows * (topk + 1) + topk, pad, mask=row_mask)

        positions = tl.load(positions_ptr + row, mask=row_mask, other=0)
        current = (positions // region_size).to(tl.int32)
        slot = tl.minimum(live, topk)
        tl.store(wide_ptr + rows * (topk + 1) + slot, current, mask=row_mask)

        tl.store(counts_ptr + rows, live + 1, mask=row_mask)


def _append_current_region_torch(
    ids: torch.Tensor,
    positions: torch.Tensor,
    *,
    groups: int,
    region_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference implementation used for CPU tensors."""
    rows, topk = ids.shape
    tokens = rows // groups
    ids3 = ids.view(groups, tokens, topk)
    counts = (ids3 != -1).sum(dim=-1, dtype=torch.int32)
    wide = torch.cat(
        [
            ids3,
            torch.full(
                (groups, tokens, 1), -1, device=ids.device, dtype=ids.dtype
            ),
        ],
        dim=-1,
    )
    current = (positions.to(torch.int32) // region_size).view(1, tokens, 1)
    slot = counts.clamp(max=topk).to(torch.int64).unsqueeze(-1)
    wide.scatter_(-1, slot, current.expand(groups, tokens, 1))
    return wide, counts + 1


def _append_current_region_triton(
    ids: torch.Tensor,
    positions: torch.Tensor,
    *,
    groups: int,
    region_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch the fused Triton pack kernel on device tensors."""
    rows, topk = ids.shape
    tokens = rows // groups
    wide = torch.empty(
        (groups, tokens, topk + 1), device=ids.device, dtype=ids.dtype
    )
    counts = torch.empty(
        (groups, tokens), device=ids.device, dtype=torch.int32
    )
    block_k = triton.next_power_of_2(min(topk, 512))
    block_t = max(1, min(64, 8192 // block_k))
    grid = (groups, triton.cdiv(tokens, block_t))
    _append_current_region_kernel[grid](
        ids,
        positions,
        wide,
        counts,
        tokens,
        region_size,
        topk=topk,
        block_k=block_k,
        block_t=block_t,
    )
    return wide, counts


def _append_current_region(
    ids: torch.Tensor,
    positions: torch.Tensor,
    *,
    groups: int,
    region_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Append each row's current region into a ``topk + 1`` wide id tensor.

    ``ids`` is ``[G*T, topk]`` ascending logical ids with ``-1`` tail padding;
    the selection pool is strictly below ``floor(position / region_size)``,
    so the appended current region is larger than every selected id and the
    row stays ascending. It lands at slot ``count`` (the extra ``topk``-th
    column when the selection is full), matching the Triton pack kernels'
    current-region append. Counts become ``live + 1``.

    Returns ``(idx [G, T, topk + 1], counts [G, T])`` ready for the operator.
    """
    if _HAS_TRITON and ids.device.type in _TRITON_DEVICE_TYPES:
        return _append_current_region_triton(
            ids, positions, groups=groups, region_size=region_size
        )
    return _append_current_region_torch(
        ids, positions, groups=groups, region_size=region_size
    )


def gbsa_sparse_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    ids: torch.Tensor,
    positions: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seqused_kv: torch.Tensor,
    *,
    num_kv_groups: int,
    region_size: int,
    softmax_scale: float,
    max_query_len: int,
) -> torch.Tensor:
    """Sparse attention over per-(group, token) selected logical regions.

    Args:
        query: ``[total_q, num_heads, head_dim]`` (``TND``).
        key_cache, value_cache: ``[num_blocks, page_size, G, head_dim]`` paged
            cache views (``PA_BBND``), passed through to the operator.
        ids: ``[G * total_q, topk]`` int32 logical region ids, ascending,
            ``-1`` tail padding, current region not included.
        positions: ``[total_q]`` absolute position of each query token; the
            current region is derived per token and appended internally.
        block_table: ``[num_reqs, max_pages]`` int32 logical-to-physical map.
        cu_seqlens_q: ``[num_reqs + 1]`` cumulative query lengths.
        seqused_kv: ``[num_reqs]`` used KV length per request (right-aligned
            causal anchor; must be ``>=`` the request's query length).
        num_kv_groups: G, the number of KV head groups.
        region_size: tokens per region (sparse block height).
        softmax_scale: attention softmax scale.
        max_query_len: max per-request query length (1 for the decode path,
            K+1 for the MTP verify path).

    Returns:
        ``out`` shaped ``[total_q, num_heads, head_dim]``, same dtype as
        ``query``.
    """
    total_q, num_heads, head_dim = query.shape
    idx, counts = _append_current_region(
        ids, positions, groups=num_kv_groups, region_size=region_size
    )
    block_shape = [1, region_size]
    cu_q = cu_seqlens_q.to(torch.int64)
    used_kv = seqused_kv.to(torch.int32)
    # Static upper bound from the table width: keeps the launch shapes
    # graph-capturable (no device max reduction, no host sync).
    max_seqlen_kv = int(block_table.shape[1]) * int(key_cache.shape[1])

    common = dict(
        cu_seqlens_q=cu_q,
        seqused_kv=used_kv,
        layout_q="TND",
        layout_kv="PA_BBND",
        layout_sparse_pattern=_SPARSE_PATTERN_PER_GROUP_PER_TOKEN,
        mask_mode=_MASK_CAUSAL,
        quant_mode=_QUANT_NONE,
        softmax_precision=_SOFTMAX_PRECISION_FP32,
        is_consistent_topk=False,
    )
    metadata = generic_block_sparse_attention_metadata(
        idx,
        counts,
        num_heads,
        num_kv_groups,
        head_dim,
        block_shape,
        max_seqlen_q=max_query_len,
        max_seqlen_kv=max_seqlen_kv,
        **common,
    )
    out, _lse = generic_block_sparse_attention(
        query,
        key_cache,
        value_cache,
        idx,
        counts,
        block_shape,
        metadata=metadata,
        block_table=block_table,
        softmax_scale=softmax_scale,
        return_softmax_lse=False,
        **common,
    )
    return out
