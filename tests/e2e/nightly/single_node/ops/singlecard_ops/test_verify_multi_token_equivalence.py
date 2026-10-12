#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
"""E2E UT (single NPU): MTP verify metadata flattening equivalence.

The MTP verify path keeps each request a single multi-token varlen request
(``query_len = K+1``): the fused indexer's per-token mask3 bound
``floor((actual_kv_lengths[n] + j) / region_size)`` and the aclnn
GenericBlockSparseAttention right-aligned causal bound (``seqused_kv`` with
``q_len = K+1``) hold every row to its own position as causal bound.

This test pins the equivalence against the alternative flattened metadata
(one single-token request per row, ``kv_seqlen = position + 1``): with
identical inputs, both metadata encodings must select bitwise-identical
top-k region ids and produce identical attention outputs. Contexts are
chosen to span the causal regimes: sub-region, region-boundary aligned,
exact-fill (bound == topk) and 512-region (bound == topk) contexts.
"""

import pytest
import torch
import torch_npu

from vllm_ascend.utils import enable_custom_op

torch_npu.npu.config.allow_internal_format = True
enable_custom_op()

from vllm_ascend.models.step5.generic_block_sparse_attention import (  # noqa: E402
    gbsa_expected_counts,
    gbsa_sparse_attention,
    gbsa_sparse_attention_metadata,
)
from vllm_ascend.models.step5.indexer_region_score import (  # noqa: E402
    indexer_score_topk,
)

DEVICE = "npu:0"

# Production-derived constants (see test_indexer_region_score.py).
G = 1  # kv groups per rank after TP4 splits the model's 4 groups
H = 4  # index heads per group (== query heads per kv group here)
D = 256  # index proxy dim
QD = 128  # attention head dim
REGION_SIZE = 8
TOPK = 512
SUMMARIES_PER_PAGE = 16  # block_size 128 / region_size 8
PAGE_SIZE = 128

K = 4  # verify rows per request (query_len = K)
N = 4
CTX_LIST = [5, 8, 63, 4096]  # committed context per request
SEQ_LENS = [c + K for c in CTX_LIST]
TOTAL = N * K
MAX_REGION_LEN = 520  # covers the largest context's region count (512)
SOFTMAX_SCALE = 0.125


def _build_inputs(seed: int = 0):
    torch.manual_seed(seed)
    positions = torch.cat(
        [torch.arange(c, c + K, device=DEVICE) for c in CTX_LIST]
    )
    pages_per_req = (MAX_REGION_LEN + SUMMARIES_PER_PAGE - 1) // SUMMARIES_PER_PAGE
    num_pages = N * pages_per_req
    block_table = (
        torch.randperm(num_pages, dtype=torch.int32)
        .reshape(N, pages_per_req)
        .to(DEVICE)
    )
    index_q = torch.randn(TOTAL, G, H, D, device=DEVICE, dtype=torch.bfloat16)
    weights = torch.randn(TOTAL, G, H, device=DEVICE, dtype=torch.float32)
    mean_cache = (
        torch.randn(num_pages, SUMMARIES_PER_PAGE, 1, D, dtype=torch.bfloat16)
        .to(torch.float8_e4m3fn)
        .to(DEVICE)
    )
    query = torch.randn(TOTAL, H, QD, device=DEVICE, dtype=torch.bfloat16)
    key_cache = torch.randn(
        num_pages, PAGE_SIZE, G, QD, device=DEVICE, dtype=torch.bfloat16
    )
    value_cache = torch.randn(
        num_pages, PAGE_SIZE, G, QD, device=DEVICE, dtype=torch.bfloat16
    )
    return positions, block_table, index_q, weights, mean_cache, query, key_cache, value_cache


def _run(bundle, multi_token: bool):
    """Run the indexer + GBSA chain under one of the two metadata encodings."""
    positions, block_table, index_q, weights, mean_cache, query, key_cache, value_cache = bundle
    seq_lens = torch.tensor(SEQ_LENS, dtype=torch.int64, device=DEVICE)
    if multi_token:
        # One varlen request per verify request (the serving path).
        # actual_kv_lengths holds the effective KV length including the
        # current queries (seq_len); the operator restores the first query
        # position as seq_len - q_len.
        bt = block_table[:N]
        cu_seq_lens = torch.arange(N + 1, device=DEVICE, dtype=torch.int32) * K
        seq_lens_i32 = seq_lens[:N].to(torch.int32)
        actual_kv_lengths = seq_lens_i32
        max_query_len = K
        seqused_kv = seq_lens_i32
    else:
        # Flattened encoding: one single-token request per row. Its
        # effective KV length is the row's position + 1.
        token_to_req = torch.arange(TOTAL, device=DEVICE, dtype=torch.int32) // K
        bt = block_table[:N][token_to_req.to(torch.long)]
        cu_seq_lens = torch.arange(TOTAL + 1, device=DEVICE, dtype=torch.int32)
        row_kv_seqlens = positions[:TOTAL].to(torch.int32) + 1
        actual_kv_lengths = row_kv_seqlens
        max_query_len = 1
        seqused_kv = row_kv_seqlens

    _scores, ids = indexer_score_topk(
        index_q[:TOTAL],
        weights[:TOTAL],
        mean_cache,
        bt,
        cu_seq_lens,
        actual_kv_lengths,
        max_region_len=MAX_REGION_LEN,
        topk=TOPK,
        region_size=REGION_SIZE,
    )
    # Per-step AICPU scheduling metadata, as the builder produces it for
    # each encoding: position-derived counts plus the encoding's own
    # seq-lens / max query length.
    counts = gbsa_expected_counts(
        positions[:TOTAL],
        groups=G,
        region_size=REGION_SIZE,
        topk=TOPK,
    )
    metadata = gbsa_sparse_attention_metadata(
        counts,
        num_heads=H,
        num_kv_groups=G,
        head_dim=QD,
        region_size=REGION_SIZE,
        topk=TOPK,
        cu_seqlens_q=cu_seq_lens,
        seqused_kv=seqused_kv,
        max_query_len=max_query_len,
        max_pages=int(bt.shape[1]),
        page_size=PAGE_SIZE,
    )
    out = gbsa_sparse_attention(
        query,
        key_cache,
        value_cache,
        ids,
        positions[:TOTAL],
        bt,
        cu_seq_lens,
        seqused_kv,
        metadata,
        num_kv_groups=G,
        region_size=REGION_SIZE,
        softmax_scale=SOFTMAX_SCALE,
    )
    return ids, out


@pytest.mark.skipif(not torch.npu.is_available(), reason="requires NPU")
def test_verify_multi_token_matches_flattened():
    bundle = _build_inputs()
    ids_flat, out_flat = _run(bundle, multi_token=False)
    ids_multi, out_multi = _run(bundle, multi_token=True)

    assert ids_flat.shape == (G * TOTAL, TOPK)
    # Identical inputs + identical per-row causal bounds => identical
    # selection, bitwise.
    assert torch.equal(ids_flat, ids_multi)
    # Identical selection and identical per-row causal masks => identical
    # attention output.
    torch.testing.assert_close(out_flat, out_multi, rtol=1e-2, atol=1e-2)
