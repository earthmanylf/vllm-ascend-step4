#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
"""NPU test: Step4 paged indexer scoring kernel vs torch reference.

Covers ``vllm_ascend.models.step4.dsa_kernels.indexer_logits_paged`` (the
``_indexer_logits_kernel`` launch wrapper): weighted-ReLU indexer scores
against the paged fp8 summary cache, one kernel launch per batch.

Batch shapes under test:

* decode: one query token per request (identity token -> request map),
* spec-decode verify: ``k`` query tokens per request (repeated map),
* packed prefill: mixed query lengths incl. zero-length requests, unmapped
  block-table pages (``-1``), and more key columns than one key tile.

Also pins the two-level mean-cache addressing: a cache whose page stride
exceeds ``summaries_per_page * proxy_dim`` (padded sidecar backing) must be
read through the page/fragment strides, not a flat region index.

The reference quantizes queries to e4m3 on the host (deployed semantics,
matching upstream's host pre-quantization for its WGMMA entrypoint) and
accumulates in fp32.
"""

import torch

from vllm_ascend.models.step4.dsa_kernels import indexer_logits_paged

DEVICE = "npu"

NUM_PAGES = 32
SUMMARIES_PER_PAGE = 2
PROXY_DIM = 256
GROUPS = 4
HEADS = 4


def _make_cache(padded: bool = False) -> torch.Tensor:
    """Random fp8 mean cache; ``padded`` adds inter-page stride slack."""
    row = SUMMARIES_PER_PAGE * 1 * PROXY_DIM
    stride0 = row + (64 if padded else 0)
    backing = torch.zeros((NUM_PAGES, stride0), device=DEVICE, dtype=torch.uint8)
    mean_cache = (
        backing[:, :row]
        .view(NUM_PAGES, SUMMARIES_PER_PAGE, 1, PROXY_DIM)
        .view(torch.float8_e4m3fn)
    )
    mean_cache.copy_(
        torch.randn(mean_cache.shape, device=DEVICE).to(torch.float8_e4m3fn)
    )
    return mean_cache


def _make_qw(total_tokens: int):
    q = torch.randn(
        total_tokens, GROUPS, HEADS, PROXY_DIM, device=DEVICE, dtype=torch.bfloat16
    )
    w = torch.randn(
        total_tokens, GROUPS, HEADS, device=DEVICE, dtype=torch.float32
    )
    return q, w


def _ref_scores(mean_cache, block_table, token_to_req, q, w, width):
    """score[t, r] = sum_h relu(dot(q_th, k_r)) * w_th, fp32 accumulation.

    Group-major output rows (``group * total + t``), matching the kernel's
    write contract; ``token_to_req`` selects each token's key stream.
    """
    total = q.shape[0]
    req = torch.as_tensor(token_to_req, device=DEVICE, dtype=torch.long)
    cols = torch.arange(width, device=DEVICE)
    page_cols = (cols // SUMMARIES_PER_PAGE).expand(total, -1)
    frags = (cols % SUMMARIES_PER_PAGE).expand(total, -1)
    pages = block_table[req[:, None], page_cols]  # [total, width]
    ok = pages >= 0
    # Widen before advanced indexing: aclnnIndex rejects fp8 dtypes, and the
    # e4m3 -> fp32 widening is exact.
    kv = mean_cache.to(torch.float32)[pages.clamp_min(0).long(), frags, 0]
    q_r = q.to(torch.float8_e4m3fn).to(torch.bfloat16).float()
    out = torch.zeros(GROUPS * total, width, device=DEVICE)
    for g in range(GROUPS):
        dots = torch.einsum("thd,trd->trh", q_r[:, g], kv)
        scores = (dots.relu() * w[:, g, None, :]).sum(-1)
        out[g * total : (g + 1) * total] = torch.where(ok, scores, 0.0)
    return out


def test_decode_identity_map():
    torch.manual_seed(0)
    mean_cache = _make_cache()
    n = 6
    bt = torch.randint(0, NUM_PAGES, (n, 3), device=DEVICE, dtype=torch.int32)
    bt[2, 2] = -1  # one request with an unmapped tail page
    width = 3 * SUMMARIES_PER_PAGE
    q, w = _make_qw(n)
    out = torch.zeros(GROUPS * n, width, device=DEVICE, dtype=torch.float32)
    indexer_logits_paged(
        q,
        w,
        mean_cache,
        bt,
        torch.arange(n, device=DEVICE, dtype=torch.int32),
        out=out,
        summaries_per_page=SUMMARIES_PER_PAGE,
    )
    torch.testing.assert_close(
        out, _ref_scores(mean_cache, bt, list(range(n)), q, w, width),
        rtol=1e-4, atol=1e-4,
    )


def test_spec_decode_verify_repeated_map():
    torch.manual_seed(1)
    mean_cache = _make_cache()
    k = 3
    n = 2
    bt = torch.randint(0, NUM_PAGES, (n, 3), device=DEVICE, dtype=torch.int32)
    width = 3 * SUMMARIES_PER_PAGE
    q, w = _make_qw(n * k)
    token_to_req = [row for row in range(n) for _ in range(k)]
    out = torch.zeros(GROUPS * n * k, width, device=DEVICE, dtype=torch.float32)
    indexer_logits_paged(
        q,
        w,
        mean_cache,
        bt,
        torch.tensor(token_to_req, device=DEVICE, dtype=torch.int32),
        out=out,
        summaries_per_page=SUMMARIES_PER_PAGE,
    )
    torch.testing.assert_close(
        out, _ref_scores(mean_cache, bt, token_to_req, q, w, width),
        rtol=1e-4, atol=1e-4,
    )


def test_prefill_packed_batch():
    torch.manual_seed(2)
    mean_cache = _make_cache()
    # Mixed query lengths with a zero-length request, one unmapped page, and
    # 40 key columns so the default block_k=32 needs two key tiles.
    lens = [5, 0, 7]
    pages = 20
    total = sum(lens)
    width = pages * SUMMARIES_PER_PAGE
    bt = torch.randint(0, NUM_PAGES, (3, pages), device=DEVICE, dtype=torch.int32)
    bt[2, 4] = -1
    q, w = _make_qw(total)
    token_to_req = [row for row, length in enumerate(lens) for _ in range(length)]
    out = torch.zeros(GROUPS * total, width, device=DEVICE, dtype=torch.float32)
    indexer_logits_paged(
        q,
        w,
        mean_cache,
        bt,
        torch.tensor(token_to_req, device=DEVICE, dtype=torch.int32),
        out=out,
        summaries_per_page=SUMMARIES_PER_PAGE,
    )
    torch.testing.assert_close(
        out, _ref_scores(mean_cache, bt, token_to_req, q, w, width),
        rtol=1e-4, atol=1e-4,
    )


def test_padded_mean_cache_page_stride():
    torch.manual_seed(3)
    # Regression: the cache backing may carry bytes past the mean payload, so
    # stride(0) > summaries_per_page * proxy_dim. Physical addressing must go
    # through the page/fragment strides instead of a flat region index.
    mean_cache = _make_cache(padded=True)
    assert mean_cache.stride(0) > SUMMARIES_PER_PAGE * PROXY_DIM
    q, w = _make_qw(2)
    bt = torch.randint(0, NUM_PAGES, (1, 3), device=DEVICE, dtype=torch.int32)
    width = 3 * SUMMARIES_PER_PAGE
    out = torch.zeros(GROUPS * 2, width, device=DEVICE, dtype=torch.float32)
    indexer_logits_paged(
        q,
        w,
        mean_cache,
        bt,
        torch.zeros(2, device=DEVICE, dtype=torch.int32),
        out=out,
        summaries_per_page=SUMMARIES_PER_PAGE,
    )
    torch.testing.assert_close(
        out, _ref_scores(mean_cache, bt, [0, 0], q, w, width),
        rtol=1e-4, atol=1e-4,
    )
