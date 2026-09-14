#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
"""NPU test: standalone paged indexer scoring kernel vs torch reference.

Covers ``indexer_logits_paged``: weighted-ReLU indexer scores against the
paged fp8 summary cache, one kernel launch per batch.

Batch shapes under test:

* decode: one query token per request (identity token -> request map),
* spec-decode verify: ``k`` query tokens per request (repeated map),
* packed prefill: mixed query lengths, unmapped block-table pages (``-1``),
  and more key columns than one key tile.

Also pins the two-level mean-cache addressing: a cache whose page stride
exceeds ``summaries_per_page * proxy_dim`` (padded sidecar backing) must be
read through the page/fragment strides, not a flat region index.

The reference quantizes queries to e4m3 on the host (deployed semantics,
matching upstream's host pre-quantization for its WGMMA entrypoint) and
accumulates in fp32.

Usage:
    pytest test_step4_indexer_logits_paged.py -v
"""

import torch

from vllm_ascend.models.step4.dsa_kernels import indexer_logits_paged

DEVICE = "npu"

NUM_PAGES = 32
SUMMARIES_PER_PAGE = 16
PROXY_DIM = 256
GROUPS = 1
HEADS = 16


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
    lens = [5, 3, 7]
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


def test_prefill_single_batch_q16384_kv65536():
    """Single request, 16K queries, 64K original KV tokens (8192 summaries).

    Output occupies 2 GiB. Check every score using a tiled Torch reference,
    without allocating a [query, region, proxy_dim] gather or a full reference
    output. Causal candidate restriction belongs to the subsequent top-k stage.
    """
    torch.manual_seed(4)
    total_q, kv_tokens, region_size = 16384, 65536, 8
    width = kv_tokens // region_size
    num_pages = width // SUMMARIES_PER_PAGE
    assert kv_tokens % region_size == 0
    assert width % SUMMARIES_PER_PAGE == 0
    q, w = _make_qw(total_q)
    mean_cache = torch.randn(
        num_pages, SUMMARIES_PER_PAGE, 1, PROXY_DIM,
        device=DEVICE, dtype=torch.bfloat16,
    ).to(torch.float8_e4m3fn)
    bt = torch.randperm(num_pages, device=DEVICE).to(torch.int32).reshape(1, num_pages)
    token_to_req = torch.zeros(total_q, device=DEVICE, dtype=torch.int32)
    out = torch.empty(GROUPS * total_q, width, device=DEVICE, dtype=torch.float32)
    indexer_logits_paged(
        q, w, mean_cache, bt, token_to_req,
        out=out, summaries_per_page=SUMMARIES_PER_PAGE
    )
    torch.npu.synchronize()
    print(
        f"single-batch prefill: q={tuple(q.shape)}, w={tuple(w.shape)}, "
        f"cache={tuple(mean_cache.shape)}, block_table={tuple(bt.shape)}, "
        f"token_to_req={tuple(token_to_req.shape)}, out={tuple(out.shape)}"
    )

    logical_summaries = mean_cache.float()[bt[0].long()].reshape(width, PROXY_DIM)
    q_fp8 = q.to(torch.float8_e4m3fn)
    query_chunk, region_chunk = 128, 1024
    for q_begin in range(0, total_q, query_chunk):
        q_end = min(q_begin + query_chunk, total_q)
        q_ref = q_fp8[q_begin:q_end].float()
        w_ref = w[q_begin:q_end]
        for r_begin in range(0, width, region_chunk):
            r_end = min(r_begin + region_chunk, width)
            keys = logical_summaries[r_begin:r_end]
            dots = torch.matmul(q_ref, keys.T)
            expected = (dots.relu() * w_ref[..., None]).sum(dim=2)
            for group in range(GROUPS):
                actual = out[group * total_q + q_begin:group * total_q + q_end,
                             r_begin:r_end]
                torch.testing.assert_close(
                    actual, expected[:, group], rtol=1e-4, atol=1e-4,
                    msg=(f"group={group}, queries=[{q_begin},{q_end}), "
                         f"regions=[{r_begin},{r_end})"),
                )
