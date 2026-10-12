#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
"""E2E UT (single NPU): fused aclnn IndexerRegionScore adapter.

Covers ``vllm_ascend.models.step5.indexer_region_score.indexer_score_topk``
(``indexer_region_score`` -> ``aclnnIndexerRegionScore``), the fused
score + causal-topk + ascending-sort operator wired into the Step5 DSA
decode/prefill paths.

Three-layer verification strategy:

1. Formula-free invariants (exact): output shapes/dtypes; per-row ids strictly
   ascending, duplicate-free, in ``[0, bound)`` with
   ``bound = floor((actual_kv_lengths[n] - q_len + j) / region_size)``
   (actual_kv_lengths is the effective KV length including the current
   queries; the current region never competes, mask_mode=3); ``-1`` only
   as tail padding with
   exactly ``max(0, topk - bound)`` pads. Rows with ``bound <= topk`` select
   *every* candidate, so their ids must equal ``arange(bound)`` exactly.
2. Self-consistency (formula-free): torch.topk over the operator's own
   scores, masked by the causal bound, must reproduce the operator's ids.
 3. Numeric reference (tolerance): scores vs the pre-fusion Triton kernel
   ``indexer_logits_paged`` (same formula ``sum_h relu(q . k) * w``; the
   adapter passes head_scale=1.0 because model.py already folded H**-0.5
   into the weights) on causally-valid columns.

Production shapes (Step5, TP4): the 4 kv groups are TP-split so G=1 per rank,
H=4 index heads per group, D=256 proxy dim, region_size=8, topk=512, cached
context up to 64k tokens; max_region_len 16384 covers 128k-token contexts
(128 * 1024 / 8). Decode with MTP flattens to 4 rows per request, so row
counts are multiples of 4.
"""

import pytest
import torch
import torch_npu

from vllm_ascend.utils import enable_custom_op

torch_npu.npu.config.allow_internal_format = True
enable_custom_op()

from vllm_ascend.models.step5.dsa_kernels import indexer_logits_paged
from vllm_ascend.models.step5.indexer_region_score import indexer_score_topk

DEVICE = "npu:0"

# Production-derived constants (see module docstring).
G = 1  # kv groups per rank after TP4 splits the model's 4 groups
H = 4  # index heads per group
D = 256  # index proxy dim (operator meta requires [T,G,H,256])
REGION_SIZE = 8
TOPK = 512
# model.py pre-scales the per-head weights by H**-0.5 before the attention
# call, so the adapter calls the operator with head_scale=1.0 and the fused
# scores must match the Triton kernel (sum_h relu(q.k) * w) directly.
HEAD_SCALE = 1.0
SUMMARIES_PER_PAGE = 16  # paged summary cache: block_size 128 / region_size 8
WIDE_REGION_LEN = 16384  # 128k-token context budget / region_size


def _build_inputs(
    actual_kv_lens: list[int],
    tokens_per_req: list[int],
    max_region_len: int,
    seed: int = 0,
):
    """Build a self-consistent input bundle for ``indexer_score_topk``.

    ``actual_kv_lens`` holds effective KV lengths including the current
    queries (``ctx + q_len`` per request). Every block-table entry is
    mapped to a distinct physical page (no -1), so all ``bound``
    causal-candidate regions are valid candidates.
    """
    torch.manual_seed(seed)
    num_reqs = len(actual_kv_lens)
    total_tokens = sum(tokens_per_req)
    pages_per_req = (max_region_len + SUMMARIES_PER_PAGE - 1) // SUMMARIES_PER_PAGE
    num_pages = num_reqs * pages_per_req

    index_q = torch.randn(total_tokens, G, H, D, device=DEVICE, dtype=torch.bfloat16)
    weights = torch.randn(total_tokens, G, H, device=DEVICE, dtype=torch.float32)
    block_table = torch.randperm(num_pages, dtype=torch.int32).reshape(
        num_reqs, pages_per_req).to(DEVICE)
    mean_cache = torch.randn(num_pages, SUMMARIES_PER_PAGE, 1, D,
                             dtype=torch.bfloat16).to(torch.float8_e4m3fn).to(DEVICE)

    cu_seq_lens = torch.zeros(num_reqs + 1, dtype=torch.int32)
    cu_seq_lens[1:] = torch.cumsum(
        torch.tensor(tokens_per_req, dtype=torch.int32), dim=0)
    cu_seq_lens = cu_seq_lens.to(DEVICE)
    actual_kv_lengths = torch.tensor(actual_kv_lens, dtype=torch.int32).to(DEVICE)
    return index_q, weights, mean_cache, block_table, cu_seq_lens, actual_kv_lengths


def _token_bounds(cu_seq_lens, actual_kv_lengths, total_tokens):
    """Causal bound per output row.

    ``actual_kv_lengths`` holds the effective KV length including the
    current queries; the operator restores the first query position as
    ``actual_kv_lengths[n] - q_len``, so token j of request n sits at
    ``floor((actual_kv_lengths[n] - q_len + j) / 8)`` visible regions.
    """
    cu = cu_seq_lens.cpu()
    akl = actual_kv_lengths.cpu()
    bounds = torch.empty(total_tokens, dtype=torch.int64)
    for n in range(akl.numel()):
        q_len = int(cu[n + 1]) - int(cu[n])
        for t in range(int(cu[n]), int(cu[n + 1])):
            j = t - int(cu[n])
            bounds[t] = (int(akl[n]) - q_len + j) // REGION_SIZE
    return bounds


def _assert_id_contract(ids, bounds, topk):
    """Layer-1 invariants: sort order, causal range, -1 tail padding."""
    ids_cpu = ids.cpu()
    assert ids.dtype == torch.int32
    for row in range(ids_cpu.shape[0]):
        bound = int(bounds[row])
        entries = ids_cpu[row]
        num_valid = int((entries != -1).sum())
        # bound candidates (all mapped) capped by topk.
        assert num_valid == min(topk, bound), (
            f"row {row}: {num_valid} valid ids, expected {min(topk, bound)}")
        # -1 only as tail padding.
        assert bool((entries[num_valid:] == -1).all())
        assert bool((entries[:num_valid] != -1).all())
        if num_valid == 0:
            continue
        valid = entries[:num_valid]
        # strictly below the causal bound, strictly ascending (dup-free).
        assert int(valid.max()) < bound
        assert int(valid.min()) >= 0
        assert bool((valid[:-1] < valid[1:]).all())
        if bound <= topk:
            # every candidate region is selected: exact contract.
            assert torch.equal(valid, torch.arange(bound, dtype=torch.int32))


def _masked_topk_ids(scores, bounds, topk):
    """Reference selection: causal-mask ``scores`` then topk + ascending sort.

    ``scores`` is ``[rows, R]``; masked columns are set to -inf before the
    topk, selected sets are emitted ascending with a -1 tail.
    """
    rows, regions = scores.shape
    masked = scores.clone()
    for row in range(rows):
        bound = int(bounds[row])
        if bound < regions:
            masked[row, bound:] = float("-inf")
    k = min(topk, regions)
    vals, idx = torch.topk(masked.float(), k, dim=-1)
    out = torch.full((rows, topk), -1, dtype=torch.int32)
    for row in range(rows):
        keep = idx[row][~torch.isinf(vals[row])]
        keep = torch.sort(keep).values.to(torch.int32)
        out[row, :keep.numel()] = keep
    return out


def _assert_valid_topk_selection(ids, scores, bounds, topk, eps=0.0):
    """Tie-agnostic check: ids must be *a* valid top-k set of ``scores``.

    With ``k = min(topk, bound)`` and ``cutoff`` the k-th largest causally
    valid score: every selected id scores ``>= cutoff - eps`` and every
    unselected valid region scores ``<= cutoff + eps``. Use ``eps=0`` when
    ``scores`` is the very tensor the selector consumed (exact); a small eps
    for cross-implementation comparison.

    Why ties matter here: the scoring formula ``sum_h relu(q.k) * w`` with
    signed weights yields score == 0 for every region whose per-head dots are
    all negative (P = 1/16 per region), and a row whose four head weights are
    all negative has *every* score <= 0, so the top-512 cutoff lands inside a
    large exact-zero tie cluster. Any tie-breaking order is equally correct;
    torch.topk and the fused operator simply pick different tied ids.
    """
    s = scores.cpu() if not isinstance(scores, torch.Tensor) or scores.is_npu else scores
    ids_cpu = ids.cpu()
    for row in range(ids_cpu.shape[0]):
        bound = int(bounds[row])
        k = min(topk, bound)
        entries = ids_cpu[row]
        selected = entries[entries != -1].long()
        assert selected.numel() == k
        if k == 0:
            continue
        row_scores = s[row, :bound].float()
        cutoff = torch.topk(row_scores, k).values[-1]
        assert bool((row_scores[selected] >= cutoff - eps).all()), (
            f"row {row}: selected id below cutoff {cutoff}")
        unselected_mask = torch.ones(bound, dtype=torch.bool)
        unselected_mask[selected] = False
        assert bool((row_scores[unselected_mask] <= cutoff + eps).all()), (
            f"row {row}: unselected region above cutoff {cutoff}")


def _row_overlap(a, b):
    """Mean per-row Jaccard-style overlap of two id sets (ignoring -1)."""
    ratios = []
    for row in range(a.shape[0]):
        sa = set(a[row][a[row] != -1].cpu().tolist())
        sb = set(b[row][b[row] != -1].cpu().tolist())
        union = sa | sb
        ratios.append(len(sa & sb) / len(union) if union else 1.0)
    return sum(ratios) / len(ratios)


def _run_fused(index_q, weights, mean_cache, block_table, cu_seq_lens,
               actual_kv_lengths, max_region_len):
    scores, ids = indexer_score_topk(
        index_q,
        weights,
        mean_cache,
        block_table,
        cu_seq_lens,
        actual_kv_lengths,
        max_region_len=max_region_len,
        topk=TOPK,
        region_size=REGION_SIZE,
    )
    return scores, ids


def _run_triton_reference(index_q, weights, mean_cache, block_table,
                          cu_seq_lens, total_tokens, max_region_len):
    token_to_req = torch.zeros(total_tokens, dtype=torch.int32)
    cu = cu_seq_lens.cpu()
    for n in range(cu.numel() - 1):
        token_to_req[int(cu[n]):int(cu[n + 1])] = n
    ref = torch.empty(G * total_tokens, max_region_len,
                      device=DEVICE, dtype=torch.float32)
    indexer_logits_paged(
        index_q,
        weights,
        mean_cache,
        block_table,
        token_to_req.to(DEVICE),
        out=ref,
        summaries_per_page=SUMMARIES_PER_PAGE,
    )
    return ref


def _valid_column_mask(bounds, regions):
    mask = torch.zeros(len(bounds), regions, dtype=torch.bool)
    for row, bound in enumerate(bounds):
        mask[row, :int(bound)] = True
    return mask


def _check_vs_triton(scores, ids, index_q, weights, mean_cache, block_table,
                     cu_seq_lens, bounds, max_region_len):
    ref = _run_triton_reference(index_q, weights, mean_cache, block_table,
                                cu_seq_lens, scores.shape[0] // G,
                                max_region_len)
    valid = _valid_column_mask(bounds, max_region_len).to(DEVICE)
    # The Triton kernel computes sum_h relu(q.k) * w; the adapter passes
    # head_scale=1.0 (model.py already folded H**-0.5 into the weights), so
    # both must agree on causally-valid columns (the Triton path emits
    # unmasked scores).
    torch.testing.assert_close(
        scores[valid],
        ref[valid] * HEAD_SCALE,
        rtol=5e-2,
        atol=1e-1,
        msg=lambda m: f"fused scores diverge from Triton reference: {m}",
    )
    ref_ids = _masked_topk_ids(ref.cpu(), bounds, TOPK)
    overlap = _row_overlap(ids.cpu(), ref_ids)
    assert overlap >= 0.9, (
        f"fused ids overlap {overlap:.4f} with reference < 0.9")
    # Correctness gate (tie-agnostic): ids are a valid top-k of the
    # reference scores, allowing for the small cross-implementation numeric
    # wiggle (observed max abs score deviation < 1e-4).
    _assert_valid_topk_selection(ids, ref, bounds, TOPK, eps=1e-3)


def test_decode_rows_contract():
    """Decode/MTP form: cu_seq_lens = arange(rows + 1), 1 token per row.

    16 rows (4 requests x 4 MTP tokens) spanning every causal regime:
    bound 0 (all -1), bound < topk (exact arange), bound == topk (no pad),
    bound > topk (selection), and a 64k-context row.
    """
    kv_lens = [1, 6, 9, 17, 64, 65, 101, 4096, 4097, 4101, 8193, 16384,
               32769, 65529, 65536, 65537]
    inputs = _build_inputs(kv_lens, [1] * 16, max_region_len=WIDE_REGION_LEN)
    scores, ids = _run_fused(*inputs, WIDE_REGION_LEN)
    assert scores.shape == (G * 16, WIDE_REGION_LEN)
    assert scores.dtype == torch.float32
    assert ids.shape == (G * 16, TOPK)

    bounds = _token_bounds(inputs[4], inputs[5], 16)
    _assert_id_contract(ids, bounds, TOPK)
    # self-consistency (exact, tie-agnostic): ids are a valid top-k of the
    # operator's own scores.
    _assert_valid_topk_selection(ids, scores, bounds, TOPK)


def test_prefill_rows_contract():
    """Prefill form: multi-token requests driven by query_start_loc.

    Includes a zero-context request (length = q_len, every token's bound
    is 0 -> all -1) and per-token causal drift inside a request (bound
    grows with j). kv_lens are effective KV lengths: ctx + q_len.
    """
    kv_lens = [4, 104, 65536]
    tokens_per_req = [4, 4, 8]
    total = sum(tokens_per_req)
    inputs = _build_inputs(kv_lens, tokens_per_req, max_region_len=WIDE_REGION_LEN)
    scores, ids = _run_fused(*inputs, WIDE_REGION_LEN)
    assert scores.shape == (G * total, WIDE_REGION_LEN)
    assert ids.shape == (G * total, TOPK)

    bounds = _token_bounds(inputs[4], inputs[5], total)
    # request 0 has no history: four all-padding rows.
    assert int((ids[:4] == -1).sum()) == 4 * TOPK
    # request 1: bounds 12,12,13,13 -> exact arange rows.
    _assert_id_contract(ids, bounds, TOPK)
    _assert_valid_topk_selection(ids, scores, bounds, TOPK)


def test_scores_and_ids_vs_triton_reference():
    """Numeric comparison against the pre-fusion Triton scoring kernel."""
    kv_lens = [64, 101, 4096, 4101, 8193, 16384, 65529, 65536]
    inputs = _build_inputs(kv_lens, [1] * 8, max_region_len=WIDE_REGION_LEN,
                           seed=7)
    scores, ids = _run_fused(*inputs, WIDE_REGION_LEN)
    bounds = _token_bounds(inputs[4], inputs[5], 8)
    _assert_id_contract(ids, bounds, TOPK)
    _check_vs_triton(scores, ids, *inputs[:4], inputs[4], bounds,
                     WIDE_REGION_LEN)


def test_prefill_vs_triton_reference():
    """Same comparison for the multi-token (prefill) input form.

    kv_lens are effective KV lengths (ctx + q_len) for ctx = 1000, 65500.
    """
    kv_lens = [1004, 65504]
    tokens_per_req = [4, 4]
    inputs = _build_inputs(kv_lens, tokens_per_req,
                           max_region_len=WIDE_REGION_LEN, seed=11)
    scores, ids = _run_fused(*inputs, WIDE_REGION_LEN)
    bounds = _token_bounds(inputs[4], inputs[5], sum(tokens_per_req))
    _assert_id_contract(ids, bounds, TOPK)
    _check_vs_triton(scores, ids, *inputs[:4], inputs[4], bounds,
                     WIDE_REGION_LEN)


def test_determinism():
    """The fused operator must be bit-stable across repeated calls."""
    inputs = _build_inputs([4101, 65536, 101, 9], [1] * 4,
                           max_region_len=WIDE_REGION_LEN, seed=3)
    scores1, ids1 = _run_fused(*inputs, WIDE_REGION_LEN)
    scores2, ids2 = _run_fused(*inputs, WIDE_REGION_LEN)
    assert torch.equal(scores1, scores2)
    assert torch.equal(ids1, ids2)
