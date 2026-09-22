"""Reference implementations of the DSA stages.

These are the fp32 specification leg of the parity harness: each function is a
plain-PyTorch port of the operator's upstream unit-test reference, the definition
of "correct" that every gate compares an implementation against.

The references were validated at step3p5 geometry with ``rtol=1e-3, atol=5e-2``;
the only change made here is removing the module-level step3p5 constants so the
same functions can run step4 geometry.

The e4m3 round-trips are not incidental: the shipping checkpoint stores indexer
weights in bf16, but the deployed kernel quantizes the *activations* to
e4m3 before the score matmul. Dropping the round-trip changes which regions get
selected, so it is part of the semantics we have to reproduce.
"""

from __future__ import annotations

import torch

# Prefill packs a selection entry as ``phys_region | (valid_tokens << 24)``.
PREFILL_REGION_VALID_SHIFT = 24
PREFILL_REGION_ID_MASK = (1 << PREFILL_REGION_VALID_SHIFT) - 1

# Decode packs ``start_token | (phys_region << 32)`` in int64 and carries no
# ``valid_tokens`` -- the visible length comes from ``kv_seqlens`` at read time.
DECODE_PHYS_SHIFT = 32
DECODE_LOW_MASK = (1 << DECODE_PHYS_SHIFT) - 1


def ref_indexer_logits(
    index_q: torch.Tensor,
    weights: torch.Tensor,
    index_k: torch.Tensor,
) -> torch.Tensor:
    """Weighted-ReLU indexer scores.

    Args:
        index_q: ``[seq_q, groups, heads_per_group, proxy_dim]``
        weights: ``[seq_q, groups, heads_per_group]``, already carrying the
            ``heads_per_group ** -0.5`` prescale.
        index_k: ``[seq_k, 1, proxy_dim]`` -- one shared key head (MQA).

    Returns:
        ``[groups * seq_q, seq_k]`` float32, group-major then query.
    """
    seq_q, groups = int(index_q.shape[0]), int(index_q.shape[1])
    seq_k = int(index_k.shape[0])
    q_f = index_q.to(torch.float8_e4m3fn).float()
    k_f = index_k.to(torch.float8_e4m3fn).float()
    scores = torch.einsum("sghd,rd->sghr", q_f, k_f[:, 0])
    scores = torch.relu(scores) * weights.float().unsqueeze(-1)
    return scores.sum(dim=2).permute(1, 0, 2).reshape(groups * seq_q, seq_k)


def ref_topk_select(
    logits: torch.Tensor, lengths: torch.Tensor, *, topk: int
) -> torch.Tensor:
    """Select logical region ids per row: ``torch.topk`` with ties by ascending id.

    Order the visible scores by ``(-score, id)`` and take the first ``min(topk, visible)``.
    A row always fills exactly that many slots -- ties never cause under-selection, so the
    output length is a function of the geometry alone. Verified on H200 across seven cases,
    including a tied group straddling the cut (a prefix of the group is taken) and an
    all-tied row (the lowest ``topk`` ids win).

    Ties are not hypothetical here: post-ReLU indexer scores are exactly zero about
    6.25% of the time (4 heads per group, all four negative), so the group at the cut is
    usually the zero group and the tie-break decides which zero-scored regions get
    attended to.

    Output order is score-descending with ties by ascending id. That is a deterministic
    choice, not a contract: a kernel may emit in radix-bucket order, which varies with the
    input distribution, so callers must be order-insensitive.
    """
    rows, _ = logits.shape
    out = torch.full((rows, int(topk)), -1, dtype=torch.int32, device=logits.device)

    for row in range(rows):
        visible = int(lengths[row])
        if visible <= 0:
            continue
        values = logits[row, :visible].float()
        budget = min(int(topk), visible)
        # Descending sort is stable in torch, so equal scores keep ascending id order.
        chosen = torch.argsort(-values, stable=True)[:budget].to(torch.int32)
        out[row, :budget] = chosen

    return out


def _pack_prefill(
    logical: torch.Tensor,
    q_positions: torch.Tensor,
    block_table: torch.Tensor,
    *,
    region_size: int,
    regions_per_page: int,
) -> torch.Tensor:
    """Pack logical region ids as ``phys_region | (valid_tokens << 24)``, order preserved."""
    safe = logical.clamp_min(0).to(torch.int64)
    page = safe // regions_per_page
    slot_in_page = safe % regions_per_page
    phys = block_table[0].to(torch.int64)[page] * regions_per_page + slot_in_page

    valid_tokens = (
        q_positions.to(torch.int64).view(-1, 1) + 1 - safe * region_size
    ).clamp(0, region_size)
    packed = (phys | (valid_tokens << PREFILL_REGION_VALID_SHIFT)).to(torch.int32)
    return torch.where(logical < 0, torch.full_like(packed, -1), packed).contiguous()


def ref_topk_pack(
    logits: torch.Tensor,
    lengths: torch.Tensor,
    q_positions: torch.Tensor,
    block_table: torch.Tensor,
    *,
    topk: int,
    region_size: int,
    regions_per_page: int,
) -> torch.Tensor:
    """Reference for the fused top-k select + pack operator.

    Measured on H200 2026-08-11: the fused selector does **not** sort its output --
    score-descending order is preserved. A reference that instead sorted the packed
    ids by logical region would agree on the *set* but disagree on order, which reads
    like a packing bug; this reference matches the kernel and does not sort.
    """
    logical = ref_topk_select(logits, lengths, topk=topk)
    return _pack_prefill(
        logical,
        q_positions,
        block_table,
        region_size=region_size,
        regions_per_page=regions_per_page,
    )


def unpack_prefill(packed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split packed prefill meta into (phys_region, valid_tokens), keeping -1."""
    invalid = packed < 0
    phys = torch.where(invalid, packed, packed & PREFILL_REGION_ID_MASK)
    valid_tokens = torch.where(
        invalid, torch.zeros_like(packed), packed >> PREFILL_REGION_VALID_SHIFT
    )
    return phys, valid_tokens


def ref_indexer_norm_rope(
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    index_z: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    k_norm_bias: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    *,
    head_dim: int,
    num_q_heads: int,
    num_k_heads: int,
    rotary_dim: int,
    eps: float = 1e-6,
    q_norm_weight_bias: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize and partially rotate the indexer projections.

    Ported from the indexer norm + partial-RoPE operator's upstream reference.

    The three paths are deliberately asymmetric and it is easy to get wrong:

    * **q** gets *RMSNorm* with a zero-centered weight, ``x / rms * (w + 1)``. The ``+1``
      is ``q_norm_weight_bias`` and is a property of the checkpoint's norm class, not a
      constant -- the vLLM backend passes ``1.0 if q_norm.zero_centered else 0.0``.
    * **k** gets *LayerNorm* -- mean subtracted -- with both a weight and a bias, and no
      zero-centering. This is why the checkpoint has ``k_norm.bias`` but no
      ``q_norm.bias``; treating k as RMSNorm because q is would pass shape checks and
      silently shift every key.
    * **z** is passed through untouched: no norm, no rotation. It is only the softmax
      logits for CSA compression, so normalizing it would rescale the region weights.

    RoPE covers the leading ``2 * rotary_dim`` of ``head_dim`` in half-split (NeoX) form,
    and is applied *after* the norm. ``cos``/``sin`` are ``[seq, rotary_dim]``, i.e. half
    as wide as the rotated span.
    """
    tokens = index_q.shape[0]
    q = index_q.view(tokens, num_q_heads, head_dim).float()
    k = index_k.view(tokens, num_k_heads, head_dim).float()

    q_weight = q_norm_weight.float() + q_norm_weight_bias
    q_rms = torch.sqrt(q.pow(2).mean(dim=-1, keepdim=True) + eps)
    q_normed = (q / q_rms) * q_weight

    k_mean = k.mean(dim=-1, keepdim=True)
    k_variance = (k - k_mean).pow(2).mean(dim=-1, keepdim=True)
    k_normed = (k - k_mean) / torch.sqrt(k_variance + eps) * k_norm_weight.float()
    k_normed = k_normed + k_norm_bias.float()

    index = positions.to(torch.long)
    cos_selected = cos[index].float().unsqueeze(1)
    sin_selected = sin[index].float().unsqueeze(1)

    def rotate(tensor: torch.Tensor) -> torch.Tensor:
        out = tensor.clone()
        real = tensor[..., :rotary_dim]
        imaginary = tensor[..., rotary_dim : 2 * rotary_dim]
        out[..., :rotary_dim] = real * cos_selected - imaginary * sin_selected
        out[..., rotary_dim : 2 * rotary_dim] = real * sin_selected + imaginary * cos_selected
        return out

    return (
        rotate(q_normed).to(index_q.dtype).view(index_q.shape),
        rotate(k_normed).to(index_k.dtype).view(index_k.shape),
        index_z.contiguous(),
    )


def ref_csa_region_summary(
    index_k: torch.Tensor, index_z: torch.Tensor
) -> torch.Tensor:
    """CSA compression of one region's tokens into a single summary vector.

    ``index_k`` and ``index_z`` are ``[tokens, dim]``. The weight is a softmax over
    the token axis of ``index_z``, taken **per dimension**, so the summary is an
    element-wise weighted mean rather than a single convex combination of the key
    vectors:

        s_d = sum_t exp(z_td - m) * k_td / sum_t exp(z_td - m)

    The shift ``m`` is a scalar over both axes, matching the kernel's per-head
    running max. It cancels algebraically; the kernel keeps it because the streaming
    form needs it to stay in range, and the reference keeps it so the two agree bit
    for bit in the last places rather than merely to tolerance.
    """
    keys = index_k.float()
    logits = index_z.float()
    shift = logits.amax()
    weights = torch.exp(logits - shift)
    denominator = weights.sum(dim=0)
    summary = (weights * keys).sum(dim=0) / denominator.clamp_min(1.0e-20)
    return torch.where(denominator > 0, summary, torch.zeros_like(summary))


def quantize_summary_e4m3(mean: torch.Tensor) -> torch.Tensor:
    """FP32 summary -> raw e4m3 bytes, matching the kernel's double rounding.

    The kernel rounds through bfloat16 on the way to e4m3 rather than straight from
    fp32, which is observable: bf16 has 8 mantissa bits and e4m3 has 3, so a value
    sitting on an e4m3 tie can round the other way once bf16 has already moved it.
    Reproduce the two-step rounding, do not shortcut it. Returned as ``uint8`` so the
    stage-1 gate can assert exact byte equality on the FP8 store.
    """
    return mean.to(torch.bfloat16).to(torch.float8_e4m3fn).view(torch.uint8)


def ref_sparse_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    packed_regions: torch.Tensor,
    region_counts: torch.Tensor,
    *,
    num_kv_groups: int,
    region_size: int,
    softmax_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference sparse attention, ported from the sparse-attention forward unit test.

    Lifted out of its per-group loop so one call covers all groups.

    Two things this makes explicit that the kernel relies on:

    * the gathered token list is **deduplicated** before the softmax, so overlapping
      regions contribute once. Our selector never emits duplicates, so kernel and
      reference agree without the kernel needing a dedup pass -- but that is a
      precondition, not a property of the math.
    * there is **no causal mask** here. ``valid_tokens`` in the packed metadata is the
      only thing bounding what a query sees.
    """
    total_q, num_heads, head_dim = query.shape
    heads_per_group = num_heads // num_kv_groups
    out = torch.zeros_like(query)
    lse = torch.full(
        (total_q, num_heads), float("-inf"), device=query.device, dtype=torch.float32
    )

    for group in range(num_kv_groups):
        keys = key_cache[:, group].float()
        values = value_cache[:, group].float()
        head_slice = slice(group * heads_per_group, (group + 1) * heads_per_group)
        for q_index in range(total_q):
            row = group * total_q + q_index
            spans = []
            for slot in range(int(region_counts[row])):
                meta = int(packed_regions[row, slot])
                if meta < 0:
                    continue
                physical = meta & PREFILL_REGION_ID_MASK
                valid = meta >> PREFILL_REGION_VALID_SHIFT
                if valid <= 0:
                    continue
                start = physical * region_size
                spans.append(
                    torch.arange(start, start + valid, device=query.device, dtype=torch.long)
                )
            if not spans:
                continue
            tokens = torch.cat(spans).unique(sorted=True)
            scores = query[q_index, head_slice].float() @ keys[tokens].T * softmax_scale
            out[q_index, head_slice] = (
                torch.softmax(scores, dim=-1) @ values[tokens]
            ).to(query.dtype)
            lse[q_index, head_slice] = torch.logsumexp(scores, dim=-1)
    return out, lse


def ref_sparse_attention_decode(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    packed_regions: torch.Tensor,
    region_counts: torch.Tensor,
    kv_seqlens: torch.Tensor,
    *,
    num_kv_groups: int,
    region_size: int,
    softmax_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference sparse decode, ported from the decode unit test's reference.

    The paged ``[pages, page_size, 1, dim]`` cache is flattened to
    ``[tokens, groups, dim]`` and the single KV head generalised to
    ``num_kv_groups``. Upstream's
    ``phys_page = phys_region >> 1`` / ``phys_half = phys_region & 1`` split is exactly
    ``phys_region * region_size`` in the flattened space, which is why flattening is safe.

    Two contract differences from prefill, both load-bearing:

    * ``valid_tokens`` is **derived**, ``clamp(kv_seqlen - start_token, 0, region_size)``,
      not read out of the packed word. This is what makes the ``-1`` and
      ``0x7FFF_FFFF_FFFF_FFFF`` padding sentinels self-neutralising rather than special
      cases: both leave ``0xFFFF_FFFF`` in the low word.
    * the gathered spans are concatenated with **no dedup**. Prefill's reference calls
      ``.unique()``; this one does not, so a duplicated region really would be counted
      twice. Distinctness is a precondition of the selector, not of the kernel.
    """
    num_reqs, num_heads, _ = query.shape
    heads_per_group = num_heads // num_kv_groups
    out = torch.zeros_like(query)
    lse = torch.full(
        (num_reqs, num_heads), float("-inf"), device=query.device, dtype=torch.float32
    )

    for group in range(num_kv_groups):
        keys = key_cache[:, group].float()
        values = value_cache[:, group].float()
        head_slice = slice(group * heads_per_group, (group + 1) * heads_per_group)
        for request in range(num_reqs):
            row = group * num_reqs + request
            kv_len = int(kv_seqlens[request])
            spans = []
            for slot in range(int(region_counts[row])):
                meta = int(packed_regions[row, slot])
                start_token = meta & DECODE_LOW_MASK
                physical = (meta >> DECODE_PHYS_SHIFT) & DECODE_LOW_MASK
                valid = max(0, min(region_size, kv_len - start_token))
                if valid <= 0:
                    continue
                start = physical * region_size
                spans.append(
                    torch.arange(start, start + valid, device=query.device, dtype=torch.long)
                )
            if not spans:
                continue
            tokens = torch.cat(spans)
            scores = query[request, head_slice].float() @ keys[tokens].T * softmax_scale
            out[request, head_slice] = (
                torch.softmax(scores, dim=-1) @ values[tokens]
            ).to(query.dtype)
            lse[request, head_slice] = torch.logsumexp(scores, dim=-1)
    return out, lse


def ref_prefill_sparse_meta(
    logical: torch.Tensor,
    q_positions: torch.Tensor,
    block_table: torch.Tensor,
    request_indices: torch.Tensor | None = None,
    *,
    topk: int,
    region_size: int,
    regions_per_page: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference for prefill metadata construction (stage 4).

    Source: NV's ``_step3p5_prefill_tile_pack_kernel`` in ``step3p5_dsa_attn.py``, which
    consumes the selector's raw output and produces the ``topk + 1`` window. Two things it
    does that a plainer implementation would not:

    * the history is sorted by **logical** region id and its ``-1`` slots stay *in place*,
      because ``region_counts`` is positional (``min(current_region, topk) + 1``) rather
      than a count of live entries;
    * the current region is stored at ``history_count`` with an *unmasked* write, so the
      history store has to be masked -- otherwise its invalid tail overwrites the appended
      slot with ``-1``. The upstream kernel carries a comment saying exactly this, which is
      a good sign it was found the hard way.
    """
    rows = int(logical.shape[0])
    device = logical.device
    packed = torch.full((rows, int(topk) + 1), -1, dtype=torch.int32, device=device)
    counts = torch.empty((rows,), dtype=torch.int32, device=device)
    tables = block_table.tolist()
    requests = (
        [0] * rows if request_indices is None else [int(v) for v in request_indices.tolist()]
    )

    for row in range(rows):
        table = tables[requests[row] if len(tables) > 1 else 0]
        position = int(q_positions[row])
        current_region = max(position // region_size, 0)
        history_count = min(current_region, int(topk))
        history = sorted(
            int(region) for region in logical[row, :history_count].tolist() if region >= 0
        )
        for slot, region in enumerate(history):
            page = table[region // regions_per_page]
            if page < 0:
                continue
            physical = page * regions_per_page + region % regions_per_page
            valid = min(max(position + 1 - region * region_size, 0), region_size)
            packed[row, slot] = physical | (valid << PREFILL_REGION_VALID_SHIFT)
        page = table[current_region // regions_per_page]
        if page >= 0:
            physical = page * regions_per_page + current_region % regions_per_page
            valid = min(max(position + 1 - current_region * region_size, 0), region_size)
            packed[row, history_count] = physical | (valid << PREFILL_REGION_VALID_SHIFT)
        counts[row] = history_count + 1
    return packed, counts


def ref_decode_sparse_meta(
    logical: torch.Tensor,
    kv_seqlens: torch.Tensor,
    block_table: torch.Tensor,
    request_indices: torch.Tensor | None = None,
    *,
    topk: int,
    region_size: int,
    regions_per_page: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference for decode metadata construction (stage 4).

    Source: ``tests/test_sparse_gqa_decode_meta_step3p5.py::_reference_decode_meta`` with
    ``window=0``, generalised from its hardcoded two-regions-per-page. Upstream's
    per-request ``req_offsets`` subtraction is dropped: it exists because the score buffer
    is one flat concatenation over requests, and our ``logits`` are already request-local.

    The conventions differ from prefill in every detail that could plausibly have been
    shared, so they are worth restating: entries are **compacted** (a dropped region does
    not leave a hole), sorted by the **packed word** -- which is physical-major, since the
    physical region sits in the high 32 bits -- the current region goes last and is *not*
    re-sorted into place, and the tail is padded with **zero**. Zero is a structurally
    valid word: physical region 0, start token 0, a full complement of tokens. Only
    ``region_counts`` says it is padding.
    """
    rows = int(logical.shape[0])
    num_reqs = int(kv_seqlens.shape[0])
    device = logical.device
    packed = torch.zeros((rows, int(topk) + 1), dtype=torch.int64, device=device)
    counts = torch.empty((rows,), dtype=torch.int32, device=device)
    tables = block_table.tolist()
    requests = (
        [row % num_reqs for row in range(rows)]
        if request_indices is None
        else [int(v) for v in request_indices.tolist()]
    )

    def pack(table: list[int], region: int) -> int:
        page = table[region // regions_per_page]
        if page < 0:
            return -1
        physical = page * regions_per_page + region % regions_per_page
        return (region * region_size) | (physical << DECODE_PHYS_SHIFT)

    for row in range(rows):
        table = tables[requests[row] if len(tables) > 1 else 0]
        kv_len = int(kv_seqlens[requests[row]])
        valid_regions = -(-kv_len // region_size)
        entries = []
        for region in logical[row].tolist():
            region = int(region)
            if region < 0 or region >= valid_regions:
                continue
            word = pack(table, region)
            if word >= 0:
                entries.append(word)
        entries.sort()
        current_region = max((kv_len - 1) // region_size, 0)
        if current_region < valid_regions:
            word = pack(table, current_region)
            if word >= 0:
                entries.append(word)
        counts[row] = len(entries)
        if entries:
            packed[row, : len(entries)] = torch.tensor(
                entries, dtype=torch.int64, device=device
            )
    return packed, counts


def ref_merge_split_states(
    partial_out: torch.Tensor, partial_lse: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference for the split-KV state merge.

    Source: ``inline_case_utils.reference_merge_variable_split_nat_lse_states_sm90_gqa``,
    with the variable split plan replaced by a uniform leading axis. The upstream form
    weights by ``softmax(partial_lse)`` over the split axis; that is only well defined when
    at least one split is finite, so the all-empty column is handled here explicitly
    instead of being left to produce ``0/0``.
    """
    lse = torch.logsumexp(partial_lse.float(), dim=0)
    weights = torch.softmax(partial_lse.float(), dim=0)
    merged = (partial_out.float() * weights.unsqueeze(-1)).sum(dim=0)
    empty = torch.isneginf(lse)
    merged = torch.where(empty.unsqueeze(-1), torch.zeros_like(merged), merged)
    return merged.to(partial_out.dtype), lse
