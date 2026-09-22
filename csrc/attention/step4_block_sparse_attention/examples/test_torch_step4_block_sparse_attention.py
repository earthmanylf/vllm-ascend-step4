#!/usr/bin/env python3
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Call step4_block_sparse_attention via torch_extension and check vs FP32 CPU golden."""

from __future__ import annotations

import math
import os
import random
import sys
from dataclasses import dataclass, replace
from typing import List, Optional, Tuple

import torch
import torch_npu  # noqa: F401

REGION_ID_MASK = 0x00FFFFFF
VALID_SHIFT = 24
PREFILL_SKIP = 0xFFFFFFFF  # int32 -1; skip is this sentinel only
DECODE_TAIL_SENTINEL = 0x7FFFFFFFFFFFFFFF
O_RTOL, O_ATOL = 1e-2, 4e-3
LSE_RTOL, LSE_ATOL = 1e-3, 1e-3
# BF16 can hold ~1e38; garbage V/O often shows up as huge finite, not IEEE Inf.
HUGE_ABS = 1e10


def tag_matches(tag: str, filt: str) -> bool:
    """Path filter: exact, prefix (`scene` → `scene/...`), or suffix (`causal_chunk512`).

    `fp16/...` only matches when the filter itself starts with `fp16`, so
    CASE_TAG=scene/causal_chunk512 no longer also pulls fp16/scene/causal_chunk512.
    """
    if not filt:
        return True
    hit = tag == filt or tag.startswith(filt + "/") or tag.endswith("/" + filt)
    if not hit:
        return False
    if tag.startswith("fp16/") and not (filt == "fp16" or filt.startswith("fp16/")):
        return False
    return True


def encode_prefill_i32(phys: int, valid: int) -> int:
    u = (int(phys) & REGION_ID_MASK) | ((int(valid) & 0xFF) << VALID_SHIFT)
    return u - (1 << 32) if u >= (1 << 31) else u


def parse_prefill_slot(meta: int) -> Tuple[int, int]:
    u = int(meta) & 0xFFFFFFFF
    if u == PREFILL_SKIP:
        return 0, 0
    return u & REGION_ID_MASK, u >> VALID_SHIFT


def _load_op():
    try:
        from cann_ops_transformer_custom.ops import step4_block_sparse_attention
        return step4_block_sparse_attention
    except ImportError:
        from cann_ops_transformer.ops import step4_block_sparse_attention
        return step4_block_sparse_attention


step4_block_sparse_attention = _load_op()


@dataclass
class Case:
    q_t: int
    kv_t: int
    n1: int
    g: int
    d: int
    region_size: int
    top_k: int
    # Prefill: only the last live packed slot may be short (1..region_size). 0 = all full.
    tail_valid: int = 0
    empty_first: bool = False
    n_live_regions: int = 0
    sparse_mode: int = 0
    kv_seqlen: int = 0
    skip_stride: int = 1
    vary_seqlen: bool = False
    fill_mode: int = 0
    reverse_slots: bool = False
    inline_sentinels: bool = False
    overlap: int = 0
    start_offset: int = 0
    empty_last: bool = False
    empty_all: bool = False
    mid_skip: bool = False
    vary_live: bool = False
    scale: Optional[float] = None
    return_lse: bool = True
    dtype: torch.dtype = torch.bfloat16
    tag: str = ""
    # Scene packing (named, not a GBSA-style shape sweep):
    # causal: each query only sees regions up to its own position (chunk at end of cache).
    # per_group: the 4 KV groups pick shifted region sets.
    # hash_select: scatter 513 phys inside the visible window (seeded, not one shared table).
    # dense_prefix: phys 0..512 consecutive on a long cache (gather coalescing).
    # seqlen_list: decode TND pack with per-query kv_seqlens.
    causal: bool = False
    per_group: bool = False
    hash_select: bool = False
    dense_prefix: bool = False
    pack_seed: int = 0
    seqlen_list: Optional[Tuple[int, ...]] = None
    page_size: int = 128


def fill_low_prec(n: int, scale: float, offset: float, dtype: torch.dtype) -> torch.Tensor:
    idx = torch.arange(n, dtype=torch.float32)
    return (torch.sin(idx * scale + offset) * 0.25).to(dtype)


def fill_low_prec_randn(n: int, seed: int, dtype: torch.dtype) -> torch.Tensor:
    gen = torch.Generator()
    gen.manual_seed(seed)
    return (torch.rand(n, generator=gen) * 2.0 - 1.0).to(dtype)


def fill_inputs(cs: Case) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q_n = cs.q_t * cs.n1 * cs.d
    kv_n = cs.kv_t * cs.g * cs.d
    if cs.fill_mode == 1:
        q = fill_low_prec_randn(q_n, 1, cs.dtype).view(cs.q_t, cs.n1, cs.d)
        k = fill_low_prec_randn(kv_n, 2, cs.dtype).view(cs.kv_t, cs.g, cs.d)
        v = fill_low_prec_randn(kv_n, 3, cs.dtype).view(cs.kv_t, cs.g, cs.d)
    elif cs.fill_mode == 2:
        q = fill_low_prec(q_n, 0.017, 0.1, cs.dtype).view(cs.q_t, cs.n1, cs.d)
        k = torch.full((cs.kv_t, cs.g, cs.d), 1.0).to(cs.dtype)
        v = fill_low_prec(kv_n, 0.011, 0.7, cs.dtype).view(cs.kv_t, cs.g, cs.d)
    else:
        q = fill_low_prec(q_n, 0.017, 0.1, cs.dtype).view(cs.q_t, cs.n1, cs.d)
        k = fill_low_prec(kv_n, 0.013, 0.3, cs.dtype).view(cs.kv_t, cs.g, cs.d)
        v = fill_low_prec(kv_n, 0.011, 0.7, cs.dtype).view(cs.kv_t, cs.g, cs.d)
    return q, k, v


def _fill_packed_row(cs: Case, packed_i32: torch.Tensor, packed_i64: torch.Tensor, row: int,
                     live: int, n_region: int, is_decode: bool) -> int:
    n_slot = 0
    inserted_mid = False
    for i in range(live):
        if cs.skip_stride > 1:
            r = i * cs.skip_stride
        elif cs.dense_prefix:
            r = i
        elif live < n_region:
            r = i * n_region // live
        else:
            r = i
        if r >= n_region:
            break
        if cs.mid_skip and (not inserted_mid) and i == max(live // 2, 1) and n_slot + 1 < cs.top_k:
            if is_decode:
                packed_i64[row, n_slot] = DECODE_TAIL_SENTINEL
            else:
                packed_i32[row, n_slot] = -1
            n_slot += 1
            inserted_mid = True
        repeats = 1 + (cs.overlap if i == 0 else 0)
        for _ in range(repeats):
            if n_slot >= cs.top_k:
                break
            if is_decode:
                start = r * cs.region_size + cs.start_offset
                packed_i64[row, n_slot] = int(start) | (int(r) << 32)
            else:
                packed_i32[row, n_slot] = encode_prefill_i32(r, cs.region_size)
            n_slot += 1
    if cs.inline_sentinels and is_decode and n_slot + 2 <= cs.top_k:
        packed_i64[row, n_slot] = -1
        n_slot += 1
        packed_i64[row, n_slot] = DECODE_TAIL_SENTINEL
        n_slot += 1
    if cs.reverse_slots and n_slot > 1:
        if is_decode:
            packed_i64[row, :n_slot] = packed_i64[row, :n_slot].flip(0)
        else:
            packed_i32[row, :n_slot] = packed_i32[row, :n_slot].flip(0)
    if (not is_decode) and cs.tail_valid > 0:
        _set_row_last_valid(packed_i32, row, n_slot, cs.tail_valid, cs.region_size)
    return n_slot


def _decode_slot_visible(word: int, region_size: int, kv_seqlen: int) -> int:
    """Visible tokens for one Decode INT64 slot; 0 means drop (out of range / sentinel)."""
    start = int(word) & 0xFFFFFFFF
    return max(0, min(region_size, int(kv_seqlen) - start))


def _compact_decode_row(packed_i64: torch.Tensor, row: int, n_slot: int, region_size: int,
                        kv_seqlen: int) -> int:
    """Production Decode meta: compacted live slots only. History full-8; only last may be partial.

    Drops start>=kv_seqlen (valid==0) and sentinel words so region_counts matches the contract
    used by CompactedPackedTokenCount O(1). Then sorts by start ascending so a hash/per_group
    scatter cannot leave a partial region in the middle of the list.
    """
    live_words = []
    for slot in range(n_slot):
        word = int(packed_i64[row, slot].item())
        if _decode_slot_visible(word, region_size, kv_seqlen) == 0:
            continue
        live_words.append(word)
    live_words.sort(key=lambda w: w & 0xFFFFFFFF)
    out = len(live_words)
    for i, word in enumerate(live_words):
        packed_i64[row, i] = word
    if out < packed_i64.shape[1]:
        packed_i64[row, out:].zero_()
    return out


def _decode_visible_n_region(kv_seqlen: int, region_size: int, start_offset: int, n_region_cache: int) -> int:
    """How many region ids can contribute any token under kv_seqlen (start = r*rs + off)."""
    if kv_seqlen <= start_offset or region_size <= 0 or n_region_cache <= 0:
        return 0
    return min(n_region_cache, (int(kv_seqlen) - int(start_offset) - 1) // int(region_size) + 1)


def _set_row_last_valid(packed_i32: torch.Tensor, row: int, n_slot: int, tail_valid: int,
                        region_size: int) -> None:
    if n_slot <= 0 or tail_valid <= 0:
        return
    word = int(packed_i32[row, n_slot - 1].item()) & 0xFFFFFFFF
    if word == PREFILL_SKIP:
        return
    packed_i32[row, n_slot - 1] = encode_prefill_i32(word & REGION_ID_MASK, min(int(tail_valid), region_size))


def _apply_packed_tail_valid(packed: torch.Tensor, counts: torch.Tensor, tail_valid: int,
                             region_size: int) -> None:
    if tail_valid <= 0:
        return
    rows = packed.shape[0]
    cnt = counts.to(torch.int64)
    last = (cnt - 1).clamp(min=0)
    idx = torch.arange(rows)
    word = packed[idx, last].to(torch.int64) & 0xFFFFFFFF
    skip = (word == PREFILL_SKIP) | (cnt <= 0)
    phys = word & REGION_ID_MASK
    tv = min(int(tail_valid), region_size)
    new_u = (phys & REGION_ID_MASK) | (tv << VALID_SHIFT)
    new_i32 = torch.where(new_u >= (1 << 31), new_u - (1 << 32), new_u).to(torch.int32)
    packed[idx, last] = torch.where(skip, packed[idx, last], new_i32)


def _scene_pack(cs: Case) -> bool:
    return bool(cs.causal or cs.per_group or cs.hash_select or cs.seqlen_list is not None)


def _visible_tokens(cs: Case, seqlens: torch.Tensor) -> torch.Tensor:
    if cs.seqlen_list is not None or cs.sparse_mode == 1:
        return seqlens.to(torch.int64)
    if cs.causal:
        history = max(cs.kv_t - cs.q_t, 0)
        return history + torch.arange(cs.q_t, dtype=torch.int64) + 1
    return torch.full((cs.q_t,), cs.kv_t, dtype=torch.int64)


def _select_phys(cs: Case, visible_regions: torch.Tensor, live: int) -> torch.Tensor:
    i = torch.arange(live, dtype=torch.int64)
    q = torch.arange(cs.q_t, dtype=torch.int64)
    vis = visible_regions.clamp(min=1)
    if cs.hash_select:
        phys = (i[None, :] * 4099 + q[:, None] * (cs.pack_seed * 17 + 1) + cs.pack_seed) % vis[:, None]
    else:
        phys = (i[None, :] * vis[:, None]) // live
        phys = torch.minimum(phys, vis[:, None] - 1)
    phys = phys.unsqueeze(0).expand(cs.g, -1, -1).clone()
    if cs.per_group:
        phys = (phys + torch.arange(cs.g, dtype=torch.int64)[:, None, None]) % vis[None, :, None]
    return phys


def pack_scene(cs: Case, kv_seqlen: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    is_decode = cs.sparse_mode == 1
    n_region = max(cs.kv_t // cs.region_size, 1)
    if cs.seqlen_list is not None:
        if len(cs.seqlen_list) != cs.q_t:
            raise ValueError(f"{cs.tag}: seqlen_list length {len(cs.seqlen_list)} != q_t {cs.q_t}")
        seqlens = torch.tensor(cs.seqlen_list, dtype=torch.int32)
    else:
        seqlens = torch.full((cs.q_t,), kv_seqlen, dtype=torch.int32)
    vis_tok = _visible_tokens(cs, seqlens)
    # Decode/Prefill scene: only regions that intersect the visible prefix. No min=1 phantom
    # when seqlen==0 (empty gather). Partial last region is included via ceil.
    if is_decode:
        # Region r contributes iff start=r*rs+off < seqlen.
        spanned = (vis_tok - cs.start_offset).clamp(min=0)
        vis_reg = torch.div(spanned + cs.region_size - 1, cs.region_size, rounding_mode="floor")
        vis_reg = vis_reg.clamp(min=0, max=n_region)
    else:
        vis_reg = torch.div(vis_tok, cs.region_size, rounding_mode="floor").clamp(min=0, max=n_region)
        # Prefill still needs the current partial region when rem>0.
        rem_tok = vis_tok - vis_reg * cs.region_size
        vis_reg = torch.where((rem_tok > 0) & (vis_reg < n_region), vis_reg + 1, vis_reg)
        vis_reg = torch.maximum(vis_reg, (vis_tok > 0).to(vis_reg.dtype))
    live_cap = min(cs.n_live_regions if cs.n_live_regions > 0 else 3, cs.top_k)
    live_q = torch.minimum(vis_reg, torch.tensor(live_cap, dtype=vis_reg.dtype))
    max_live = int(live_q.max().item())
    if max_live <= 0:
        if is_decode:
            packed = torch.zeros((cs.g * cs.q_t, cs.top_k), dtype=torch.int64)
        else:
            packed = torch.full((cs.g * cs.q_t, cs.top_k), -1, dtype=torch.int32)
        counts = torch.zeros(cs.g * cs.q_t, dtype=torch.int32)
        return packed, counts, seqlens
    # _select_phys requires vis>=1 per selected row; zero-live rows stay empty via slot_ok.
    vis_for_sel = vis_reg.clamp(min=1)
    phys = _select_phys(cs, vis_for_sel, max_live)
    slot_ok = torch.arange(max_live, dtype=torch.int64)[None, :] < live_q[:, None]
    phys = torch.where(slot_ok[None, :, :], phys, torch.zeros_like(phys))
    counts = live_q.to(torch.int32).repeat(cs.g)
    if is_decode:
        packed = torch.zeros((cs.g * cs.q_t, cs.top_k), dtype=torch.int64)
        start = phys * cs.region_size + cs.start_offset
        word = start.to(torch.int64) | (phys.to(torch.int64) << 32)
        word = torch.where(slot_ok[None, :, :], word, torch.zeros_like(word))
        for kv_group in range(cs.g):
            packed[kv_group * cs.q_t:(kv_group + 1) * cs.q_t, :max_live] = word[kv_group]
        # Drop any residual start>=seqlen (e.g. start_offset past seq).
        for row in range(packed.shape[0]):
            q_idx = row % cs.q_t
            counts[row] = _compact_decode_row(packed, row, int(counts[row].item()), cs.region_size,
                                             int(seqlens[q_idx].item()))
        return packed, counts, seqlens
    packed = torch.full((cs.g * cs.q_t, cs.top_k), -1, dtype=torch.int32)
    last_reg = (vis_reg - 1).clamp(min=0)
    rem = vis_tok - last_reg * cs.region_size
    rem = rem.clamp(1, cs.region_size)
    valid = torch.full_like(phys, cs.region_size)
    if cs.tail_valid <= 0:
        is_last = phys == last_reg[None, :, None]
        valid = torch.where(is_last, rem[None, :, None].expand_as(phys), valid)
    valid = torch.where(slot_ok[None, :, :], valid, torch.zeros_like(valid))
    word = (phys.to(torch.int64) & REGION_ID_MASK) | (valid.to(torch.int64) << VALID_SHIFT)
    word = torch.where(slot_ok[None, :, :], word, torch.full_like(word, PREFILL_SKIP))
    meta_i32 = word & 0xFFFFFFFF
    meta_i32 = torch.where(meta_i32 >= (1 << 31), meta_i32 - (1 << 32), meta_i32).to(torch.int32)
    for kv_group in range(cs.g):
        packed[kv_group * cs.q_t:(kv_group + 1) * cs.q_t, :max_live] = meta_i32[kv_group]
    if cs.tail_valid > 0:
        _apply_packed_tail_valid(packed, counts, cs.tail_valid, cs.region_size)
    return packed, counts, seqlens


def pack_regions(cs: Case, kv_seqlen: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if _scene_pack(cs):
        return pack_scene(cs, kv_seqlen)
    is_decode = cs.sparse_mode == 1
    rows = cs.g * cs.q_t
    packed_i32 = torch.full((rows, cs.top_k), -1, dtype=torch.int32)
    packed_i64 = torch.zeros((rows, cs.top_k), dtype=torch.int64)
    counts = torch.zeros(rows, dtype=torch.int32)
    seqlens = torch.full((cs.q_t,), kv_seqlen, dtype=torch.int32)
    if cs.vary_seqlen:
        for q_idx in range(cs.q_t):
            seqlens[q_idx] = max(1, kv_seqlen - q_idx * 48)

    n_region = max(cs.kv_t // cs.region_size, 1)
    if cs.empty_all:
        packed = packed_i64 if is_decode else packed_i32
        return packed, counts, seqlens
    # Decode: sample phys only inside the kv_seqlen-visible window so the last live slot
    # can be the partial current region (contract), not a far cache block that later drops.
    n_region_fill = n_region
    if is_decode:
        n_region_fill = _decode_visible_n_region(int(seqlens[0].item()), cs.region_size, cs.start_offset,
                                                 n_region)
        if n_region_fill <= 0:
            packed = packed_i64
            return packed, counts, seqlens
    live0 = min(cs.n_live_regions if cs.n_live_regions > 0 else 3, cs.top_k, n_region_fill)
    n_slot = _fill_packed_row(cs, packed_i32, packed_i64, 0, live0, n_region_fill, is_decode)
    if not cs.vary_live:
        if is_decode:
            packed_i64[:] = packed_i64[0]
        else:
            packed_i32[:] = packed_i32[0]
        counts.fill_(n_slot)
        if is_decode:
            for row in range(rows):
                q_idx = row % cs.q_t
                counts[row] = _compact_decode_row(packed_i64, row, int(counts[row].item()),
                                                 cs.region_size, int(seqlens[q_idx].item()))
        if cs.empty_first or cs.empty_last:
            for kv_group in range(cs.g):
                for q_idx in range(cs.q_t):
                    if (cs.empty_first and q_idx == 0) or (cs.empty_last and q_idx == cs.q_t - 1):
                        row = kv_group * cs.q_t + q_idx
                        counts[row] = 0
                        if is_decode:
                            packed_i64[row].zero_()
                        else:
                            packed_i32[row].fill_(-1)
        packed = packed_i64 if is_decode else packed_i32
        return packed, counts, seqlens
    counts[0] = n_slot
    for kv_group in range(cs.g):
        for q_idx in range(cs.q_t):
            row = kv_group * cs.q_t + q_idx
            if row == 0:
                continue
            n_fill = n_region
            if is_decode:
                n_fill = _decode_visible_n_region(int(seqlens[q_idx].item()), cs.region_size, cs.start_offset,
                                                  n_region)
                if n_fill <= 0:
                    counts[row] = 0
                    packed_i64[row].zero_()
                    continue
            live = max(1, live0 - q_idx - kv_group)
            live = min(live, cs.top_k, n_fill)
            counts[row] = _fill_packed_row(cs, packed_i32, packed_i64, row, live, n_fill, is_decode)
    if is_decode:
        for row in range(rows):
            q_idx = row % cs.q_t
            counts[row] = _compact_decode_row(packed_i64, row, int(counts[row].item()),
                                             cs.region_size, int(seqlens[q_idx].item()))
    packed = packed_i64 if is_decode else packed_i32
    return packed, counts, seqlens


def gather_tokens_prefill(packed: torch.Tensor, counts: torch.Tensor, row: int, region_size: int) -> List[int]:
    tokens: List[int] = []
    count = int(counts[row].item())
    for slot in range(count):
        phys, valid = parse_prefill_slot(int(packed[row, slot].item()))
        if valid == 0:
            continue
        tokens.extend(phys * region_size + j for j in range(valid))
    return tokens


def gather_tokens_decode(packed: torch.Tensor, counts: torch.Tensor, row: int, region_size: int,
                         kv_seqlen: int) -> List[int]:
    tokens: List[int] = []
    count = int(counts[row].item())
    for slot in range(count):
        word = int(packed[row, slot].item()) & ((1 << 64) - 1)
        start = word & 0xFFFFFFFF
        phys = word >> 32
        vis = max(0, min(region_size, kv_seqlen - start))
        tokens.extend(phys * region_size + j for j in range(vis))
    return tokens


def cpu_golden(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, row_tokens: List[List[int]],
               scale: float, device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    q_t, n1, d = q.shape
    g = k.shape[1]
    group_size = n1 // g
    q_f = q.float()
    k_f = k.float()
    v_f = v.float()
    if device is not None:
        q_f = q_f.to(device)
        k_f = k_f.to(device)
        v_f = v_f.to(device)
    out = torch.zeros_like(q_f)
    lse = torch.zeros(q_t, n1, dtype=torch.float32, device=q_f.device)
    for kv_group in range(g):
        heads = slice(kv_group * group_size, (kv_group + 1) * group_size)
        buckets: dict = {}
        for q_idx in range(q_t):
            key = id(row_tokens[kv_group * q_t + q_idx])
            buckets.setdefault(key, []).append(q_idx)
        for key, q_ids in buckets.items():
            tokens = row_tokens[kv_group * q_t + q_ids[0]]
            if not tokens:
                continue
            tok = torch.tensor(tokens, dtype=torch.long, device=q_f.device)
            q_slice = q_f[q_ids][:, heads]
            k_sel = k_f[tok, kv_group]
            v_sel = v_f[tok, kv_group]
            scores = torch.matmul(q_slice, k_sel.transpose(0, 1)) * scale
            lse[q_ids, heads] = torch.logsumexp(scores, dim=-1)
            out[q_ids, heads] = torch.softmax(scores, dim=-1) @ v_sel
    return out.cpu(), lse.cpu()


def packed_to_tok_mask(cs: Case, packed: torch.Tensor, counts: torch.Tensor, seqlens: torch.Tensor,
                       device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    is_decode = cs.sparse_mode == 1
    live = int(counts.max().item())
    rs = cs.region_size
    meta = packed[:, :live].cpu()
    cnt = counts.cpu().to(torch.int64).unsqueeze(-1)
    slot_ok = torch.arange(live, dtype=torch.int64)[None, :] < cnt
    if is_decode:
        word = meta.to(torch.int64) & ((1 << 64) - 1)
        start = word & 0xFFFFFFFF
        phys = torch.bitwise_right_shift(word, 32)
        seq_rows = seqlens.cpu().to(torch.int64)[None, :].expand(cs.g, cs.q_t).reshape(-1, 1)
        vis = (seq_rows - start).clamp(min=0, max=rs)
    else:
        word = meta.to(torch.int64) & 0xFFFFFFFF
        invalid = word == PREFILL_SKIP
        phys = word & REGION_ID_MASK
        vis = torch.bitwise_right_shift(word, VALID_SHIFT).masked_fill(invalid, 0)
    vis = vis.masked_fill(~slot_ok, 0)
    j = torch.arange(rs, dtype=torch.int64)
    tok = phys.unsqueeze(-1) * rs + j
    ok = j < vis.unsqueeze(-1)
    tok = tok.reshape(cs.g, cs.q_t, -1).to(device)
    ok = ok.reshape(cs.g, cs.q_t, -1).to(device)
    return tok, ok


def cpu_golden_masked(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, tok: torch.Tensor, ok: torch.Tensor,
                      scale: float, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    q_t, n1, d = q.shape
    g = k.shape[1]
    group_size = n1 // g
    q_g = q.float().to(device).view(q_t, g, group_size, d).permute(1, 0, 2, 3).contiguous()
    k_f = k.float().to(device)
    v_f = v.float().to(device)
    tok = tok.to(device)
    ok = ok.to(device)
    g_ix = torch.arange(g, device=device)[:, None, None].expand_as(tok)
    k_sel = k_f[tok.clamp(0, k_f.shape[0] - 1), g_ix]
    v_sel = v_f[tok.clamp(0, v_f.shape[0] - 1), g_ix]
    scores = torch.matmul(q_g, k_sel.transpose(-1, -2)) * scale
    scores = scores.masked_fill(~ok.unsqueeze(2), float("-inf"))
    row_has = ok.any(dim=-1)
    lse = torch.logsumexp(scores, dim=-1)
    attn = torch.softmax(scores, dim=-1)
    attn = torch.where(row_has[:, :, None, None], attn, torch.zeros_like(attn))
    lse = torch.where(row_has.unsqueeze(-1), lse, torch.zeros_like(lse))
    out = torch.matmul(attn, v_sel)
    out = out.permute(1, 0, 2, 3).reshape(q_t, n1, d)
    lse = lse.permute(1, 0, 2).reshape(q_t, n1)
    return out.cpu(), lse.cpu()


def report_metrics(name: str, golden: torch.Tensor, got: torch.Tensor) -> None:
    actual_f = got.float().flatten()
    high_f = golden.float().flatten()
    n_nan = int(torch.isnan(actual_f).sum().item())
    n_inf = int(torch.isinf(actual_f).sum().item())
    n_huge = int((actual_f.abs() > HUGE_ABS).sum().item())
    valid = ~(torch.isnan(actual_f) | torch.isnan(high_f) | torch.isinf(actual_f) | torch.isinf(high_f))
    if n_nan or n_inf or n_huge:
        print(f"  [{name}] nan={n_nan} inf={n_inf} huge(>{HUGE_ABS:g})={n_huge} finite={int(valid.sum().item())}",
              flush=True)
    if int(valid.sum().item()) == 0:
        print(f"  [{name}] All elements are NaN/Inf, cannot compare.", flush=True)
        return
    actual_v = actual_f[valid]
    high_v = high_f[valid]
    diff = (actual_v - high_v).abs()
    max_diff = diff.max().item() if diff.numel() else 0.0
    mean_diff = diff.mean().item() if diff.numel() else 0.0
    high_norm = high_v.abs().sum().item()
    l1_rel = diff.sum().item() / (high_norm + 1e-12)
    if actual_v.abs().sum().item() < 1e-12 and high_norm < 1e-12:
        cos_sim = 1.0
    else:
        cos_sim = torch.nn.functional.cosine_similarity(actual_v.unsqueeze(0), high_v.unsqueeze(0)).item()
    print(f"  [{name}] max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}", flush=True)
    print(f"    L1norm(actual - golden)/L1norm(golden) = {l1_rel:.6f}", flush=True)
    print(f"    cosine_similarity(actual, golden) = {cos_sim:.8f}", flush=True)


def _dump_fail_coords(name: str, golden: torch.Tensor, got: torch.Tensor, mask: torch.Tensor,
                      kv_heads: Optional[int] = None) -> None:
    idx = torch.nonzero(mask.flatten(), as_tuple=False).flatten()[:5]
    gflat, bflat = golden.flatten(), got.flatten()
    for i in idx.tolist():
        print(f"    {name} mismatch[{i}] golden={gflat[i].item()} got={bflat[i].item()}", flush=True)
    if name != "attention_out" or got.dim() != 3:
        return
    row = mask.any(dim=-1)
    qh = torch.nonzero(row, as_tuple=False)
    if qh.numel() == 0:
        return
    q_t, n1, _d = got.shape
    kv_heads = int(kv_heads) if kv_heads else 0
    gsz = (n1 // kv_heads) if kv_heads else 0
    qs = qh[:, 0].tolist()
    hs = qh[:, 1].tolist()
    nd = sorted(set(mask.sum(dim=-1)[row].tolist()))
    print(f"    fail (q,h) rows={qh.shape[0]} q={sorted(set(qs))[:24]}"
          f"{'...' if len(set(qs)) > 24 else ''} h={sorted(set(hs))} fail_dims={nd}",
          flush=True)
    if gsz <= 0:
        return
    tasks: dict = {}
    for q, h in zip(qs, hs):
        kv = h // gsz
        loc = h % gsz
        tasks.setdefault((q, kv), []).append(loc)
    evens = sorted({h % gsz for h in hs if (h % gsz) % 2 == 0})
    odds = sorted({h % gsz for h in hs if (h % gsz) % 2 == 1})
    print(f"    map gsz={gsz} kvHeads={kv_heads} tasks={len(tasks)} kv={sorted({k for _, k in tasks})}"
          f" local_even={evens} local_odd={odds}",
          flush=True)
    for (q, kv), locs in sorted(tasks.items())[:6]:
        task_idx = kv * q_t + q
        locs = sorted(set(locs))
        odd_stats = []
        for loc in range(gsz):
            if loc % 2 == 0:
                continue
            h = kv * gsz + loc
            dmax = (got[q, h] - golden[q, h]).abs().max().item()
            amax = got[q, h].abs().max().item()
            odd_stats.append(f"{loc}:fail={int(bool(mask[q, h].any()))}/dmax={dmax:.3g}/|o|={amax:.3g}")
        print(f"    task q={q} kv={kv} idx={task_idx} core24={task_idx % 24} "
              f"fail_local={locs} odd_sib={' | '.join(odd_stats)}",
              flush=True)


def compare(name: str, golden: torch.Tensor, got: torch.Tensor, rtol: float, atol: float, verbose: bool,
            kv_heads: Optional[int] = None) -> bool:
    del verbose
    report_metrics(name, golden, got)
    diff = (golden - got).abs()
    denom = torch.maximum(golden.abs(), got.abs())
    rel = torch.where(denom > 1e-12, diff / denom, diff)
    fail_mask = diff > (atol + rtol * golden.abs())
    fail = int(fail_mask.sum().item())
    max_rel = rel.max().item() if rel.numel() else 0.0
    status = "PASS" if fail == 0 else "FAIL"
    print(f"    rtol={rtol} atol={atol} fail={fail}/{golden.numel()} max_rel={max_rel:.6g} {status}", flush=True)
    if fail:
        _dump_fail_coords(name, golden, got, fail_mask, kv_heads=kv_heads)
    return fail == 0


def effective_scale(cs: Case) -> float:
    if cs.scale is None:
        return 1.0 / math.sqrt(cs.d)
    if cs.scale == 0.0:
        return 1.0 / math.sqrt(cs.d)
    return cs.scale


def run_case(device: torch.device, cs: Case, op, verbose: bool,
             qkv: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None) -> bool:
    is_decode = cs.sparse_mode == 1
    kv_seqlen = cs.kv_t
    if is_decode:
        kv_seqlen = cs.kv_seqlen if cs.kv_seqlen >= 0 else cs.kv_t
    scale = effective_scale(cs)
    q, k, v = fill_inputs(cs) if qkv is None else qkv
    packed, counts, seqlens = pack_regions(cs, kv_seqlen)

    use_masked = (_scene_pack(cs) or cs.vary_live or cs.vary_seqlen) and cs.q_t <= 512
    if use_masked:
        tok, ok = packed_to_tok_mask(cs, packed, counts, seqlens, device)
        golden_o, golden_lse = cpu_golden_masked(q, k, v, tok, ok, scale, device)
    else:
        rows = cs.g * cs.q_t
        row_tokens: List[List[int]] = []
        if cs.vary_live or cs.vary_seqlen:
            for row in range(rows):
                row_seq = int(seqlens[row % cs.q_t].item())
                if is_decode:
                    row_tokens.append(gather_tokens_decode(packed, counts, row, cs.region_size, row_seq))
                else:
                    row_tokens.append(gather_tokens_prefill(packed, counts, row, cs.region_size))
        else:
            full_row = 1 if cs.empty_first and rows > 1 else 0
            row_seq = int(seqlens[full_row % cs.q_t].item())
            if is_decode:
                toks_full = gather_tokens_decode(packed, counts, full_row, cs.region_size, row_seq)
            else:
                toks_full = gather_tokens_prefill(packed, counts, full_row, cs.region_size)
            for row in range(rows):
                q_idx = row % cs.q_t
                if cs.empty_all or (cs.empty_first and q_idx == 0) or (cs.empty_last and q_idx == cs.q_t - 1):
                    row_tokens.append([])
                else:
                    row_tokens.append(toks_full)
        golden_o, golden_lse = cpu_golden(q, k, v, row_tokens, scale, device)

    q_npu = q.contiguous().to(device)
    k_npu = k.contiguous().to(device)
    v_npu = v.contiguous().to(device)
    packed_npu = packed.contiguous().to(device)
    counts_npu = counts.contiguous().to(device)
    seqlens_npu = seqlens.contiguous().to(device) if is_decode else None
    op_scale = 0.0 if cs.scale == 0.0 else scale
    attn, lse = op(
        q_npu, k_npu, v_npu, packed_npu, counts_npu,
        num_heads=cs.n1, num_key_value_heads=cs.g,
        kv_seqlens=seqlens_npu,
        region_size=cs.region_size,
        page_size=cs.page_size,
        scale_value=op_scale,
        return_softmax_lse=cs.return_lse,
    )
    torch.npu.synchronize()
    got_o = attn.float().cpu()
    ok_cmp = compare("attention_out", golden_o, got_o, O_RTOL, O_ATOL, verbose, kv_heads=cs.g)
    if cs.return_lse:
        got_lse = lse.float().cpu()
        ok_cmp = compare("softmax_lse", golden_lse, got_lse, LSE_RTOL, LSE_ATOL, verbose) and ok_cmp
    elif lse.numel() != 0:
        print(f"  expected empty lse, got shape={tuple(lse.shape)}", flush=True)
        ok_cmp = False
    return ok_cmp


# Model: D=192, GQA 64/4, region_size=8, page_size=128.
# KV cache / kv_seqlens is long; attention selects topk=513 blocks (~4104 tokens).
MODEL_N1, MODEL_G, MODEL_RS = 64, 4, 8
MODEL_TOPK = 513
MODEL_ATTEND = MODEL_TOPK * MODEL_RS  # 4104 tokens actually attended
MODEL_KV = 32 * 1024  # 32768
MODEL_KV_LONG = 64 * 1024  # 65536


def mk(tag: str, mode: int = 0, q_t: int = 1, n1: int = MODEL_N1, g: int = MODEL_G, rs: int = MODEL_RS,
       live: int = MODEL_TOPK, **kw) -> Case:
    kw.pop("d", None)  # D is fixed at 192; not a generalization axis.
    kv_t = kw.pop("kv_t", None)
    if kv_t is None:
        kv_t = MODEL_KV
    extra = 4 if kw.get("mid_skip") or kw.get("inline_sentinels") or kw.get("overlap") else 2
    top_k = kw.pop("top_k", max(live + extra + kw.get("overlap", 0), 8))
    kv_seqlen = kw.pop("kv_seqlen", kv_t if mode == 1 else 0)
    return Case(q_t, kv_t, n1, g, 192, rs, top_k, n_live_regions=live, sparse_mode=mode, kv_seqlen=kv_seqlen,
                tag=tag, **kw)


def both(tag: str, **kw) -> List[Case]:
    prefill_kw = dict(kw)
    decode_kw = dict(kw)
    if "q_t" not in kw:
        prefill_kw["q_t"] = MODEL_ATTEND
        decode_kw["q_t"] = 1
    return [mk(f"{tag}/prefill", 0, **prefill_kw), mk(f"{tag}/decode", 1, **decode_kw)]


def build_cases() -> List[Case]:
    cases: List[Case] = []
    fp16 = dict(dtype=torch.float16)

    # Prefill: long qT on a long cache, select 513 blocks. Decode: qT=1/MTP, kv_seqlens=32k/64k.
    cases.append(mk("model/prefill_qt4104", 0, q_t=MODEL_ATTEND))
    cases.append(mk("model/prefill_qt8192", 0, q_t=8192, kv_t=MODEL_KV_LONG))
    for q_t in (1, 2, 4):
        cases.append(mk(f"model/decode_qt{q_t}", 1, q_t=q_t))
    cases.append(mk("model/decode_kv64k", 1, kv_t=MODEL_KV_LONG, kv_seqlen=MODEL_KV_LONG))
    cases.append(mk("model/decode_unaligned", 1, kv_t=MODEL_KV + 8, kv_seqlen=MODEL_KV + 1))

    cases.append(mk("fp16/model/prefill_qt4104", 0, q_t=MODEL_ATTEND, **fp16))
    cases.append(mk("fp16/model/decode_qt1", 1, **fp16))
    cases.append(mk("fp16/model/decode_qt4", 1, q_t=4, **fp16))
    cases.append(mk("fp16/model/decode_kv64k", 1, kv_t=MODEL_KV_LONG, kv_seqlen=MODEL_KV_LONG, **fp16))

    cases += both("pack/skip2", skip_stride=2)
    cases += both("pack/rev", reverse_slots=True)
    cases += both("pack/empty_all", empty_all=True)
    cases.append(mk("pack/empty_first/prefill", 0, q_t=MODEL_ATTEND, empty_first=True))
    cases.append(mk("pack/empty_first/decode", 1, q_t=2, empty_first=True))
    cases.append(mk("pack/empty_last/prefill", 0, q_t=MODEL_ATTEND, empty_last=True))
    cases.append(mk("pack/empty_last/decode", 1, q_t=2, empty_last=True))
    cases += both("pack/mid_skip", mid_skip=True)
    cases += both("pack/overlap2", overlap=2)
    cases.append(mk("pack/dec_vary_live", 1, q_t=4, vary_live=True))
    cases.append(mk("pack/prefill_partial", 0, q_t=MODEL_ATTEND, tail_valid=3))
    cases.append(mk("pack/dec_start_off3", 1, start_offset=3))
    cases.append(mk("pack/dec_packed0", 1, live=1))
    cases.append(mk("pack/dec_sentinels", 1, inline_sentinels=True))
    cases.append(mk("pack/dec_vary_seq", 1, q_t=4, vary_seqlen=True))
    cases.append(mk("pack/dec_start_gt_seq", 1, start_offset=64, kv_seqlen=8))

    for seq in (0, 1, 8, 9, MODEL_KV):
        cases.append(mk(f"dec/seq{seq}", 1, kv_seqlen=seq))

    cases += both("pack/dense_prefix", dense_prefix=True)

    # Named scenes: same model geometry, different packed tables / length combos.
    cases.append(mk("scene/causal_chunk64", 0, q_t=64, causal=True, per_group=True))
    cases.append(mk("scene/causal_chunk512", 0, q_t=512, causal=True))
    cases.append(mk("scene/hash_chunk512", 0, q_t=512, causal=True, hash_select=True, pack_seed=7))
    cases.append(mk("scene/hash_decode", 1, hash_select=True, pack_seed=11, per_group=True))
    cases.append(mk("scene/hetero_decode", 1, q_t=8, kv_t=MODEL_KV + 8,
                    seqlen_list=(8192, 16384, 24576, 32768, 8193, 12000, 32769, 4096),
                    kv_seqlen=MODEL_KV + 1, hash_select=True, pack_seed=3, per_group=True))
    cases.append(mk("fp16/scene/causal_chunk64", 0, q_t=64, causal=True, per_group=True, **fp16))
    cases.append(mk("fp16/scene/hetero_decode", 1, q_t=8, kv_t=MODEL_KV + 8,
                    seqlen_list=(8192, 16384, 24576, 32768, 8193, 12000, 32769, 4096),
                    kv_seqlen=MODEL_KV + 1, hash_select=True, pack_seed=3, per_group=True, **fp16))

    cases += both("fill/randn", fill_mode=1)
    cases += both("fill/const_k", fill_mode=2)
    cases += both("fill/scale_half", scale=0.5)
    cases += both("fill/no_lse", return_lse=False)
    cases.append(mk("fp16/fill/prefill_randn", 0, q_t=MODEL_ATTEND, fill_mode=1, **fp16))
    cases.append(mk("fp16/fill/decode_randn", 1, fill_mode=1, **fp16))

    # Extra named cases: keep 64/4 D=192 topk=513; vary qT / seqlen / packed table / dtype.
    cases.append(mk("model/prefill_qt1024", 0, q_t=1024))
    cases.append(mk("model/decode_qt3", 1, q_t=3))
    cases.append(mk("model/decode_qt8", 1, q_t=8))
    cases.append(mk("model/decode_kv64k_qt4", 1, q_t=4, kv_t=MODEL_KV_LONG, kv_seqlen=MODEL_KV_LONG))
    cases.append(mk("fp16/model/decode_qt2", 1, q_t=2, **fp16))
    cases.append(mk("fp16/model/decode_unaligned", 1, kv_t=MODEL_KV + 8, kv_seqlen=MODEL_KV + 1, **fp16))

    cases += both("pack/skip3", skip_stride=3)
    cases.append(mk("pack/dec_start_off1", 1, start_offset=1))
    cases.append(mk("pack/dec_start_off7", 1, start_offset=7))
    cases.append(mk("pack/dec_vary_live_qt8", 1, q_t=8, vary_live=True))
    cases.append(mk("pack/empty_first/decode_qt4", 1, q_t=4, empty_first=True))
    cases.append(mk("fp16/pack/skip2/decode", 1, skip_stride=2, **fp16))
    cases.append(mk("fp16/pack/empty_all/decode", 1, empty_all=True, **fp16))
    cases.append(mk("fp16/pack/dec_packed0", 1, live=1, **fp16))
    cases.append(mk("fp16/pack/dense_prefix/decode", 1, dense_prefix=True, **fp16))

    for seq in (7, 15, 16, 64, 4096, 8192, 16384, 32767):
        cases.append(mk(f"dec/seq{seq}", 1, kv_seqlen=seq))

    cases.append(mk("scene/causal_chunk1", 0, q_t=1, causal=True, per_group=True))
    cases.append(mk("scene/causal_chunk128", 0, q_t=128, causal=True, per_group=True))
    cases.append(mk("scene/causal_chunk256", 0, q_t=256, causal=True))
    cases.append(mk("scene/hash_chunk64", 0, q_t=64, causal=True, hash_select=True, pack_seed=13, per_group=True))
    cases.append(mk("scene/hash_chunk128", 0, q_t=128, causal=True, hash_select=True, pack_seed=5))
    cases.append(mk("scene/hash_decode_qt4", 1, q_t=4, hash_select=True, pack_seed=23, per_group=True))
    cases.append(mk("scene/hash_decode_kv64k", 1, kv_t=MODEL_KV_LONG, kv_seqlen=MODEL_KV_LONG,
                    hash_select=True, pack_seed=29, per_group=True))
    cases.append(mk("scene/hetero_decode_qt4", 1, q_t=4, kv_t=MODEL_KV,
                    seqlen_list=(4096, 8192, 16384, 32768),
                    kv_seqlen=MODEL_KV, hash_select=True, pack_seed=41, per_group=True))
    cases.append(mk("scene/hetero_decode_short", 1, q_t=4, kv_t=MODEL_KV,
                    seqlen_list=(1, 8, 9, 64),
                    kv_seqlen=64, hash_select=True, pack_seed=2, per_group=True))
    cases.append(mk("fp16/scene/causal_chunk512", 0, q_t=512, causal=True, **fp16))
    cases.append(mk("fp16/scene/hash_decode", 1, hash_select=True, pack_seed=11, per_group=True, **fp16))
    cases.append(mk("fp16/scene/hash_chunk64", 0, q_t=64, causal=True, hash_select=True,
                    pack_seed=13, per_group=True, **fp16))

    cases += both("fill/scale_one", scale=1.0)
    cases.append(mk("fp16/fill/const_k/decode", 1, fill_mode=2, **fp16))
    cases.append(mk("fp16/fill/no_lse/decode", 1, return_lse=False, **fp16))

    # Prefill last-slot remainder: prefix slots are full region_size; only the last live slot is short.
    for tv in (1, 2, 4, 5, 6, 7):
        cases.append(mk(f"pack/prefill_tail{tv}", 0, q_t=128, tail_valid=tv))
    cases.append(mk("pack/prefill_tail3_qt1", 0, q_t=1, tail_valid=3))
    cases.append(mk("pack/prefill_tail3_qt16", 0, q_t=16, tail_valid=3))
    cases.append(mk("pack/prefill_tail3_qt512", 0, q_t=512, tail_valid=3))
    # Probes: isolate task-count (topK fixed at 513) from topK/tile-count.
    for qt in (24, 32, 48, 64, 96):
        cases.append(mk(f"pack/probe_qt{qt}", 0, q_t=qt, tail_valid=3))
    for qt in (1, 8, 16, 64):
        cases.append(mk(f"pack/probe_live1_qt{qt}", 0, q_t=qt, live=1))
    for qt in (1, 8, 16):
        cases.append(mk(f"pack/probe_live32_qt{qt}", 0, q_t=qt, live=32, tail_valid=3))
    cases.append(mk("pack/prefill_tail1_live1", 0, q_t=64, live=1, tail_valid=1))
    cases.append(mk("pack/prefill_tail7_live1", 0, q_t=64, live=1, tail_valid=7))
    cases.append(mk("pack/prefill_live1", 0, q_t=64, live=1))
    cases.append(mk("pack/prefill_tail3_live2", 0, q_t=64, live=2, tail_valid=3))
    cases.append(mk("pack/prefill_tail3_live16", 0, q_t=128, live=16, tail_valid=3))
    cases.append(mk("pack/prefill_tail3_live64", 0, q_t=128, live=64, tail_valid=3))
    cases.append(mk("pack/prefill_tail3_skip2", 0, q_t=128, skip_stride=2, tail_valid=3))
    cases.append(mk("pack/prefill_tail3_rev", 0, q_t=128, reverse_slots=True, tail_valid=3))
    cases.append(mk("pack/prefill_tail3_dense", 0, q_t=128, dense_prefix=True, tail_valid=3))
    cases.append(mk("pack/prefill_tail3_vary_live", 0, q_t=8, live=32, vary_live=True, tail_valid=3))
    cases.append(mk("pack/prefill_tail3_kv64k", 0, q_t=128, kv_t=MODEL_KV_LONG, tail_valid=3))
    cases.append(mk("fp16/pack/prefill_tail3", 0, q_t=128, tail_valid=3, **fp16))
    cases.append(mk("fp16/pack/prefill_tail1_live1", 0, q_t=64, live=1, tail_valid=1, **fp16))
    cases.append(mk("scene/causal_tail1_chunk64", 0, q_t=64, causal=True, tail_valid=1))
    cases.append(mk("scene/causal_tail7_chunk64", 0, q_t=64, causal=True, per_group=True, tail_valid=7))
    cases.append(mk("scene/hash_tail3_chunk64", 0, q_t=64, causal=True, hash_select=True,
                    pack_seed=7, tail_valid=3))
    cases.append(mk("scene/hash_tail1_chunk128", 0, q_t=128, causal=True, hash_select=True,
                    pack_seed=17, per_group=True, tail_valid=1))

    return cases


def run_pd_shared(device: torch.device, op, verbose: bool) -> bool:
    pre = mk("scene/pd_shared/prefill", 0, q_t=512, causal=True, per_group=True)
    dec = mk("scene/pd_shared/decode", 1, q_t=1, hash_select=True, pack_seed=19, per_group=True)
    q_p, k, v = fill_inputs(pre)
    print(case_label(-1, pre), flush=True)
    ok1 = run_case(device, pre, op, verbose, qkv=(q_p, k, v))
    print(f"  {'PASS' if ok1 else 'FAIL'}", flush=True)
    q_d = fill_inputs(dec)[0]
    print(case_label(-1, dec), flush=True)
    ok2 = run_case(device, dec, op, verbose, qkv=(q_d, k, v))
    print(f"  {'PASS' if ok2 else 'FAIL'}", flush=True)
    return ok1 and ok2


def case_label(i: int, cs: Case) -> str:
    tag = cs.tag or "case"
    dt = "fp16" if cs.dtype == torch.float16 else "bf16"
    return (f"=== [{i}] {tag} dtype={dt} mode={cs.sparse_mode} qT={cs.q_t} kvT={cs.kv_t} n1={cs.n1} g={cs.g} "
            f"d={cs.d} rs={cs.region_size} live={cs.n_live_regions} kvSeqlen={cs.kv_seqlen} "
            f"fill={cs.fill_mode} ===")


def main() -> int:
    device_id = int(os.environ.get("TEST_DEVICE_ID", "5"))
    torch.npu.set_device(device_id)
    device = torch.device(f"npu:{device_id}")
    all_cases = build_cases()
    case_filter = os.environ.get("CASE_ID")
    tag_filter = os.environ.get("CASE_TAG")
    verbose = os.environ.get("TEST_VERBOSE", "0") == "1"
    n_repeat = int(os.environ.get("CASE_REPEAT", "1"))
    cases = []
    for i, cs in enumerate(all_cases):
        if case_filter is not None and str(i) != case_filter:
            continue
        if tag_filter and not tag_matches(cs.tag or "", tag_filter):
            continue
        cases.append((i, cs))
    print(f"device=npu:{device_id} cases={len(cases)}/{len(all_cases)} repeat={n_repeat}", flush=True)
    failed = 0
    failed_tags: List[str] = []
    for trial in range(n_repeat):
        if n_repeat > 1:
            print(f"######## trial {trial + 1}/{n_repeat} ########", flush=True)
        for i, cs in cases:
            print(case_label(i, cs), flush=True)
            try:
                ok = run_case(device, cs, step4_block_sparse_attention, verbose)
            except Exception as exc:
                print(f"  raised: {exc}", flush=True)
                ok = False
            print(f"  {'PASS' if ok else 'FAIL'}", flush=True)
            if not ok:
                failed += 1
                failed_tags.append(cs.tag or str(i))
        want_pd = case_filter is None and (
            not tag_filter or tag_filter in ("scene", "pd_shared") or tag_filter.startswith("pd_shared")
            or tag_filter.startswith("scene/pd_shared"))
        extra = 0
        if want_pd:
            extra = 1
            try:
                ok_pd = run_pd_shared(device, step4_block_sparse_attention, verbose)
            except Exception as exc:
                print(f"  raised: {exc}", flush=True)
                ok_pd = False
            if not ok_pd:
                failed += 1
                failed_tags.append("scene/pd_shared")
    n_run = (len(cases) + extra) * n_repeat
    print(f"summary: {failed}/{n_run} cases failed", flush=True)
    if failed_tags:
        print("failed tags: " + ", ".join(failed_tags), flush=True)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
