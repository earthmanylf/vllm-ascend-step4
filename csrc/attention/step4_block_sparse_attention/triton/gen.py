"""Deterministic synthetic inputs for the DSA parity harness.

Distributions mirror the upstream unit tests so a failure here is attributable to
our port rather than to input pathology. Indexer weights deliberately remain signed:
production obtains them from an unconstrained linear projection, so applying a softmax
here would erase the negative-score cases the selector must handle. The weights also
carry the production ``heads_per_group ** -0.5`` prescale, which the upstream test
omits (it notes the kernel is scale-agnostic).
"""

from __future__ import annotations

from typing import NamedTuple

import torch

from .geometry import Step4Geometry


class IndexerInputs(NamedTuple):
    index_q: torch.Tensor  # [seq_q, groups, heads_per_group, proxy_dim] bf16
    weights: torch.Tensor  # [seq_q, groups, heads_per_group] bf16
    index_k: torch.Tensor  # [seq_k, 1, proxy_dim] bf16


def make_indexer_inputs(
    geo: Step4Geometry,
    *,
    seq_q: int,
    seq_k: int,
    device: torch.device,
    seed: int,
) -> IndexerInputs:
    generator = torch.Generator(device=device).manual_seed(seed)
    shape_q = (seq_q, geo.num_provider_groups, geo.q_heads_per_group, geo.proxy_dim)

    index_q = torch.randn(
        shape_q, generator=generator, device=device, dtype=torch.bfloat16
    ).contiguous()
    index_k = torch.randn(
        (seq_k, 1, geo.proxy_dim), generator=generator, device=device, dtype=torch.bfloat16
    ).contiguous()
    raw_weights = torch.randn(
        shape_q[:-1], generator=generator, device=device, dtype=torch.float32
    )
    weights = (
        raw_weights.mul_(geo.indexer_weight_prescale)
        .to(torch.bfloat16)
        .contiguous()
    )
    return IndexerInputs(index_q=index_q, weights=weights, index_k=index_k)


class TopkInputs(NamedTuple):
    logits: torch.Tensor  # [groups * total_q, seq_regions] fp32, signed indexer scores
    lengths: torch.Tensor  # [groups * total_q] int32, causally visible region count
    q_positions: torch.Tensor  # [groups * total_q] int32
    block_table: torch.Tensor  # [1, pages] int32
    cu_seq_lens: torch.Tensor  # [2] int32
    total_q: int
    seq_regions: int


def make_topk_inputs(
    geo: Step4Geometry,
    *,
    total_q: int,
    seq_regions: int,
    device: torch.device,
    seed: int,
    tie_block: int = 0,
) -> TopkInputs:
    """Synthetic selector inputs.

    Logits are signed weighted-ReLU scores, matching the unconstrained indexer
    projection used in production. The selector must therefore use a signed ordered
    float key rather than treating the sign bit as metadata to discard.

    ``tie_block`` forces the first ``tie_block`` visible regions of every row to
    an identical score. That is the only way to observe the kernel's tie-break
    rule directly rather than inferring it from the torch reference, and ties are
    not hypothetical here: with 4 heads per group, ~6.25% of real indexer scores
    are exactly zero.
    """
    generator = torch.Generator(device=device).manual_seed(seed)
    groups = geo.num_provider_groups
    rows = groups * total_q

    logits = torch.randn(
        (rows, seq_regions), generator=generator, device=device, dtype=torch.float32
    )
    if tie_block > 0:
        logits[:, :tie_block] = 0.5

    q_pos = torch.arange(total_q, device=device, dtype=torch.int32)
    visible = (q_pos // geo.region_block_size + 1).clamp_max(seq_regions)

    pages = -(-seq_regions // geo.regions_per_page)
    return TopkInputs(
        logits=logits.contiguous(),
        lengths=visible.repeat(groups).contiguous(),
        q_positions=q_pos.repeat(groups).contiguous(),
        block_table=torch.arange(pages, device=device, dtype=torch.int32)
        .view(1, pages)
        .contiguous(),
        cu_seq_lens=torch.tensor([0, total_q], device=device, dtype=torch.int32),
        total_q=total_q,
        seq_regions=seq_regions,
    )


def make_banded_tie_inputs(
    geo: Step4Geometry,
    *,
    total_q: int,
    seq_regions: int,
    device: torch.device,
    seed: int,
    high_count: int,
    tied_count: int,
    constant_visible: int | None = None,
) -> TopkInputs:
    """Inputs with three score bands: distinct-high, exactly-tied, distinct-low.

    ``make_topk_inputs(tie_block=...)`` ties *every* visible region, which
    degenerates the distribution and exercises the selector's histogram pruning
    rather than its boundary behaviour. Banding lets the tied group be placed
    either straddling the top-k cut or strictly inside it, which is what
    distinguishes "drops the group sitting at the threshold" from "drops any tied
    group at all".

    ``constant_visible`` overrides the causal region count. Causality realism is
    irrelevant when probing selector semantics, and a fixed count keeps every row
    in the same regime so one disagreeing row characterises the whole case.
    """
    base = make_topk_inputs(
        geo, total_q=total_q, seq_regions=seq_regions, device=device, seed=seed
    )
    lengths = base.lengths
    if constant_visible is not None:
        lengths = torch.full_like(lengths, min(constant_visible, seq_regions))

    logits = torch.zeros_like(base.logits)
    generator = torch.Generator(device=device).manual_seed(seed + 1)
    for row in range(logits.shape[0]):
        visible = int(lengths[row])
        high = min(high_count, visible)
        logits[row, :high] = torch.linspace(
            0.9, 0.7, high, device=device, dtype=torch.float32
        )
        tie_end = min(high + tied_count, visible)
        logits[row, high:tie_end] = 0.5
        remaining = visible - tie_end
        if remaining > 0:
            logits[row, tie_end:visible] = (
                torch.rand(remaining, generator=generator, device=device) * 0.3 + 0.05
            )

    return base._replace(logits=logits.contiguous(), lengths=lengths.contiguous())


class CsaMetadata(NamedTuple):
    flat_slot: torch.Tensor  # [rows] int64, global logical region id, -1 = skip
    reset_slots: torch.Tensor  # [rows] int64, region id if this row starts it
    token_valid: torch.Tensor  # [rows] bool
    token_positions: torch.Tensor  # [rows] int64, position within its own request
    query_start_loc: torch.Tensor  # [B+1] int32
    seq_lens: torch.Tensor  # [B] int32
    task_prefix: torch.Tensor  # [B+1] int32, kernel-filled scratch
    index_k: torch.Tensor  # [rows, 1, proxy_dim]
    index_z: torch.Tensor  # [rows, 1, proxy_dim]
    region_block_size: int
    regions_per_request: int


def make_csa_prefill_inputs(
    geo: Step4Geometry,
    *,
    seq_lens: tuple[int, ...],
    regions_per_request: int,
    device: torch.device,
    seed: int,
    dtype: torch.dtype = torch.bfloat16,
) -> CsaMetadata:
    """One prefill launch covering ``len(seq_lens)`` requests, packed back to back.

    Requests get disjoint region ranges of ``regions_per_request``, mirroring how the
    backend maps a request's regions through its block table. ``reset_slots`` marks the
    row that opens a region; without it the kernel would fold a fresh region into
    whatever stale accumulator the recycled slot still held.
    """
    generator = torch.Generator(device=device).manual_seed(seed)
    region_size = geo.region_block_size
    rows = int(sum(seq_lens))

    request_ids = torch.cat(
        [
            torch.full((length,), index, device=device, dtype=torch.int64)
            for index, length in enumerate(seq_lens)
        ]
    )
    positions = torch.cat(
        [torch.arange(length, device=device, dtype=torch.int64) for length in seq_lens]
    )
    flat_slot = request_ids * regions_per_request + positions // region_size
    reset_slots = torch.where(
        positions % region_size == 0, flat_slot, torch.full_like(flat_slot, -1)
    )

    starts = torch.tensor(
        [0, *torch.tensor(seq_lens).cumsum(0).tolist()], device=device, dtype=torch.int32
    )
    shape = (rows, geo.indexer_num_k_heads, geo.proxy_dim)
    return CsaMetadata(
        flat_slot=flat_slot.contiguous(),
        reset_slots=reset_slots.contiguous(),
        token_valid=torch.ones(rows, device=device, dtype=torch.bool),
        token_positions=positions.contiguous(),
        query_start_loc=starts,
        seq_lens=torch.tensor(seq_lens, device=device, dtype=torch.int32),
        task_prefix=torch.zeros(
            len(seq_lens) + 1, device=device, dtype=torch.int32
        ),
        index_k=torch.randn(shape, generator=generator, device=device, dtype=dtype),
        index_z=torch.randn(shape, generator=generator, device=device, dtype=dtype),
        region_block_size=region_size,
        regions_per_request=regions_per_request,
    )
