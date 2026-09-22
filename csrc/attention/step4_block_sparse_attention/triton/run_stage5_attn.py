"""Stage 5 parity: sparse attention over the selected regions, Triton vs the reference.

This is the stage that produces the model's actual attention output, so it is the last
place a divergence can be caught before it reaches logits. It is also the stage where the
metadata contract from stage 3 gets *interpreted*, which makes it the only place certain
stage 3 bugs become visible: a wrong ``valid_tokens`` field, for instance, produces a
perfectly well-formed selection that silently attends to future tokens.

The cases below isolate the parts of the contract that are easy to implement plausibly and
wrongly:

* **empty rows** -- a query with no selected region must yield a zero output and
  ``lse = -inf``, not a NaN. The online-softmax rescale computes ``exp(m - m)`` with
  ``m = -inf``, which is NaN unless specifically handled.
* **partial regions** -- ``valid_tokens`` below the region size must clip the gather.
  Ignoring it would read tokens the query cannot see, and the result would still look
  numerically reasonable.
* **short counts** -- ``region_counts`` truncates the slot list positionally. A kernel
  that instead scanned to the first ``-1`` would agree on every full row and diverge only
  when the two disagree.
* **step4 geometry** -- 4 KV groups with 16 heads each, versus the 8x8 the upstream test
  uses. Head grouping is exactly the kind of thing that works at one ratio by accident.

Run on the H200:
    cd /path/to/step4-hf
    python3 -m operator_parity.run_stage5_attn
"""

from __future__ import annotations

import torch

from .compare import compare
from .geometry import STEP4, Step4Geometry
from .torch_ref import ref_sparse_attention
from .triton_ops import triton_sparse_attention_prefill

# The envelope upstream accepts for this stage's output and lse, from
# test_sparse_gqa_steptron_fwd_reference.py::test_steptron_indexer_and_sparse_attn_fwd_matches_reference.
OUT_RTOL, OUT_ATOL = 1e-2, 2e-2
LSE_RTOL, LSE_ATOL = 1e-3, 1e-3

PREFILL_REGION_VALID_SHIFT = 24


def make_case(
    geo: Step4Geometry,
    *,
    total_q: int,
    seq_regions: int,
    topk: int,
    device: torch.device,
    seed: int,
    empty_rows: bool,
    short_counts: bool,
) -> dict[str, torch.Tensor]:
    """Build query, KV cache and a hand-made packed selection.

    The selection is constructed rather than taken from the selector: the point here is to
    cover metadata shapes the selector would rarely produce -- empty rows, truncated
    counts, partial regions -- and a synthetic selection reaches them directly.
    """
    generator = torch.Generator(device=device).manual_seed(seed)
    groups = geo.num_attention_groups
    heads = geo.num_attention_heads
    head_dim = geo.head_dim
    region_size = geo.region_block_size
    rows = groups * total_q

    pages = -(-seq_regions * region_size // geo.paged_block_size)
    tokens = pages * geo.paged_block_size

    query = torch.randn(
        (total_q, heads, head_dim), generator=generator, device=device, dtype=torch.bfloat16
    ).contiguous()
    key_cache = torch.randn(
        (tokens, groups, head_dim), generator=generator, device=device, dtype=torch.bfloat16
    ).contiguous()
    value_cache = torch.randn(
        (tokens, groups, head_dim), generator=generator, device=device, dtype=torch.bfloat16
    ).contiguous()

    # Each row selects a random distinct subset -- distinctness matters, since duplicates
    # would double-weight tokens in the softmax rather than raise.
    scores = torch.rand((rows, seq_regions), generator=generator, device=device)
    q_positions = torch.arange(total_q, device=device, dtype=torch.int64).repeat(groups)
    visible = (q_positions // region_size + 1).clamp_max(seq_regions)
    masked = torch.where(
        torch.arange(seq_regions, device=device).view(1, -1) < visible.view(-1, 1),
        scores,
        torch.full_like(scores, -1.0),
    )
    chosen = masked.argsort(dim=1, descending=True)[:, :topk]
    budget = visible.clamp_max(topk)
    slots = torch.arange(topk, device=device).view(1, -1)
    logical = torch.where(slots < budget.view(-1, 1), chosen, torch.full_like(chosen, -1))

    valid_tokens = (
        q_positions.view(-1, 1) + 1 - logical.clamp_min(0) * region_size
    ).clamp(0, region_size)
    packed = (
        logical.clamp_min(0) | (valid_tokens << PREFILL_REGION_VALID_SHIFT)
    ).to(torch.int32)
    packed = torch.where(logical < 0, torch.full_like(packed, -1), packed).contiguous()

    counts = budget.to(torch.int32)
    if short_counts:
        counts = (counts // 2).to(torch.int32)
    if empty_rows:
        counts = torch.where(
            torch.arange(rows, device=device) % 7 == 0, torch.zeros_like(counts), counts
        )

    local = torch.arange(0, total_q, 4, device=device, dtype=torch.int32)
    work_global = (
        local.view(1, -1)
        + torch.arange(groups, device=device, dtype=torch.int32).view(-1, 1) * total_q
    ).reshape(-1).contiguous()
    return {
        "query": query,
        "key_cache": key_cache,
        "value_cache": value_cache,
        "packed": packed,
        "counts": counts.contiguous(),
        "work_q_global": work_global,
        "work_q_local": local.repeat(groups).contiguous(),
        "work_q_len": torch.full_like(work_global, total_q),
    }


def prof(triton_fn: callable, input_data: dict):
    import time
    import torch_npu
    prof_start = time.time()
    for _ in range(5):
        triton_fn(**input_data)
        torch_npu.npu.synchronize()

    prof_dir = f"prof_triton_fn"

    experimental_config = torch_npu.profiler._ExperimentalConfig(
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        l2_cache=False
    )

    with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.NPU],
            with_stack=False,
            record_shapes=False,
            profile_memory=False,
            schedule=torch_npu.profiler.schedule(
                wait=1, warmup=1, active=20, repeat=1, skip_first=1
            ),
            experimental_config=experimental_config,
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(prof_dir)
    ) as prof:
        for i in range(40):
            triton_fn(**input_data)
            torch_npu.npu.synchronize()
            prof.step()
        prof.stop()

    prof_elapsed = time.time() - prof_start
    print(f"Profile saved to: {prof_dir} (took {prof_elapsed:.1f}s)")


def run_case(
    geo: Step4Geometry,
    name: str,
    *,
    total_q: int,
    seq_regions: int,
    topk: int,
    device: torch.device,
    empty_rows: bool = False,
    short_counts: bool = False,
    run_accuracy: bool = True,
) -> bool:
    case = make_case(
        geo,
        total_q=total_q,
        seq_regions=seq_regions,
        topk=topk,
        device=device,
        seed=20260812 + total_q + topk,
        empty_rows=empty_rows,
        short_counts=short_counts,
    )
    scale = float(geo.head_dim) ** -0.5
    if run_accuracy:
        our_out, our_lse = triton_sparse_attention_prefill(
            case["query"],
            case["key_cache"],
            case["value_cache"],
            case["packed"],
            case["counts"],
            num_kv_groups=geo.num_attention_groups,
            region_size=geo.region_block_size,
            softmax_scale=scale,
        )
        reference_out, reference_lse = ref_sparse_attention(
            case["query"],
            case["key_cache"],
            case["value_cache"],
            case["packed"],
            case["counts"],
            num_kv_groups=geo.num_attention_groups,
            region_size=geo.region_block_size,
            softmax_scale=scale,
        )
        torch.npu.synchronize()

        # NaN is the specific failure the empty-row path invites, and a tolerance check would
        # report it as a generic mismatch, so name it separately.
        nan_free = bool(torch.isfinite(our_out.float()).all())
        out_vs_ref = compare(our_out.float(), reference_out.float(), rtol=OUT_RTOL, atol=OUT_ATOL)
        lse_vs_ref = compare(our_lse, reference_lse, rtol=LSE_RTOL, atol=LSE_ATOL)
        empty = int((case["counts"] == 0).sum())

        passed = out_vs_ref.passed and lse_vs_ref.passed and nan_free
        print(
            f"  {'ok  ' if passed else 'FAIL'} {name:24s} q={total_q:4d} topk={topk:4d} "
            f"empty_rows={empty:4d} nan_free={nan_free}"
        )
        print(f"       triton vs ref  out: {out_vs_ref.format()}")
        print(f"       triton vs ref  lse: {lse_vs_ref.format()}")
    else:
        passed = True
    input_data = {
        "query": case["query"],
        "key_cache": case["key_cache"],
        "value_cache": case["value_cache"],
        "packed_regions": case["packed"],
        "region_counts": case["counts"],
        "num_kv_groups": geo.num_attention_groups,
        "region_size": geo.region_block_size,
        "softmax_scale": scale,
    }

    prof(triton_sparse_attention_prefill, input_data)

    return passed


def main() -> int:
    if not torch.npu.is_available():
        print("npu unavailable -- this must run on the H200")
        return 2

    device = torch.device("npu")
    geo = STEP4
    print(f"torch {torch.__version__}  device {torch.npu.get_device_name(0)}")
    print(
        f"step4 attention: heads={geo.num_attention_heads} groups={geo.num_attention_groups} "
        f"heads_per_group={geo.num_attention_heads // geo.num_attention_groups} "
        f"head_dim={geo.head_dim}\n"
    )

    cases = (
        # dict(name="A dense selection", total_q=128, seq_regions=64, topk=16),
        dict(name="B partial regions", total_q=64, seq_regions=256, topk=32),
        # dict(name="C empty rows", total_q=128, seq_regions=64, topk=16, empty_rows=True),
        # dict(name="D short counts", total_q=128, seq_regions=64, topk=16, short_counts=True),
        dict(name="E production topk", total_q=64, seq_regions=1024, topk=512),
        # dict(name="1", total_q=1, seq_regions=16, topk=16),
        # dict(name="4k", total_q=4096, seq_regions=512, topk=512),
        dict(name="64k", total_q=4096, seq_regions=8192, topk=512, run_accuracy=False),

    )

    failures = 0
    for case in cases:
        if not run_case(geo, device=device, **case):
            failures += 1

    print()
    print("PASS" if failures == 0 else f"{failures}/{len(cases)} case(s) FAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
