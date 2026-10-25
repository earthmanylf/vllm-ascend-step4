# Step5 adaptation review and fixed-acceptance benchmark

Date: 2026-10-25, Asia/Shanghai.

## Delivery decision

**Candidate only; do not merge yet.** The user made merge conditional on the
code review being clear. Four inherited Step5 MTP contract tests still fail on
both the untouched fbc6ba6f84 baseline and the candidate. No expected assertions
were relaxed, skipped, or relabeled as passes to clear that gate.

Prepared branch: `step5-review-acceptance06-20261025`, based on
`fbc6ba6f84b657bdd849aa580d5edcd4512ca6ac`. The target branch is
`step5-preview-tailring-npu`, observed at `c7b7c802fb` before preparation.
The original dirty adaptation worktree and Catlass submodule are preserved.
This is a source candidate, not a rebuilt or released image.

## Previous session

The requested ID prefix `sess_a52R5l9d6cK59` maps to
`sess_a52R5l9d6cK59w` / thread `01a171b7-8bc0-78b2-92c4-1f7e47a3cf43`.
That session completed a natural-acceptance 21-iteration benchmark, not a
fixed-0.6 run. Its BS1/16/64 aggregate acceptance ratios were approximately
0.0068 / 0.5823 / 0.8459. Keep those historical results under that protocol.

## Fixed-0.6 run

Hardware: four nodes, 32 Ascend 950DT devices, CANN9.2, DP8/TP4/EP32,
Step5 W8A8-MXFP8, MTP3. Target FULL_DECODE_ONLY and draft graph execution.
Input 131072, output 2048, BS1/2/4/8/16/32/64, three iterations per case.
Image `7b7e9631ca8ccbfcba33a5556c66e972dc7eb7e23e170171774f90b398e30bd8`
plus the seven validated Python overlays. This is a diagnostic overlay image,
not a pure-source release built from the candidate commit.

Server configuration:

```json
{"method":"mtp","num_speculative_tokens":3,"enforce_eager":false,"rejection_sample_method":"synthetic","synthetic_acceptance_length":2.8}
```

Acceptance means accepted draft tokens divided by drafted tokens. The pinned
vLLM implementation includes one target/bonus token in acceptance length, so
2.8 gives 1.8 accepted drafts per three drafts, or 0.6 in expectation. Its
minimum-variance unconditional prefix rates are `[1.0, 0.8, 0.0]`. This is not
a conditional Bernoulli probability of 0.6 at every position. Record measured
counter deltas; never substitute the target value for the observation.

| Batch size | Prefill tok/s | Decode overlap tok/s | Observed acceptance |
| ---: | ---: | ---: | ---: |
| 1 | 8,951.97 | 85.83 | 0.603567 |
| 2 | 15,903.60 | 172.15 | 0.602924 |
| 4 | 26,798.88 | 341.63 | 0.597757 |
| 8 | 41,275.05 | 683.04 | 0.596375 |
| 16 | 43,493.92 | 1,311.15 | 0.602357 |
| 32 | 45,784.77 | 2,362.10 | 0.598468 |
| 64 | 46,893.94 | 4,276.82 | 0.600703 |

Exit 0; 21/21 measured iterations and 381/381 requests complete,
780288 completion tokens, all decode windows valid, all acceptance
counter windows valid, preemption delta zero. Total acceptance:
`501594 / 835938 = 0.600037323342`.
Individual iteration ratios span 0.593246–0.603752.
The fixed seeds correlate request outcomes, so do not infer independent-sample
confidence intervals from token count alone. End state: health200 and all eight
DP engines running0/waiting0/preemptions0.

The decode metric retains the client's common output overlap window, not a
pure kernel rate. Synthetic sampling changes generated outputs and does not
validate model correctness. The prior natural-acceptance BS64 decode5857.19
at acceptance0.8459 is not a matched comparison to this run's fixed0.6 rate.

## Source changes and validation

The candidate includes the earlier SWA reach, draft-spec marker, fail-closed
BLNHC storage, FIA V1 window, and bucket-owned DSA graph metadata fixes. It also
adds `slot.to(tl.int64)` before summary page-stride multiplication. Production
formatting was checked for unchanged Python AST against the tested overlays.
The summary fix has independent NPU IR and short-long-short model evidence in
the preceding MTP review; synthetic acceptance is not used to replace it.

The checked-in benchmark collector now preserves engine/model labels, measures
counter deltas across the complete iteration, rejects missing/reset evidence,
and samples in a child process. The fixed profile is in
`ci/benchmark/mtp3-acceptance-0.6.json`; historical production startup defaults
continue to use the standard sampler. Raw runtime benchmark source identities
are preserved in the run metadata; candidate cleanup includes formatting,
import ordering and removal of unused assignments.

Validation in this review:

- Focused DSA/summary/decode-graph contracts: 82 passed, 5 skipped because the
  tests could not locate the matching upstream source through their loader.
- Benchmark transport/counter/process tests: 27 passed after cleanup.
- Source-level BLNHC storage tests: 13 passed; four collector package tests
  require generated `_build_info` absent from the fresh source-only tree.
  They are environment-blocked, not reported as passing. Prior isolated
  merged collector CPU evidence covers eight cases, not full package imports.
- Added high-address CPU tests execute the production store expression with
  actual int32 tensors. They complement the existing compiled NPU IR evidence.
- Ruff on all changed Python files passes; production AST equivalence recorded.
- Full Step5 contract directory: 99 passed, 5 skipped, 4 inherited failures.
  The same four failures were independently reproduced on the untouched base.

## Unresolved merge gate

In `tests/ut/models/step5/test_step5_mtp_contract.py`:

1. `MultiModuleContract.test_capture_factory_uses_same_multi_token_state_as_runtime`:
   the fixture base has no `capture`, while draft-prefill capture now delegates
   to `super().capture`.
2. `MultiModuleContract.test_metadata_factory_gets_exact_lengths_and_cache_is_the_built_dict`:
   the fixture descriptor has no `cg_mode`, required by current eager/FULL logic.
3. `ARRejectionContract.test_failed_proposal_clears_rejection_snapshot`.
4. `ARRejectionContract.test_once_per_proposal_snapshot_corrects_all_steps_in_batch_order`.

The AR test loader extracts only direct class methods, but `propose` now lives
inside version-conditioned branches. A separate temporary probe selected the
current-main branch without changing repository assertions: the snapshot
attribute and once-per-proposal D2H behavior expected by these tests are absent.
This is a real test/implementation contract discrepancy. It does not by itself
prove a defect in the measured multi-module Step5 path. Reconcile the generic
AR rejection/CPU-length contract with current upstream before declaring the
whole code review clear; do not delete these assertions to make the suite green.
No new generic-AR runtime behavior is introduced in this candidate.

## Reproduction and artifacts

Run focused contracts in the matching Torch/vLLM environment:

```bash
VLLM_SOURCE_ROOT=/opt/upgrade/src/vllm \
  python -m pytest --confcutdir=tests/ut/models/step5 -q \
  tests/ut/models/step5/test_step5_summary_contract.py \
  tests/ut/models/step5/test_step5_dsa_contract.py \
  tests/ut/models/step5/test_step5_decode_graph.py
python ci/benchmark/test_safe_copy.py -v
```

CPU5 raw run:
`/data/chensiyu/step5-correctness-20261024/mtp-int64-acceptance06/benchmark-acceptance06-full-3iter`.
Bounded evidence archive: `fixed-acceptance-evidence.tar.gz`, with 65 files
verified against its SHA256 manifest after download. Full raw request/SSE
artifacts remain at the remote run directory.

Local audit directory:
`/Users/jyxc-dz-0100403/cpu5-takeover-20261024/audit/merge-20261025/`.
It includes the final JSON, evidence archive, baseline/candidate failure logs,
AR loader probe, source equivalence and lint checks. Synthetic containers are
retained stopped after the run; the previous natural-acceptance graph service
has been restarted; CPU5 API `/health` returned HTTP 200 and the server log
confirmed application startup complete after restoration.
