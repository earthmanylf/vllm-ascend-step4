# BLNHC independent safe TPS copy with metrics evidence

This directory is a separate copy of `work/benchmark`; no parent benchmark,
BLNHC runtime branch, service or dependency was changed.

## Measured-iteration metrics

Each measured iteration (not the original warmup) captures `/metrics` immediately
before submission, at nominal 1-second intervals during execution, and after
completion/failure. **Production sampling runs in a separate Python process**
(`telemetry_worker.py`), not in the completion client's event loop or GIL. This
keeps sampling active while the client synchronously serializes/flushes all
request SSE/JSON artifacts. The child has no model/vLLM imports, uses the same
Python executable, and communicates readiness/stop/report through local files;
the parent waits for the before scrape before sending measured requests.

Each scrape uses a fresh direct HTTP connection, no proxies or redirects, and a
5-second network timeout. Completion pool/timeouts, artifact flush behavior and
request payloads are unchanged. There are no telemetry retries, catch-up bursts
or service mutation endpoints; a genuine slow/failed scrape remains invalid.
The 5-second timeout is **not** inflated to hide failures. A slow scrape, CPU
starvation or filesystem pressure can still leave honest sampling gaps.

For each scrape, `*-metrics-NNNN-{before,during,after}.txt` is the original HTTP
response body; the adjacent JSON records timing, status, headers, body SHA256,
parsed series and running/waiting/preemption/KV observations. A failed transport
produces an empty text artifact and an explicit error JSON, not a valid zero.
`*-metrics-iteration.json` holds the full trace and counter-delta result; every
configuration summary includes `metrics_iterations`, `spec_decode_acceptance`,
and `metrics_scheduler_summary`. Scheduler summaries retain per-DP observed
min/max and preemption delta validity; iteration reports also expose scrape gaps
and invalid snapshot indices. Summary extrema are observations, not a proof that
all extrema occurred at the same instant.

`*-metrics-process-ready.json` identifies the child after its before scrape;
`*-metrics-process-stop.json` records the parent iteration outcome; process
stderr is retained in `*-metrics-process.log`. The completed report records
child/parent PIDs and collector mode. The parent rechecks raw-body hashes,
snapshot JSON, parsed counters, and acceptance before incorporating the report.
Child startup/exit/report-integrity failure stops the benchmark rather than
falling back to in-process collection. Parent exit stops orphan polling; normal
finish waits for the child's after scrape and report. Any invalid during sample
still invalidates acceptance and its weighted configuration aggregate.

The parent's stop marker is sent only after the original measured iteration
returns, including its synchronous per-request and batch artifact writes. Thus
the after scrape is delayed by that flush, while the child keeps collecting
during samples throughout it. This deliberately brackets complete requests,
not the final SSE token timestamp. With an exclusive service and no intervening
workload, the additional interval contains only delayed engine accounting/idle
time; it is not a pure-decode boundary. New workload during that interval would
contaminate attribution. The worker's stop wait/control latency is recorded
through actual snapshot timestamps, not backdated to the final token.

Acceptance uses exactly one `model_name`/`engine` series for each expected DP:
`vllm:spec_decode_num_accepted_tokens_total` delta divided by
`vllm:spec_decode_num_draft_tokens_total` delta, summed across DPs. It does not
sum API/TP replicas or infer missing ranks. Missing/duplicate/malformed counters,
observed decreases/resets, changed series labels, HTTP errors, zero draft delta
or accepted delta greater than draft delta mark the result invalid with ratio
null. Configuration aggregation is the weighted sum of accepted deltas divided
by the sum of draft deltas, using the same trailing iteration selection as TPS;
invalid selected iterations cannot be silently discarded.

These are service-wide, asynchronously refreshed counters: exclusive workload
attribution is required, boundary scrapes can lag engine updates, and a restart
that recovers between samples may be unobservable. Metrics invalidity does not
rewrite or fabricate the original TPS result. The scrape interval is nominal:
slow responses leave observable sampling gaps. Additional client/HTTP/filesystem
load can perturb timings; this copy is an instrumented workload.

**No equivalence claim:** counter acceptance spans the entire measured request
iteration, including prefill; 1s running/waiting/KV samples do not prove pure
decode, simultaneous all-DP residency, absence of sub-second preemption, or
BS128 admission. KV usage fractions remain per-DP, not an additive byte budget.

The user originals in Downloads are untouched. `benchmark_pd_tps.py` preserves original random-token generation, client TTFT/prefill TPS, full-overlap E2E decode TPS formula, event ITL and iteration aggregation. `flops_calculator.py` is a byte-identical conventional-name copy. No remote access or benchmark was performed during implementation.

## CPU5 command (not executed here)

Requires Python >=3.10 and aiohttp, verified legal token IDs, exclusive service, external all-DP idle/trace checks, and the actual verified capacity metadata. Run from the uploaded `cluster-dp8tp4-20261022` directory. Set `VOCAB_SIZE` to the verified legal upper bound; do not use a guessed default. The existing service is responsible for DP8/TP4/EP32/MTP3, API8 and CANN9.2; these benchmark flags do not configure it.

```bash
set -euo pipefail
: "${VOCAB_SIZE:?set verified legal token ID upper bound}"
RUN="artifacts/run-$(date +%Y%m%d-%H%M%S)"
mkdir -p artifacts
python3 benchmark/benchmark_pd_tps.py \
  --host 127.0.0.1 --port 19082 --model step5 \
  --batch-sizes 1,2,4,8,16,32,64,128 --input-lens 131072 --output-lens 2048 \
  --tp-size 4 --dp-size 8 --max-num-batched-tokens 16384 \
  --max-connections 1024 --num-iters 3 --vocab-size "$VOCAB_SIZE" \
  --return-token-ids --disable-mfu --config-retries 0 --failure-cleanup-mode off \
  --request-total-timeout 7200 --request-sock-connect-timeout 30 \
  --request-sock-read-timeout 1800 \
  --capacity-metadata capacity-metadata.json \
  --raw-artifacts-dir "$RUN" --output-file "$RUN/summary.json"
```

For a BS1 first gate, replace only `--batch-sizes 1,2,4,8,16,32,64,128` with `--batch-sizes 1` and select a **new** run directory. There is no invented `--output-tokens` or `--min-tokens` CLI: original-supported `--output-lens 2048` drives payload `min_tokens=max_tokens=2048`, `ignore_eos=true`, `return_token_ids=true`. Original fixed warmup BS16/input1024/output10 remains, explicitly separated/labeled from measured batches.

## Capacity and strict behavior

`--capacity-metadata` is optional in argparse but required to actually run while the server schema is unsupported. Schema: `per_dp_capacity_tokens` object with exactly string ranks `0`..`7`, each positive integer; positive integer `max_model_len`; nonempty `source`. No numeric default or guessed metadata is generated. The parent supplies real log-derived metadata. Capacities are logical per-DP/group-aware capacities, never TP replica or API-local sums. The 0.9 guard on global capacity is **necessary only**; it cannot prove routing balance, per-DP residency, group fit or no preemption. External traces remain required. Current metadata's group-aware maxlen-derived capacity is conservative, not exact 131K aggregate residency.

All explicit cases are capacity/maxlen preflighted before generation. First HTTP/parser/token-count/finish/usage/quality failure stops, cancels local siblings, saves their artifacts, and exits nonzero; local cancellation does **not** prove server cleanup. No retries, pause/resume/abort/clear-cache/weight mutation; cleanup modes other than off and weight-version requests are rejected. Empty, skipped or unvalidated suites cannot succeed. Retry manually only after parent verifies global cleanup. No stable-margin/extra quality threshold was guessed; original coverage>=0.25 and positive-window/event-ITL quality are retained.

## Raw artifacts and semantics

- One UUID request stem: exact received HTTP body bytes `.sse` (not reserialized SSE); matching `.json` includes complete payload/prompt IDs + SHA256, HTTP status/headers, read offsets/lengths/perf-counter receipt timestamps, parsed chunks/token IDs, finish events, optional usage, DONE, result or error. Failure/cancellation writes artifacts too.
- Captured `.sse` is the client-observed prefix up to DONE or failure/cancellation, not unobserved server bytes. Writes occur after request completion/failure; raw receipt recording adds some client overhead.
- Batch/iteration JSON persists each request and original quality/window diagnostics, with warmup explicitly labeled. Run metadata persists argv, seed, wall-clock anchor, script hashes and metric definitions; capacity and server-info probe status/body are saved.
- `completion_tokens=sum(len(token_ids))` must equal requested output exactly; original `output_tokens` remains **nonempty event count**, not generated tokens. No text-retokenization fallback. Missing token IDs, malformed JSON, error event/payload, absent DONE, non-length finish, or optional usage count mismatch fail.
- Last actual-token timestamp is from the final nonempty token event; finish time is separate. Decode window is `(max(first actual-token time), min(last actual-token time)]`, counting non-first-event token IDs inside it. **Client full-overlap E2E only, not proven pure decode/kernel TPS.** ITLs are nonempty SSE-event intervals, not per-token latency or MTP acceptance. Prefill TPS keeps the original total input/max(relative TTFT) formula, including client/network/queue effects.
- `--disable-mfu` never imports/calls the GPU detector or runs nvidia-smi; FLOPS/MFU and peak fields are JSON null, console N/A, with disabled reason. TP/DP/precision are metadata only.
- Summary preserves original aggregation quirks: mean throughput/ITL versus last-iteration window diagnostics. Use iteration JSON for like-for-like analysis. External scheduler and MTP traces are not fabricated or collected by this script.

## Local offline checks

```bash
python3 -B benchmark/test_safe_copy.py -v
python3 -B -c 'import ast,pathlib; [ast.parse(p.read_text()) for p in pathlib.Path("benchmark").glob("*.py")]; print("syntax PASS")'
```

Synthetic in-memory HTTP/SSE tests open no socket and substitute aiohttp locally. Production still needs real aiohttp. Do not interpret these unit tests as service inference or performance validation.

## Fixed acceptance throughput profile

For controlled MTP3 throughput measurements, start every rank with the checked-in
`mtp3-acceptance-0.6.json` as its `--speculative-config` value:

```bash
vllm serve /model/target \
  --speculative-config "$(cat ci/benchmark/mtp3-acceptance-0.6.json)" \
  ...
```

Use the existing DP8/TP4/EP32, CANN9.2, target FULL_DECODE_ONLY and draft graph
settings. The benchmark client cannot change the server's acceptance policy.
The historical image startup scripts retain their natural sampler configuration;
this profile requires a separate service directory and an image/source revision
that includes the BLNHC and summary address fixes.

The target is **accepted draft tokens / drafted tokens = 0.6 in expectation**.
For three draft tokens, `synthetic_acceptance_length=2.8` includes one target or
bonus token, leaving 1.8 accepted draft tokens per step. The pinned vLLM config
resolves this to unconditional prefix acceptance rates `[1.0, 0.8, 0.0]`.
Sampling has statistical variation and request-end effects. Do not overwrite the
observed ratio with the configured target, and do not mistake a conditional
per-position probability of 0.6 for this aggregate metric.

Record the complete speculative config, source identities and all-DP
`spec_decode_num_accepted_tokens_total` / `spec_decode_num_draft_tokens_total` counter
deltas for each measured iteration. Preserve labels, reject missing or reset
series, and report the actual ratio alongside the target. The validated collector
and four-node run artifacts are described in
`../reports/step5-adaptation-review-20261025.md`.

Synthetic sampling accepts draft tokens independently of target agreement. Its
throughput results measure a controlled workload; use the standard sampler and
separate fixtures to validate model outputs and natural MTP acceptance. The
2026-10-23 and earlier 2026-10-25 natural-acceptance runs remain historical
results and must not be relabeled as fixed-0.6 measurements.
