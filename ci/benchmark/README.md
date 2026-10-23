# Independent safe TPS copy

The user originals in Downloads are untouched. `benchmark_pd_tps.py` preserves original random-token generation, client TTFT/prefill TPS, full-overlap E2E decode TPS formula, event ITL and iteration aggregation. `flops_calculator.py` is a byte-identical conventional-name copy. The safe-copy implementation originally performed no remote run. The parent subsequently ran BS1–64 x3 on CPU5 on 2026-10-23; see ../reports/benchmark-performance-summary-20261023.md.

## CPU5 command template

Requires Python >=3.10 and aiohttp, verified legal token IDs, exclusive service, external all-DP idle/trace checks, and the actual verified capacity metadata. Run from the uploaded `cluster-dp8tp4-20261022` directory. Set `VOCAB_SIZE` to the verified legal upper bound; do not use a guessed default. The existing service is responsible for DP8/TP4/EP32/MTP3, API1 and CANN9.2; these benchmark flags do not configure it.

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

## Recorded run

BS1–64 completed, 3 measured iterations each, input131072/output2048, exit0. BS128 was not executed. The all-eight-size command above is a template and intentionally fails preflight with current capacity metadata. Use BS1–64 for the recorded profile; do not bypass the guard. Per-case MTP acceptance counters were not captured.
