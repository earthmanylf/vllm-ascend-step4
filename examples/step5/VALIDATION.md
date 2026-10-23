# Basic validation checkpoint — 2026-10-23

## Verified configuration and results

Eight Ascend950DT cards, CANN 9.2.0-beta.2, official vLLM source baseline
`8407bc0bb6` plus stepcast plugin, and this Ascend branch. ModelSlim checkpoint,
TP8/EP8, 262144 configured context limit, MTP=3, FULL_DECODE_ONLY, capture size 4.

- Models, real chat requests and subsequent health checks: HTTP 200.
- Repeated greedy short requests: identical token IDs and normal stop.
- 2135-token prompt: chunked prefill beyond the 1024-token batch limit,
  51 completion tokens, normal stop; subsequent health check passed.
- Local/container SHA256 comparison: all 12 checked production files matched.
- CPU summary contracts: 44 passed; KV contracts: 21 passed.
- Four focused MTP API/bounds checks: 4 passed (not a full MTP suite pass).
- Real NPU TailRing decode/verify/prefill/chunked-prefill: byte-identical to an
  ordered FP32→BF16→FP8 reference in a 129-token cross-page case.
- Greedy rejection: known full-accept, first/middle rejection and bonus cases passed.
- GBSA verify versus single-token decode: maximum error 0.00390625;
  dense-causal comparison errors below 0.007 in the tested case.
- Sliding-window FIA verify/decode: equal in the tested cases, with dense-causal
  errors below 0.007. Page-strided native scorer and KV scatter checks passed.

## Important corrections

Summary/main/ring groups have independent manager IDs. Summary writes and scorer
reads must use **summary metadata**, never main block IDs. Physical page padding
does not co-group layers. Wrong IDs can overwrite main K in the shared allocation.

The new descriptor-backed layout and budget predicate agree: a shared pool is
allocated once; enabling the full budget without using the same allocator would
OOM. Cache-copy deduplication includes view geometry, and all source views are
snapshotted before destination writes to handle overlaps/chains correctly.

## Limits — not verified capabilities

- 262144 is a successfully configured server limit; the longest serving smoke
  prompt was 2135 tokens. No full-context or production load stress test.
- MTP on/off greedy token streams are not bit-identical. Fresh prefill/decode
  within MTP-off also differ at near-tied candidates. A same-shape future-token
  perturbation preserved the tested prefix probabilities, but this does not
  establish full-model numerical equivalence or batch invariance.
- Full historical DSA/MTP suites retain mock/API/dependency failures. Focused
  results above must not be described as a green full CI run.
- Concurrent mixed scheduling, prefix caching, preemption and every quantization
  path have not received complete hardware regression coverage.
- Snapshotting overlapping copy views increases temporary memory with copy size;
  no peak-memory/latency stress measurement.
- **PD and offload are required follow-up work, not implemented/validated here.**
  Default disabled FLY options are accepted for main API compatibility;
  enabled unsupported FLY/watermark rejection options fail explicitly.

## Evidence locations

Development container `step5-rebase-20261021`, namespace
`step5-fp8-review-20261014`, host `gpu-a950dt-0002.lgcm.sh.istep.fun`:

- `/work/step5-rebase-validation/serve-final-mtp-graph.log`
- `/work/step5-rebase-validation/serving-smoke-final.log`
- `/home/liuyi/step5-rebase/artifacts/final-validation-evidence.tar.gz`
- `/home/liuyi/step5-rebase/artifacts/final-source-sha256.txt`

Local archive outside the repository:
`/data/LY/logs/update-step5/BASIC_VALIDATION_REPORT.md` and
`/data/LY/logs/update-step5/container-logs/final-validation-evidence.tar.gz`.
Binary logs/checkpoints/compiled artifacts are deliberately not committed.
