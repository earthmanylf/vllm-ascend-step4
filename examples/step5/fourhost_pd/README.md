# Draft: four-host Step5 PD validation on Ascend 950DT

**Review-only, not production acceptance.** This AI-assisted change was requested
and exercised by the operator. Native failed-READ DMA quiescence, complete
READ→DONE→allocator-free tracing, fine-grained precision and performance remain
open. Do not infer safe automatic failure recovery from successful requests.

## Verified deployment

| Role | Hosts / node ranks | Parallelism | Endpoint |
|---|---|---|---|
| P | 0025 (`10.208.128.28`), 0028 (`10.208.128.31`) | PP4 × TP4, DP1; partition 24,25,25,18 | `:19171` |
| D | 0026 (`10.208.128.29`), 0027 (`10.208.128.30`) | DP4 × TP4, local DP2, PP1 | head `:19172` |

The scripts here are **copies of the actually executed launch scripts**, not
portable auto-discovery wrappers. Review every host, NIC, port, image and bind
mount before reuse. `launch_prefill.sh` / `launch_decode.sh` reject occupied NPUs,
reused names and selected occupied ports; they do not stop existing services.
They require `sudo -n docker` and the `bond4` interface.

Both roles use one read-only `/work/model-target` checkpoint, including its VL
configuration and original MTP layers, with no split target/draft checkpoint.
MoE SP is selected through `--all2all-backend allgather_reducescatter` + EP,
not legacy FlashComm1 flags. This does not force the Ascend runtime to use the
ALLGATHER dispatcher: the runtime separately selects MC2/other implementations.
BLNHC, prefix caching, async scheduling, MTP3 and FULL_DECODE_ONLY graph capture
(up to 512) were enabled. Both roles must use only
`{"method":"mtp","num_speculative_tokens":3}` without a batch-size schedule.
No companion vLLM constant-K configuration exception is required by this candidate;
validation of native first-admission padding is recorded separately below.

### Environment and mounts

- Image: `hub.i.basemind.com/stepcast/stepcast:0.30.0rc2-step5-20261016-vllm-e177c621f2-ascend-a2ac749ee3`.
- Python 3.12.13; torch `2.10.0+cpu`; torch_npu `2.10.0.post4`;
  transformers `5.14.1`; CANN `9.2.0-beta.2`.
- Mooncake NPU `0.3.11.post1`; loaded `libtransfer_engine.so` SHA256:
  `f9b835dc2bfe8a3223c03ab241ad0a0d56a7a771b035e3348298ce759b6be409`.
- Model config SHA256:
  `a2a0b0a5576656bb16ad871b8417bdd89c780bc474bde30b2dd0b88fc6e679be`.
- Custom `step5_ops_transformer/op_api/lib/libcust_opapi.so` SHA256:
  `d6d06852629e58b9f615176afe1a4558825bd1d9f47ca8821ca80d13ac22dcc5`.
- Host model: `/data/models/Step-5-Preview-w8a8-mxfp8` → `/work/model-target`.
- P workspace: `/home/k8s_ops/step5-pd-fourhost-049` → `/work/run`.
  Its runtime directory is `runtime-071`.
- D workspace: `/home/liuyi/step5-pd-twohost-20261027` → `/work/run`,
  using `runtime-072`.

The image's installed package versions are not source identities: `PYTHONPATH`
overlays candidate source. The native validation in experiments 070/071 used
vLLM `ca5e43ddcd` with its configuration file restored byte-for-byte, without
the constant-K exception. Ascend sources include the target's D2H optimization
and this backoff candidate. Step5 uses exact draft FIA lengths after rejection,
which restores a CPU readback for correctness; no decode performance gain is
claimed. Revalidate a fresh build of the final MR commit before production use.

## Reproduction outline

1. Stage the reviewed candidate sources with the original matching vLLM source
   in the runtime directories above. No constant-K DP exception is needed.
   Do not replace active runtimes in place.
2. Copy `serve_prefill.sh` to both P hosts as `serve_pp4tp4_071.sh`;
   copy `serve_decode.sh` to both D hosts as `serve_dp4tp4_072.sh`.
3. On P node1 run `bash launch_prefill.sh 1`, then node0
   `bash launch_prefill.sh 0`. On D node1 run `bash launch_decode.sh 1`,
   then node0 `bash launch_decode.sh 0`. These launch separate named containers.
4. Require all 16 workers of each role to finish graph capture and KV
   registration, then `/health` 200. P handshake must have 16 workers,
   PP ranks 0–3 and TP4; D has four independent DP engines.
5. Send ordinary basic-correctness requests before PD. Follow upstream NIXL's
   multi-module MTP prefill-backoff semantics: P receives the original request
   with `max_tokens=1, do_remote_decode=true`, but computes only the safe prefix.
   D receives the **original full prompt**, not P's sampled token appended to it,
   plus the returned transfer parameters, and recomputes the trailing lookahead
   locally. Return D's output to the client; P's sampled output is not a user
   generation prefix. Never reuse an export or invent DONE acknowledgements.
   This updated path requires new live acceptance; old evidence below used a
   different seed-based handoff and must not be attributed to it.
6. On ambiguous READ failure retain both owners and buffer leases; do not
   interpret timeout, process exit or bookkeeping completion as a DMA fence.

The scheduler/metadata exchange uses global layer names and stage-local
addresses/strides. P handshake exposed 141 cache entries in five global groups,
with 36/37/37/31 entries across stages. Empty projected groups retain scheduler
group indices. The historical seed-based implementation is being replaced by
upstream-style trailing-token recomputation; do not require or consume its
private seed envelope in the native handoff. Control ACK is not a DMA fence.

## Evidence collected (not an exhaustive correctness claim)

Local evidence root: `/data/LY/logs/update-step5/PD/` (not committed; scripts and
this summary are included here). Each experiment has copied scripts, raw logs,
request/response evidence and checksums.

**Historical scope:** experiments 042–059 below used native MTP3 on P but an
extra constant batch-size schedule on D, plus an uncommitted vLLM configuration
exception. They must not be cited as both-role native MTP3 acceptance. That
exception is not being submitted. Experiments 062 onward remove the added
schedule and restore the original vLLM configuration source on both roles;
062/065 first failed with the old seed protocol. Only 070/071 below establish
the new upstream-backoff success, not every experiment numbered 062 onward.

- `042`: PP producer registration/graphs, five basic checks and an export of
  five block groups plus four TP seed records.
- `048`: DP4 consumer registration/graphs and five basic checks.
- `051/052`: first four-host response, 32 greedy IDs equal to the ordinary D
  reference and finite logprobs; 28 successful READ calls / 944 ranges /
  46,776,320 bytes from both P hosts.
- `053/054`: 127/128/129/511/512/513 prompt boundaries, 32 IDs each equal to
  post-transfer ordinary references; actual READ logs correlated by request.
- `055`: fixed D DP0, fresh-prefix 513-token cold/hot pair:
  197,705,728 → 84,508,672 READ bytes. Not a throughput speedup measurement.
- `056`: separately prewarm P or D; both asymmetric cases match reference IDs.
- `058`: four independent concurrent transfers, 128 output IDs total matching
  references; normal-path idle/KV metrics return to zero.
- `059`: close a streaming client after its first returned token, then a new
  PD request matches 32 reference IDs. **Not an in-flight DMA cancel test.**

## Open acceptance gates

- Installed native fork provenance and failed-batch/retry physical drain;
  source review found a possible retry-success path with an earlier busy batch.
- Complete real READ→DONE→allocator-free tracing and in-flight cancellation.
  Python unknown paths retain/quarantine; this is not automatic recovery.
- Detailed token/logprob tolerance, forced MTP rejection/rollback and mixed load.
  Historical logprob mismatches remain open despite current short IDs matching.
- One-million-token capacity, image-input PD, long-running stability and full
  production performance. Current runs use max length 65,536 and 2 GiB KV.
- Missing audit optimizations (RMSNorm, CSA/decode layout dedup) are separate work.
- Transfer Prometheus metrics currently remain zero because the READ success
  path does not record into the stats object; direct READ logs are the evidence.
- Reasoning parser is not configured; the checkpoint template opens `<think>`.
  Basic checks inspect the final text after `</think>`, not raw thought text.

## Native MTP3 validation (replacement path, 070/071)

Both roles used only `{"method":"mtp","num_speculative_tokens":3}`. All four
`vllm/config/vllm.py` files match the original SHA256
`50fc2d9d8f6ea8e354fb433bfb3cd479ae5a63c3fb98841c63d7925329b72eb3`.
P/D each finished 16 worker graph captures without ERROR lines. P/D arithmetic
checks passed; P exported no seed and only the safe prefix.

| Original prompt | P remote credit | Successful READ calls | READ bytes | D output |
|---|---|---|---|---|
| 128 | 126 | 28 | 46,776,320 | 32 IDs equal to ordinary reference |
| 129 | 127 | 28 | 46,776,320 | 32 IDs equal to ordinary reference |
| 513 | 511 | 28 | 159,973,376 | 32 IDs equal to ordinary reference |

Logprobs are finite; exact LP tolerance, forced rejection parity, native failed
DMA quiescence, image-input PD and production performance remain open. These
are bounded success-path checks, not exhaustive acceptance.
