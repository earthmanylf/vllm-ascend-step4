# Draft: four-host Step5 PD validation on Ascend 950DT

**Review-only, not production acceptance.** This AI-assisted change was requested
and exercised by the operator. Native failed-READ DMA quiescence, complete
READ→DONE→allocator-free tracing, fine-grained precision and performance remain
open. Do not infer safe automatic failure recovery from successful requests.

## Verified deployment

| Role | Hosts / node ranks | Parallelism | Endpoint |
|---|---|---|---|
| P | 0025 (`10.208.128.28`), 0028 (`10.208.128.31`) | PP4 × TP4, DP1; partition 24,25,25,18 | `:19149` |
| D | 0026 (`10.208.128.29`), 0027 (`10.208.128.30`) | DP4 × TP4, local DP2, PP1 | head `:19147` |

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
(up to 512) were enabled. D uses a constant K=3 batch lookup for first-admission
padding; it needs the companion vLLM constant-K DP configuration change.

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
  Its runtime directory is named `runtime-047` (shared candidate snapshot).
- D workspace: `/home/liuyi/step5-pd-twohost-20261027` → `/work/run`,
  also using `runtime-047`.

The image's installed package versions are not source identities: `PYTHONPATH`
overlays candidate source. Actual NPU evidence used vLLM `ca5e43ddcd` plus the
constant-K change, and Ascend `8458ec69d` plus the working changes. This Draft is
rebased onto the newer target containing the decode D2H optimization; **that
combined candidate must be revalidated on NPU**. Existing evidence must not be
attributed to that newer combined commit.

## Reproduction outline

1. Stage the reviewed candidate sources and companion vLLM change in the runtime
   directories above. Do not replace active runtimes in place.
2. Copy `serve_prefill.sh` to both P hosts as `serve_pp4tp4_049.sh`;
   copy `serve_decode.sh` to both D hosts as `serve_dp4tp4_047.sh`.
3. On P node1 run `bash launch_prefill.sh 1`, then node0
   `bash launch_prefill.sh 0`. On D node1 run `bash launch_decode.sh 1`,
   then node0 `bash launch_decode.sh 0`. These launch separate named containers.
4. Require all 16 workers of each role to finish graph capture and KV
   registration, then `/health` 200. P handshake must have 16 workers,
   PP ranks 0–3 and TP4; D has four independent DP engines.
5. Send ordinary basic-correctness requests before PD. For PD use a fresh P
   export (`max_tokens=1`, `do_remote_decode=true`), append its sampled token
   to the original prompt IDs, and pass the returned transfer parameters to D.
   Never reuse an export after completion or invent DONE acknowledgements.
6. On ambiguous READ failure retain both owners and buffer leases; do not
   interpret timeout, process exit or bookkeeping completion as a DMA fence.

The scheduler/metadata exchange uses global layer names and stage-local
addresses/strides. P handshake exposed 141 cache entries in five global groups,
with 36/37/37/31 entries across stages. Empty projected groups retain scheduler
group indices. MTP seed publication is restricted to the final PP stage's four
TP ranks. Seed storage ACK is not KV completion.

## Evidence collected (not an exhaustive correctness claim)

Local evidence root: `/data/LY/logs/update-step5/PD/` (not committed; scripts and
this summary are included here). Each experiment has copied scripts, raw logs,
request/response evidence and checksums.

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
