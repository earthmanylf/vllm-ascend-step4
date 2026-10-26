# Step5 PD normal-path preview validation

## Release status: preview, not failure-safe production acceptance

This checkpoint packages the validated **same-topology normal pull path**. It
is not certification of the full checkpoint or of general production safety.
See [PD_PREVIEW.md](PD_PREVIEW.md) for the deployable configuration and handoff.

**Known P1:** installed Mooncake NPU `0.3.11.post1` can return from a failed or
expired batch READ while other native tasks are still active. Current worker
completion and DONE handling can then permit P/D block reclamation. A normal
workload can encounter transport errors and trigger this risk. Python-future
completion, producer timeout retention, or successful recovery does not establish
native DMA quiescence. This preview does not fix that boundary.

The independent newer Mooncake ticket build and failure-retention experiments
are deliberately **not included** in this checkpoint. Do not claim that their
existence makes this version failure-safe. Do not infer safe teardown from a
local timeout or automatically restart/reuse failed-transfer memory.

## Supported validation profile

- Registered Ascend `MooncakeConnectorV2` / `MooncakePullConnector`, producer XOR
  consumer; no connector module override.
- **MRv2 explicitly enabled**, text-only, MTP/speculation off, prefix caching off.
- KVPP, sparse offload, HiSparse and other alternative allocation paths disabled.
- Equal P/D TP/EP, PP/PCP/DCP=1, matching cache geometry, dtype/scales and groups.
- Same checkpoint/config/tokenizer on both sides: the handshake checks cache
  descriptors/topology, **not checkpoint identity**.
- Ten-layer text-only model view, TP4 P on devices0–3 and TP4 D on devices4–7,
  eager mode, context4096, batch512, one scheduled request, KV budget1GiB.

The constructor guard rejects several unsupported configurations, but passing
it alone does not prove membership in this MRv2 preview profile.

## Production behavior included

Main NHD K/V are transferred as whole blocks with matching TP shard identity.
Independent summary/ring groups keep separate block-ID mapping and loaded-group
reporting; the request-private circular ring is transferred as one physical page.

Eligible PD caches use one aligned shared slab. Its single 2MiB alignment reserve
is deducted from the original worker budget. Positive explicit block overrides
must also fit that original budget, including the reserve; typed cache views
preserve the backing storage offset. Other allocation paths retain their fallback.

P processes the **complete original prompt** and samples the first output token.
Append that token to D's prompt, request the remaining generation budget, and
compose P's token/logprob followed by D's. P has not computed KV for its sampled
token: Step5 matched credit is capped at `remote_num_prompt_tokens`.

Producer timeouts log once and retain delayed-free state until DONE. Lost DONEs
can pin cache capacity indefinitely. This prevents clock-only reclamation; it
**does not** repair the known failed-native-READ/DONE safety issue above.

## Validation evidence

| Check | Observed result | Boundary |
|---|---|---|
| Actual P4/D4 handoff | 13 lengths:1–9,127,128,129,513;104 output IDs and all generation logprobs exact; max delta0 | Reduced model, not original92-layer dualTP8 |
| Fresh installed source rerun | Same13 cases exact0; tested P/D exited; basic health200 restored | No diagnostic/test-side credit cap |
| Controlled cancellation | Abort while RECEIVE publication held after a synchronous READ returned; groups retained; probes did not reuse held IDs | Not proven cancellation during pending native DMA |
| Owned non-listening peer failure | Real no-peer failure, terminal reporting and exact recovery | Injected failing rank has no DMA; not generic failed-DMA quiescence |
| Production CPU protocol suite | 134 passed | Source-isolated methods, native networking/device doubles |
| Shared KV allocation/budget | 31 passed +20 subtests | Real CPU storage/planner; malformed geometry/override checks |
| Summary contract | 44 passed | Independent CPU reference |
| CANN-first loader | 14 passed; real import-only clean exit and online transport | V2 owner path, not all eager legacy imports |
| Shared-slab NPU fixtures | Prefixes2–6;12 reads/598016 bytes each; both children exit0 | Fixture-built metadata |
| Independent-buffer NPU fixtures | Prefixes1/7;110592 bytes each; byte-exact closed summary; exit0 | Small buffers, not full-model acceptance |

Detailed evidence is retained outside Git under `logs/update-step5/`, including
`container-logs/fresh-pd-normal-abnormal-evidence.tar.gz`. Historical PIDs are not
process-ownership or current-health promises.

## Reproduce focused CPU checks

Use a Python environment containing torch/pytest and the matching vLLM source:

```bash
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export VLLM_SOURCE_ROOT=/absolute/path/to/vllm
python3 -m pytest --confcutdir=tests/ut/kv_offload/mooncake_v2 tests/ut/kv_offload/mooncake_v2/test_step5_transfer_contract.py -q
python3 -m pytest --confcutdir=tests/ut/worker/v2 tests/ut/worker/v2/test_step5_kv_contract.py -q
python3 -m pytest --confcutdir=tests/ut/models/step5 tests/ut/models/step5/test_step5_summary_contract.py -q
python3 -m pytest --confcutdir=tests/ut/distributed/kv_transfer/utils tests/ut/distributed/kv_transfer/utils/test_mooncake_transfer_engine.py -q
```

Full import-based suites/CI are not claimed green; `pre-commit` was unavailable,
and the tested image has dependency-declaration differences recorded separately.
General EOS/stop/streaming proxy composition, long-context sparse selection,
failed READ quiescence, full-model simultaneous TP8 P/D, offload and MTP-PD remain
outside acceptance. Failure-state hardening continues independently after this
normal-path preview checkpoint.
