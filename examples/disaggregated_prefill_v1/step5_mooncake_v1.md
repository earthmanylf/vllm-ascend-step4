# Step5 PD with MooncakeConnectorV1

## Source revisions

- Ascend base: `earthmanylf/vllm-ascend-step4`, `main`, `96125cfb93087ec219685d73013e676d4e3ab3bb`.
- vLLM: the **same repository**, branch `water18`, `5d7f9124db0304ad177f71969f26a70316ecc632`.
- Install both peers from the same patched Ascend revision and the pinned vLLM revision.
- No vLLM source changes or new attention operators are required by this adapter.

## Initial supported configuration

Use `MooncakeConnectorV1`, matching P/D TP, PP=PCP=DCP=1, block size 128,
ND main KV cache, and text requests without speculative decoding/MTP.
The model's existing requirement of exactly one local DSA KV head still applies.
Configure the same `prefill` and `decode` topology on both peers.
The adapter rejects unsupported topology and incompatible handoff metadata.
Multimodal and streaming-input requests are not supported by the initial protocol;
ordinary streaming **output** is unaffected.

For example, with TP=4 and DP=1, P uses:

```json
{
  "kv_connector": "MooncakeConnectorV1",
  "kv_role": "kv_producer",
  "kv_connector_extra_config": {
    "prefill": {"tp_size": 4, "dp_size": 1, "pp_size": 1},
    "decode": {"tp_size": 4, "dp_size": 1, "pp_size": 1}
  }
}
```

D uses the same configuration with `kv_role="kv_consumer"`.
Use the existing V1 proxy in this directory: it passes P's transfer metadata to D
while preserving the original prompt. Both services must use the same checkpoint,
tokenizer, KV dtype, and attention settings. This adapter has not yet been validated
on NPU; keep model/CANN/torch-npu versions aligned with the Step5 serving baseline.

## Handoff contract

For original prompt length `L` and page size `B=128`, transfer through
`C = max(0, floor((L-B)/B)*B)`.

1. The connector's `on_new_request` hook shortens P's prompt to `C` **before**
   scheduler prefix lookup. This ensures the ordinary SWA tail ends at `C`.
2. P computes complete KV pages and complete CSA regions. Its sampled output
   token is only a proxy-protocol placeholder, as in the existing V1 proxy flow.
3. The connector registers each DSA owner's FP8 summary backing as a third,
   opaque cache component. Main K/V bindings used by attention remain two tensors.
   Summary pages follow the same source/destination block-ID mapping as their owner.
4. D reports only `C` external tokens (minus any local hit), satisfying water18's
   minimum-one-page replay ceiling. It computes the remaining 128–255 prompt tokens
   and creates fresh CSA active-slot state on its own pages.
5. Physical region IDs, reverse maps, FP32 accumulators and staged K/Z are not
   transferred. A page boundary is also a region boundary, so no incomplete source
   region is needed to resume at `C`.

For `L < 256`, `C=0`: P returns an empty block list, frees its blocks normally,
and D performs the full prefill. This preserves the existing proxy protocol but
does not accelerate these short requests.

The summary buffers are separately registered with 2 MiB alignment. The extra
alignment allocation (at most 2 MiB minus one byte per DSA layer) is included in
the side-storage fixed memory budget. Completed summary payload size and page
budgeting retain the existing Step5 calculation.

## Validation

CPU/mock connector tests cover actual byte copies of K/V and summary pages under
noncontiguous block remapping, registration/handshake metadata, handoff boundaries,
short-prompt release, partial local hits, water18 Request mutation, and unsupported
configuration rejection. Run from the Ascend worktree with the pinned vLLM first
on `PYTHONPATH`:

```bash
PYTHONPATH=/Users/zhuyixiang/Projects/vllm-step5-water18:/Users/zhuyixiang/Projects/vllm-ascend-step4 \
UV_PROJECT_ENVIRONMENT=/Users/zhuyixiang/Projects/vllm-workspace/vllm-ascend/.venv \
uv run --no-sync pytest -q tests/ut/kv_offload/test_mooncake_connector.py
```

The external-service system test compares PD with local prefill on the same D
service, with distinct cache salts to force the remote-load path. It checks greedy
output tokens, log probabilities, region/page boundaries, an 8193-token prompt,
concurrent handoffs and repeated block reuse. With actual P/D services running:

```bash
uv run --no-sync pytest -q \
  --confcutdir=tests/e2e/nightly/multi_node/step5_pd \
  tests/e2e/nightly/multi_node/step5_pd \
  --step5-prefill-url=http://P_HOST:8000 \
  --step5-decode-url=http://D_HOST:8000 \
  --step5-tokenizer=/path/to/step5 \
  --step5-served-model=step5
```

Repeat on eager and graph modes, with chunked prefill and prefix caching as used
in deployment. Set the service context limit above 8209 tokens for the long case.
Passing CPU tests or collecting the service tests is not an NPU accuracy result.

Local verification on 2026-09-24: the connector suite passed **133 tests and
42 subtests** against the pinned water18 source. Ruff checks passed for the
connector, transfer helper and tests; `git diff --check` passed. The 14 external
service tests were collected and skipped because P/D endpoints were not supplied.
The model-level Step5 tests could not run on this Mac because Triton is unavailable;
the repository's existing summary test also still imports the old `step4` namespace.
