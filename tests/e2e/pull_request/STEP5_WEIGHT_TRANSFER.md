# Step5 live weight-update regression tests

The oracle follows [PR #18634](https://github.com/vllm-project/vllm-ascend/pull/18634):

```text
dummy != startup(W)
live_update_1(W) == startup(W)
live_update_2(W) == startup(W)
```

The reference server loads the original safetensors checkpoint directly, not
a checkpoint written by the transfer source. The transport preserves native
checkpoint names, shapes, values and dtypes, including FP32 router parameters.
No HF-to-checkpoint converter is introduced. The source contract UT checks
identity mapping; it does not claim to validate upstream model-loader mappings.

Reference and live servers run sequentially with TP1, BF16, NZ disabled,
FULL_DECODE_ONLY, identical tokenizer/processor assets and sampling settings.
Signatures compare generated tokens and finite logprobs exactly. Reference
repeatability is checked first; nondeterminism is a failure, not a reason to
silently relax tolerances. Prefix caching is disabled. IPC discards and restores
weight/KV allocations between transactions. The VL lane also disables processor
caching and requires a visual-only corruption to change the image signature,
then verifies restoration of the original weights.

## Fixtures and execution

Use an exclusively allocated Linux NPU container. IPC needs one card, HCCL needs
two (inference on card 0, trainer on card 1); HCCL here is unpacked and is not a
TP2 test. Leave ASCEND_RT_VISIBLE_DEVICES unset in the pytest process. The
test gives the server its own device visibility mapping.

Supply local **complete reduced BF16 checkpoints**, not full-scale weights with
a layer override. Text and VL checkpoints must retain routed/shared experts,
router bias, attention/indexer parameters and lm_head. A VL checkpoint also
needs vision/projector weights, tokenizer, chat template and processor assets.
Keep model-specific dimensions/schedules valid. Confirm that the IPC server
weights plus the complete unpacked trainer payload fit simultaneously on one
card; sleep releases KV storage but does not remove that requirement.

```bash
pytest -sv tests/e2e/pull_request/one_card/test_step5_weight_transfer.py \
  --step5-text-checkpoint /fixtures/step5-text-bf16 \
  --step5-vl-checkpoint /fixtures/step5-vl-bf16

pytest -sv tests/e2e/pull_request/two_card/test_step5_hccl_weight_transfer.py \
  --step5-text-checkpoint /fixtures/step5-text-bf16

pytest -q tests/ut/models/test_step5_model.py \
  tests/ut/models/test_step5_vl_weights.py \
  tests/ut/distributed/weight_transfer/test_step5_weight_source.py
```

An omitted fixture option causes an explicit skip, not a passing hardware
result. Configured fixtures with wrong architecture or missing required weight
families fail. CI must provision the fixtures and pass these options before
claiming E2E coverage; no private A5 path is assumed or embedded in the tests.
Each E2E lane has a one-hour subprocess deadline to bound native collective
hangs, in addition to HTTP deadlines. Logs are inherited by pytest.

## Known validation gaps

- This branch's HCCL unpacked receiver calls load_weights per tensor, whereas
  Step5 validates complete fused projection streams. The E2E intentionally
  exposes any incompatibility instead of xfail-ing or changing the receiver.
- IPC START/FINISH hooks in this branch are no-ops. Post-load derived state,
  repeated updates and graph-captured storage require actual NPU verification.
- The VL ordering UT executes the real wrapper with a consuming loader double;
  it does not exercise the upstream AutoWeightsLoader or model kernels.
- Packed transfer, TP8/DP2/EP and quantized updates are out of scope. Passing
  these small-topology tests would not establish their correctness.
- A5 execution and normal repository UT execution require a compatible Step5
  vLLM installation. CPU-only checks with --noconftest are limited helper tests,
  not substitutes for that environment or for hardware acceptance.
