# Step5 decode performance on Ascend

Step5 decode is measured with the same workload throughout: 128K context,
96 requests (`bs96`, 12 per engine on DP8/TP4/EP32), 4K output, MTP with
`num_speculative_tokens=3`, prefills drained before the steady window is
sampled. Short-context timing is not evidence for 128K service latency.

## Sampled sequence lengths

`VLLM_ASCEND_TARGET_EXACT_SEQ_LENS` and `VLLM_ASCEND_MTP_EXACT_SEQ_LENS`
control whether the target pass and the drafter read the post-rejection
sequence lengths back to the host (`1`, the default) or keep upstream's
optimistic CPU upper bound (`0`).

A readback blocks on the whole target forward that was just submitted, so the
host cannot run a step ahead. On 128K/bs96 it cost 11.6 ms per decode step
(wall 48.1 -> 36.6 ms, device idle 10.1 -> 0.6 ms) with no change in device
work. Only disable it for backends that read KV lengths from device tensors:
the Step5 DSA/GBSA path does, while the ordinary FIA backends and PP>1 still
need the exact host list (the runner rejects that combination instead of
running it unverified).

## Sparse indexer summary table

BLNHC packs every layer page into one model-wide block tile, so a 4 KiB
summary page is followed by the rest of the tile. The fused
`aclnnIndexerRegionScore` only accepts a dense paged summary; handed the
strided table it gathered every visible region per layer and step (one extra
`Slice` kernel per DSA layer, 1.05 ms per step). `VLLM_ASCEND_STEP5_SUMMARY_DENSE`
(default `1`) keeps a dense mirror that both the summary kernels and the
indexer use, and publishes it as the layer's cache so the runner's block
zeroing reaches it. It is disabled automatically when a KV transfer instance
owns the pages, and set it to `0` to compare against the paged table.

## Packed projection rows

The fused qkv+indexer and qkv+gate projections already lay their outputs out
consecutively in one row. The norm+rope kernels take that row with an explicit
`row_stride`, so neither path concatenates or re-copies its inputs.

## Fused SwiGLU

The shared-expert `SwigluStepAndMul` CustomOp dispatches to a fused Triton
kernel. Routed-expert activation uses device-side valid row counts rather than
processing the entire MC2 capacity buffer. Cumulative counts are converted on
device; sparse type-2 group tables and installations without Triton safely
compute all rows. Padding output is unspecified and must not be consumed as
valid tokens. No device-to-host count read is introduced.

## Validation

- Run the activation, MoE MLP, DSA/summary, and Ascend configuration unit tests.
- Run the SwiGLU device tests, including count updates during FULL graph replay.
- Keep a long-context greedy canary (a 20K+ token prompt) and diff its output
  against the paged-summary path: a summary window that reads a previous
  owner's regions changes long generations only, so short prompts do not
  catch it.
- Compare the same workload with all prefills drained, all 96 requests running,
  and no requests waiting. Record each engine's running request count.
- Record mean acceptance length and distinguish real rejection sampling from a
  synthetic performance-only acceptance setting; a synthetic acceptance rate
  keeps the draft path's cost identical but changes how often rejections occur.
- Check reduce-scatter/all-reduce/all-gather counts and fused activation counts.
  Do not expect all `Silu` or `Clamp` kernels to disappear: other model paths
  can still use those operations.
