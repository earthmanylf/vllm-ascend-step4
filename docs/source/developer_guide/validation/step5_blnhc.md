# Step5 V2 BLNHC candidate

This candidate retains the vLLM main BLNHC planner, manager grouping,
allocation and block/layer descriptor offsets. Summary pages retain their
real 4096-byte payload, not a 98304-byte padded page.

CANN FIA and GBSA PA_BBND only allow non-contiguity between blocks, not
between token rows. The Ascend adapter therefore exposes dense K and V
planes inside each main layer page as zero-copy views. This is a native
content adapter, not CUDA interleaved K|V byte compatibility. KV transport,
offload, disabled hybrid management and PP/PCP/DCP other than one are rejected.
The DSA backend publishes GBSA's actual block-stride alignment to the
upstream planner. No private capacity formula or second backing is used.

Independent worktree: step5-preview-tailring-blnhc-20261023. The original
step5-preview-tailring-npu branch and dual-backing experiment remain intact.

## Evidence so far

- Seven CPU storage tests exercise the actual upstream planner/allocator,
  descriptor rejection, null-ID isolation, COW and shrink re-planning.
- Twenty-one retained storage contracts pass.
- One-NPU scatter/FIA/GBSA planar views match dense references. The original
  token-interleaved FIA view fails with a precise dim-1 contiguity error;
  GBSA additionally rejects unaligned dim-0 stride.
- Real small-page summary close, verify re-run across position128, scorer,
  tuple-aware zeroer, chained COW and summary graph replay pass.

These gates are not full-model acceptance. All model/MTP FULL and BS128
resident tests must be repeated on the new layout. Historical numerical or
benchmark results cannot certify this candidate. A public release-note
fragment will require the human-assigned PR number; no PR is opened here.
