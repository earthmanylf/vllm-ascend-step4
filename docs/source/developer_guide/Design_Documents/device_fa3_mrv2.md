# FA3 on the Step5 BLNHC baseline

This change is based on Ascend commit
`8458ec69de8b59c87128c7018713e9d6d1485d21`. The Step5 DSA backend,
real summary pages, Ring payload, grouping and original upstream allocator
are retained. The layer-compact review checkpoint is on a separate branch.

## Enable

```bash
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_ASCEND_ENABLE_FA3=1
export VLLM_KV_CACHE_LAYOUT=BLNHC
```

The FA3 flag defaults to `0`; its legacy Step4 name is a compatibility alias.
Sparse DSA selection is not replaced with dense FA3. Local Step5 DSA and FA3
both declare BLNHC. There is no automatic fallback to layer-compact pages.
FA3 registers its unique backend name through vLLM's `register_backend`
extension point. Returning an unregistered display name is insufficient:
the real Attention initializer resolves that name back through the enum.

## Native ABI

`tools/step5/fa3_950/` contains the content-checked native source patch and
the compiled-symbol checker. Apply it to the exact retained native source,
then build on the target with the formal CANN 9.2 toolchain. A source hash
mismatch is an error, not permission to apply to a different version.

The device template has both boolean parameters, including the softcap
specialization. The launch also passes independent uint64 K/V block strides,
in element units. On the validated aarch64 toolchain, the requested complete
symbol ends in `ffjjmm`; neither the one-boolean specialization nor the old
`ffjj` signature certifies this ABI. The Python wrapper checks the compiled
ABI version, rather than assigning an unconditional compatibility marker.

Paged caches have dense BNHD rows, but may have gaps between blocks and
nonzero storage offsets. The native address is based on the actual physical
block stride, not `physical_block_id * logical_block_size`. The wrapper
does not make paged K/V contiguous or allocate independent replacement pools.
Main's block table and shared backing remain authoritative.

## Runner integration

After profiling or real allocation, each target/draft FA3 builder binds its
own actual views. Rebinding invalidates its metadata registrations. FULL
replay uses capture-owned reader and writer-slot addresses; padded writer
slots become `-1`. MTP forwards the actual FULL descriptor rather than
guessing from initial capture sizes. FIA's updatable-task path is not used
for device-tiling FA3.

The native stride patch is separate from the baseline model's summary/Ring
algorithms. This change does not import the later layer-compact storage
profile or the rest of the review candidate's model changes.

## Validation boundary

CPU contract tests do not prove NPU correctness. A compiled matching symbol
does not prove execution. Validate the final source and native library on
BLNHC planner views, including nonzero offsets, last-block IDs, independent
K/V strides and FULL replay. Complete Step5/MTP precision and four-node
128K service acceptance remain separate gates.
