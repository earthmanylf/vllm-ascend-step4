# Step5 homogeneous pull: normal-path preview handoff

## What you can take now

This is a source checkpoint for the **validated normal PD path**, not a
failure-safe production release. Reduced-model TP4 P/TP4 D passed13 cases with
104 output IDs and all generation logprobs exactly equal to monolithic TP4.
The basic full-checkpoint TP8/EP8 service also passed health/models/chat and
2135-token chunked prefill in the fresh environment.

**Known release blocker:** Mooncake NPU0.3.11.post1 can return failed/expired
READ before every native operation is quiescent; current completion/DONE may
then permit cache reuse. This can occur during ordinary traffic. Do not assume
that sending only normal requests avoids the risk. General production rollout
is not certified. Keep this preview separate from the continuing native-ticket
failure-safety work; no experimental native library is required or included.

## Matched source and build prerequisites

### Integration revision note

The exact 13-case hardware-tested PD checkpoint is commit `09011d71a`
(including its CANN-first parent `d32a822e9`). The later merge preserves the
parallel BLNHC/capacity work and adds explicit worker-config context for layout
negotiation. Integrated CPU checks pass: KV/layout RPC39 tests +24 subtests,
protocol134 tests, DSA/summary/graph82 tests +4 subtests (5 skipped).
Post-merge NPU revalidation is pending: the validation host's SSH timed out on
both connection attempts. Historical hardware results do not certify the merged
revision. For the exact previously tested snapshot, use `09011d71a`; it remains
in this branch's history. The known native failure-safety limitation applies to
both snapshots.

## Matched source and build prerequisites

Use vLLM commit `ca5e43ddcd5b04c91e3de451702eecc1632faea8` (official main
`8407bc0bb6` plus stepcast_plugin), this Ascend source, and the matching Ascend
custom kernels. Do not install an unrelated latest vLLM or copy .py files over
site-packages. The validated backend uses Torch2.10.0+cpu, torch-npu2.10.0.post4,
Triton-Ascend3.6.0 development build and CANN9.2.0-beta.2; dependency metadata
checks have known differences and are not claimed fully green.

For a fresh CANN9.2 kernel build, the Catlass gitlink is
`41bf90da655bba3c66d0acd7e00abe33960ecfd6`. The tested compiler needs the10 static
SIMD member declarations in [catlass-cann92.patch](catlass-cann92.patch).
This patch is provided separately; the submodule commit pointer is not changed.
On a clean matching submodule, check then apply it before compiling:

```bash
PATCH=$(realpath examples/step5/catlass-cann92.patch)
git -C csrc/third_party/catlass apply --unidiff-zero --check "$PATCH"
git -C csrc/third_party/catlass apply --unidiff-zero "$PATCH"
```

Do not reapply on an already-patched tree. Existing validated installations must
preserve `_build_info.py`, native extension and `_cann_ops_custom` artifacts.
Follow `Dockerfile.a5`/the basic README installation sequence for vLLM, Ascend
and stepcast_plugin. Never replace CANN/driver libraries individually.

## Preview deployment profile

Use identical text-only model/config/tokenizer/quantization on P and D. A
separate10-layer view was validated; it does not certify original92-layer dual
TP8 deployment. MRv2 must be enabled; prefix caching, MTP/speculation, KVPP,
sparse offload and HiSparse must be disabled. PP/PCP/DCP=1; P/D TP/EP and cache
geometry must match. Passing the constructor guard alone is not this acceptance.

The merged parallel-development work preserves **unpadded BLNHC for local V2
inference**. Active PD selects the already validated compact NHD/shared-slab
profile, including padded summary pages and its alignment reserve. The selector
is a work/profile boundary, not a claim that BLNHC PD is fundamentally unsupported.
PD on the new unpadded global-tile layout needs separate transport/capacity validation.

The following commands run **inside the matched prepared environment**. Set
`MODEL` to the complete text-only model directory. First confirm the devices
and ports are free; these commands never stop existing workloads.

```bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
source /opt/step5/bin/step5-env.sh
PYTHON_BIN=/usr/local/python3.12.13/bin/python3
MODEL=/absolute/path/to/text-only-model
SP=$($PYTHON_BIN -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
export LD_LIBRARY_PATH="$SP/torch/lib:$SP/torch_npu/lib:$LD_LIBRARY_PATH"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 SOC_VERSION=ascend950dt_9582
export VLLM_USE_V2_MODEL_RUNNER=1 VLLM_ASCEND_ENABLE_STEP5_FP8=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_HOST_IP=127.0.0.1
export VLLM_PLUGINS=ascend,ascend_model,ascend_kv_connector,ascend_model_loader,ascend_service_profiling,stepcast_platform,stepcast_rs
export OMP_NUM_THREADS=1 TASK_QUEUE_ENABLE=1
ulimit -l unlimited
COMMON=(--model "$MODEL" --trust-remote-code --served-model-name step5-pd
  --host 127.0.0.1 --load-format safetensors --quantization ascend --dtype bfloat16
  --tensor-parallel-size 4 --enable-expert-parallel
  --max-model-len 4096 --max-num-seqs 1 --max-num-batched-tokens 512
  --kv-cache-memory-bytes 1073741824 --enforce-eager --no-enable-prefix-caching
  --model-loader-extra-config '{"enable_weights_track":true}' --enable-prompt-tokens-details)
# Run each role in its own supervised process/shell; do not overlap with TP8.
ASCEND_VISIBLE_DEVICES=0,1,2,3 ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
  "$PYTHON_BIN" -m vllm.entrypoints.openai.api_server "${COMMON[@]}" --port 19085 \
  --kv-transfer-config '{"kv_connector":"MooncakeConnectorV2","kv_role":"kv_producer","engine_id":"step5-preview-P","kv_port":19210,"kv_load_failure_policy":"fail"}'
ASCEND_VISIBLE_DEVICES=4,5,6,7 ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 \
  "$PYTHON_BIN" -m vllm.entrypoints.openai.api_server "${COMMON[@]}" --port 19086 \
  --kv-transfer-config '{"kv_connector":"MooncakeConnectorV2","kv_role":"kv_consumer","engine_id":"step5-preview-D","kv_port":19220,"kv_load_failure_policy":"fail"}'
```

`kv_load_failure_policy=fail` rejects failed generation; it does **not** repair
the native failure-quiescence/reuse issue. Loopback is the single-host preview;
multihost host/IP/network routing remains operator-owned and unvalidated here.

## Request handoff (required, not an interchangeable proxy detail)

1. Send the **entire original prompt** to P with remote-decode enabled and one
   output token (`do_remote_decode=true`, `do_remote_prefill=false`).
2. Preserve P's sampled token and its returned `kv_transfer_params`.
3. Append that token to D's prompt, forward the transfer params, and request the
   remaining output-token budget. Remote credit is only P's original prompt
   count: its sampled token has not yet had KV computed on P.
4. Return P's first token/logprob followed by D's. Do not replay the final
   original prompt token or silently perform independent D prefill.

The validation used non-streaming `/v1/completions`, fixed generation budgets
and explicit token IDs/logprobs. Unified proxy EOS/stop/streaming composition is
not certified or implemented by this document. See
[PD_CPU_VALIDATION.md](PD_CPU_VALIDATION.md) for tests and precise limitations.
Failure-state hardening continues after this preview checkpoint.
