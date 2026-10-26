# Step5 Ascend rebase reproduction

These scripts reproduce the **basic single-instance** deployment tested on eight
Ascend950DT cards. Basic `serve.sh` does not enable PD or offload.

The separate [PD_PREVIEW.md](PD_PREVIEW.md) packages the **validated normal pull
path**. It is not a failure-safe production release: Mooncake NPU0.3.11.post1
may return a failed/expired READ before other native work is quiescent, allowing
premature cache reuse through current completion/DONE handling. Ordinary traffic
can hit transport errors; sending only normal requests does not avoid this risk.
The independent native-ticket fix is not included in this preview.

## Validated prepared container (2026-10-25 checkpoint)

On the development host:

```bash
ssh gpfs-ops@gpu-a950dt-0002.lgcm.sh.istep.fun
sudo -n /opt/basemind/containerd/bin/ctr -n step5-fp8-review-20261014 \
  tasks exec --exec-id inspect-$(date +%s) step5-pd-fresh-20261025 bash
```

At that checkpoint the fresh container's editable installations pointed to:

- `/home/liuyi/step5-rebase/fresh-20261025/vllm` (official vLLM main + stepcast plugin)
- `/home/liuyi/step5-rebase/fresh-20261025/vllm-ascend` (this source + PD preview)

The original container/artifact trees were retained. These paths describe a
validation checkpoint, not a guarantee of current health or process ownership.

Before restarting, inspect processes and the port. **Do not start a second server
on the same cards or kill another container's workload.** Run the server script
inside the prepared container, with an absolute script path:

```bash
mkdir -p /work/step5-rebase-validation
bash /path/to/examples/step5/serve.sh \
  > /work/step5-rebase-validation/serve.log 2>&1
```

The script runs in the foreground and forwards signals through `exec`. The caller
owns process supervision. `MODEL` defaults to `/work/model-target`, the verified
complete model directory (273 safetensor shards plus config/tokenizer).
`PYTHON_BIN` defaults to `/usr/local/python3.12.13/bin/python3`.
Both can be overridden explicitly; `MODEL` must be an **absolute container path**.
Logs are not sent to a remote service.

## New container

Run `bash examples/step5/start_container.sh` **on the NPU host**, not inside a
container. Defaults match the above host/image; override `CTR`, `NS`, `CID`,
`IMAGE`, `WORK`, `SOURCE_ROOT`, and `WEIGHTS` as needed.

- Requires already available image, host driver/devices, sudo access and mounts.
- Refuses to replace any existing container; never kills/removes one.
- Uses privileged host-network access, as required by this tested development
  environment. This is not a general production isolation configuration.
- Mounts the checkpoint read-only and the prepared host source/work directories.
- Uses a 16-GiB shared-memory mount, rather than relying on the tiny image default.

**Model-directory prerequisite:** the new container mounts `WEIGHTS` at
`/models/step5-w8a8-mxfp8`; it does not create the prepared `/work/model-target`.
If that mount already contains the complete checkpoint including config and
all tokenizer files, run `MODEL=/models/step5-w8a8-mxfp8 bash /path/to/serve.sh`.
Otherwise prepare a complete model directory first and set `MODEL` to its
absolute container path. Package installation alone does not supply missing
model/tokenizer files.

**Creating the container does not install the rebase.** The base image contains
old packages. Follow the repository `Dockerfile.a5` build/install sequence before
serving: install the matching official vLLM source, this Ascend source (including
custom kernels for `SOC_VERSION=ascend950dt_9582`), and `stepcast_plugin` from the
vLLM source tree. Use pip/editable installation, not copying packages over
site-packages. Archive exports without `.git` require explicit
`SETUPTOOLS_SCM_PRETEND_VERSION`. The checkpoint itself is not part of this repo.

Inside the image, source the actual CANN 9.2 environment; `serve.sh` uses
`/opt/step5/bin/step5-env.sh` and includes torch/torch_npu library directories.
Run Python from `/tmp` to avoid source-directory import shadowing. When syncing
Python files to an existing installation, preserve `_build_info.py` and compiled
`_cann_ops_custom` artifacts. The model's FP8 switch is **STEP5**, not STEP4.

## Serving checks

```bash
curl -f http://127.0.0.1:19082/v1/models
curl -f http://127.0.0.1:19082/health
curl -f -H 'Content-Type: application/json' \
  -d '{"model":"step5","messages":[{"role":"user","content":"法国的首都是哪里？"}],"max_tokens":96,"temperature":0}' \
  http://127.0.0.1:19082/v1/chat/completions
```

The validated configuration is TP8/EP8, bfloat16 + Ascend quantization,
`max-model-len=262144`, MTP=3 and `FULL_DECODE_ONLY` with capture size 4.
The tested long prompt was **2135 tokens**, not a full 262144-token stress test.

CPU tests (use a Python environment with torch/pytest):

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 VLLM_SOURCE_ROOT=/path/to/vllm \
  pytest --confcutdir=tests/ut/models/step5 tests/ut/models/step5/test_step5_summary_contract.py -q
TORCH_DEVICE_BACKEND_AUTOLOAD=0 VLLM_SOURCE_ROOT=/path/to/vllm \
  pytest --confcutdir=tests/ut/worker/v2 tests/ut/worker/v2/test_step5_kv_contract.py -q
```

## Step5 KV transfer capability boundary (normal-path preview)

The Ascend Step5 language-model constructor validates active KV transfer
configurations and rejects active `kv_both`: Mooncake pull requires an exclusive
`kv_producer` or `kv_consumer` role.
Only the registered Ascend `MooncakeConnectorV2` and `MooncakePullConnector`
implementations are accepted; legacy/push/hybrid/Store/Nixl/MultiConnector names
and custom connector module overrides are rejected for this slice.

The current boundary is matching P/D topology and cache descriptors, with
PP/PCP/DCP=1, speculative decoding disabled (MTP off), prefix caching disabled,
and a text-only model configuration. P/D must use the same model/checkpoint,
dtype/quantization, TP/EP and logical block size. The connector checks topology
and cache geometry/group correspondence, **not checkpoint identity**. With no
active transfer instance, this guard does not change basic inference behavior.

The accepted preview additionally requires **MRv2**, with KVPP, sparse offload,
HiSparse and alternative allocation paths disabled. Those requirements are not
all enforced by the constructor guard: a successful guard check alone is not
normal-path acceptance. Use the compact homogeneous geometry and launch profile
in `PD_PREVIEW.md`.

This is an implemented protocol boundary, **not full-model online PD certification**.
CPU checks and the small NPU fixtures cover main KV, independent summary/ring,
shared allocation/registration and normal completion. A separate ten-layer
checkpoint view has also passed TP4 P + TP4 D deterministic request comparisons;
see `PD_CPU_VALIDATION.md` for the exact evidence and limitations. Full-checkpoint
simultaneous TP8 instances still require another eight-card resource group.
Existing `serve.sh` remains basic-only: its MTP=3 defaults are not a PD launch
configuration. Offload/MTP-PD and error-path transport quiescence remain open.

For the validated pull handoff, P generates one token. Append that token to D's
input, ask D for the remaining generation budget, and return P's token followed
by D's tokens. Only reuse it when P processed the complete original prompt;
it is not valid for a producer that intentionally truncated its prompt.
The sampled token has **not** yet had its KV computed on P. Step5 therefore
caps remote matched-token credit at `remote_num_prompt_tokens`, even when D's
input also contains the sampled token. This avoids re-sampling the last prompt
token with a different MoE batch shape, which produced measurable logprob
variation in the reduced-model experiment. No numerical tolerance was relaxed.
The checkpoint's first-token tests do not certify general EOS/stop/streaming
proxy behavior.

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 VLLM_SOURCE_ROOT=/path/to/vllm \
  python3 -m pytest --confcutdir=tests/ut/kv_offload/mooncake_v2 \
  tests/ut/kv_offload/mooncake_v2/test_step5_transfer_contract.py -q
```

See `PD_CPU_VALIDATION.md` for the guard/lifecycle CPU checkpoint and remaining
runtime gaps. Step5 producer release waits for delivered scheduler receive-completion
notification; timeouts log once and retain blocks rather than reclaiming without
DONE. Lost DONEs can pin cache capacity indefinitely. No remote cancel-ACK or
bounded safe reclamation protocol is implemented. Non-Step5 timeout behavior is
unchanged. Cancelled D requests retain blocks until receive completion under
the tested upstream scheduler hooks. These checks do not replace runtime
stream-ordering and deferred-free validation.

See `VALIDATION.md` for the basic results and `PD_PREVIEW.md` /
`PD_CPU_VALIDATION.md` for the normal-path checkpoint. General failed-transfer
reclamation, full-checkpoint PD acceptance, offload and MTP-PD remain follow-up
work; API route names and the preview commit do not certify them.
