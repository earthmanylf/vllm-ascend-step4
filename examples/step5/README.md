# Step5 Ascend rebase reproduction

These scripts reproduce the **basic single-instance** deployment tested on eight
Ascend950DT cards. They do not enable PD or offload.

## Existing prepared container (recommended)

On the development host:

```bash
ssh gpfs-ops@gpu-a950dt-0002.lgcm.sh.istep.fun
sudo -n /opt/basemind/containerd/bin/ctr -n step5-fp8-review-20261014 \
  tasks exec --exec-id inspect-$(date +%s) step5-rebase-20261021 bash
```

The current editable installations point to:

- `/home/liuyi/step5-rebase/artifacts/vllm-rebase` (official vLLM main + stepcast plugin)
- `/home/liuyi/step5-rebase/artifacts/vllm-ascend-rebase` (this branch)

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

See `VALIDATION.md` for results and limitations. PD and offload are mandatory
follow-up work, **not** capabilities established by these scripts or existing
API route names.
