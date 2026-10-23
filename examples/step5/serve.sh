#!/usr/bin/env bash
# Run inside the prepared rebase container. stdout/stderr belong to the caller.
set -eo pipefail
unset PYTHONPATH LD_PRELOAD VLLM_SERVER_DEV_MODE ASCEND_LAUNCH_BLOCKING
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
source /opt/step5/bin/step5-env.sh
PYTHON_BIN=${PYTHON_BIN:-/usr/local/python3.12.13/bin/python3}
MODEL=${MODEL:-/work/model-target}
[[ "$MODEL" == /* ]] || { echo 'MODEL must be an absolute container path' >&2; exit 1; }
SP=$("$PYTHON_BIN" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
export LD_LIBRARY_PATH="$SP/torch/lib:$SP/torch_npu/lib:$LD_LIBRARY_PATH"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 SOC_VERSION=ascend950dt_9582
export OMP_NUM_THREADS=1 TASK_QUEUE_ENABLE=1
export ASCEND_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_USE_V2_MODEL_RUNNER=1 VLLM_ASCEND_ENABLE_STEP5_FP8=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_HOST_IP=127.0.0.1
export VLLM_PLUGINS=ascend,ascend_model,ascend_kv_connector,ascend_model_loader,ascend_service_profiling,stepcast_platform,stepcast_rs
export HCCL_EXEC_TIMEOUT=120 HCCL_CONNECT_TIMEOUT=120
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
[[ -f "$MODEL/config.json" ]] || { echo "Missing model config in $MODEL" >&2; exit 1; }
ulimit -l unlimited
cd /tmp
exec "$PYTHON_BIN" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --trust-remote-code --served-model-name step5 \
    --host 127.0.0.1 --port 19082 \
    --load-format safetensors --quantization ascend --dtype bfloat16 \
    --tensor-parallel-size 8 --enable-expert-parallel \
    --max-model-len 262144 --max-num-seqs 1 --max-num-batched-tokens 1024 \
    --gpu-memory-utilization 0.98 \
    --compilation-config '{"mode":3,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[4]}' \
    --limit-mm-per-prompt '{"image":0}' --no-enable-prefix-caching \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3,"enforce_eager":false}'
