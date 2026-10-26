#!/usr/bin/env bash
set -eo pipefail
node=${1:?node index 0..3}
case "$node" in
  0) ip=172.27.9.105; start=0 ;;
  1) ip=172.27.9.106; start=2 ;;
  2) ip=172.27.9.107; start=4 ;;
  3) ip=172.27.9.108; start=6 ;;
  *) exit 2 ;;
esac
unset PYTHONPATH LD_PRELOAD VLLM_SERVER_DEV_MODE ASCEND_LAUNCH_BLOCKING
source /opt/upgrade/cann92-env.sh
export VIRTUAL_ENV=/opt/upgrade/.venv PATH=/opt/upgrade/.venv/bin:$PATH
PYTHON_BIN=/opt/upgrade/.venv/bin/python
SP_BASE=/usr/local/python3.12.13/lib/python3.12/site-packages
export LD_LIBRARY_PATH="$SP_BASE/torch/lib:$SP_BASE/torch_npu/lib:$LD_LIBRARY_PATH"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 SOC_VERSION=ascend950dt_9582
export OMP_NUM_THREADS=1 TASK_QUEUE_ENABLE=1
export ASCEND_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_USE_V2_MODEL_RUNNER=1 VLLM_ASCEND_ENABLE_STEP5_FP8=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ENGINE_READY_TIMEOUT_S=1800
export VLLM_HOST_IP=$ip HCCL_IF_IP=$ip
export HCCL_SOCKET_IFNAME=data0.172 GLOO_SOCKET_IFNAME=data0.172 TP_SOCKET_IFNAME=data0.172
export HCCL_EXEC_TIMEOUT=120 HCCL_CONNECT_TIMEOUT=120 HCCL_BUFFSIZE=1024
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export VLLM_PLUGINS=ascend,ascend_model,ascend_kv_connector,ascend_model_loader,ascend_service_profiling
draft_eager="${MTP_DRAFT_EAGER:-false}"
[[ "$draft_eager" == true || "$draft_eager" == false ]] || exit 2
ulimit -l unlimited
args=(/model/target --trust-remote-code --served-model-name step5
  --load-format safetensors --quantization ascend --dtype bfloat16
  --distributed-executor-backend mp --data-parallel-backend mp
  --tensor-parallel-size 4 --pipeline-parallel-size 1 --enable-expert-parallel
  --data-parallel-size 8 --data-parallel-size-local 2
  --data-parallel-address 172.27.9.105 --data-parallel-rpc-port 13345
  --max-model-len 1048576 --max-num-seqs 32 --max-num-batched-tokens 16384
  --gpu-memory-utilization 0.90
  --compilation-config '{"mode":3,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[4,8,16,32,64,128,256,512]}'
  --limit-mm-per-prompt '{"image":0}' --no-enable-prefix-caching
  --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":3,\"enforce_eager\":$draft_eager}")
if [[ "$node" == 0 ]]; then
  args+=(--api-server-count 1 --host 127.0.0.1 --port 19082)
else
  args+=(--headless --data-parallel-start-rank "$start")
fi
cd /tmp
exec "$PYTHON_BIN" -m vllm.entrypoints.cli.main serve "${args[@]}"
