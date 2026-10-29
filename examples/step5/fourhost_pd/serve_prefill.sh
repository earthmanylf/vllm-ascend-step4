#!/usr/bin/env bash
set -eo pipefail
node=${1:?node rank 0 or 1}
case "$node" in
 0) ip=10.208.128.28; extra=(--api-server-count 1) ;;
 1) ip=10.208.128.31; extra=(--headless) ;;
 *) exit 2 ;;
esac
source /opt/step5/bin/step5-env.sh
SP=/usr/local/python3.12.13/lib/python3.12/site-packages
export PYTHONPATH="/work/run/runtime-047${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$SP/torch/lib:$SP/torch_npu/lib:$LD_LIBRARY_PATH"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 SOC_VERSION=ascend950dt_9582
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ASCEND_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_USE_V2_MODEL_RUNNER=1 VLLM_ASCEND_ENABLE_STEP5_FP8=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_HOST_IP="$ip" HCCL_IF_IP="$ip" OMP_NUM_THREADS=1
export VLLM_PLUGINS=ascend,ascend_model,ascend_kv_connector,ascend_model_loader,ascend_service_profiling,stepcast_platform,stepcast_rs
export HCCL_OP_EXPANSION_MODE=CCU_MS PYTORCH_NPU_ALLOC_CONF=expandable_segments:True TASK_QUEUE_ENABLE=0
export HCCL_BUFFSIZE=1024 HCCL_BUFFSIZE_EP=4096 HCCL_EXEC_TIMEOUT=240 HCCL_CONNECT_TIMEOUT=240 HCCL_IF_BASE_PORT=53000
export GLOO_SOCKET_IFNAME=bond4 TP_SOCKET_IFNAME=bond4 HCCL_SOCKET_IFNAME=bond4
export TRITON_ALL_BLOCKS_PARALLEL=1 PYTHONUNBUFFERED=1 VLLM_USE_FASTOKENS=1 VLLM_STEP4_ENABLE_FA3=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export VLLM_PP_LAYER_PARTITION=24,25,25,18
unset VLLM_ASCEND_ENABLE_FLASHCOMM1
ulimit -l unlimited
cd /tmp
exec python3 -c 'from vllm.entrypoints.cli.main import main; main()' serve \
 --model /work/model-target --trust-remote-code --served-model-name step \
 --host "$ip" --port 19149 --quantization ascend --dtype bfloat16 --load-format safetensors \
 --model-loader-extra-config '{"enable_weights_track":true}' --safetensors-load-strategy prefetch \
 --tensor-parallel-size 4 --pipeline-parallel-size 4 --nnodes 2 --node-rank "$node" \
 --master-addr 10.208.128.28 --master-port 29649 \
 --enable-expert-parallel --all2all-backend allgather_reducescatter \
 --distributed-executor-backend mp \
 --max-model-len 65536 --max-num-seqs 32 --max-num-batched-tokens 2048 \
 --gpu-memory-utilization 0.90 --kv-cache-memory-bytes 2147483648 \
 --enable-prefix-caching --async-scheduling \
 --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","max_cudagraph_capture_size":512}' \
 --additional-config '{"enable_cpu_binding":true,"enable_shared_expert_dp":true,"multistream_overlap_shared_expert":true,"enable_force_eplb":false}' \
 --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
 --kv-transfer-config '{"kv_connector":"MooncakeConnectorV2","kv_role":"kv_producer","engine_id":"pp4-P-049","kv_port":19490,"kv_load_failure_policy":"fail"}' "${extra[@]}"
