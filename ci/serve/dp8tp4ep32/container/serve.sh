#!/usr/bin/env bash
# Launch one rank group of the Step5 DP8/TP4/EP32 service.
#
# Runs both inside the container image and directly on a bare-metal host; the
# only difference is the defaults below. Everything the two modes need to vary
# is an env var, so the same file stays authoritative for both.
#
#   MODE=host|container      (default: container)
#   LAUNCH_DIR               config dir (default: /launch, host: $RUN_DIR/launch)
#   MODEL_PATH               weights   (default: /model/target, host: $MODEL_PATH_HOST)
#   PYTHON_BIN               interpreter (default: /opt/upgrade/.venv/bin/python)
#   RUNTIME_ENV              SDK env script, sourced when present
#
# The image is used read-only: this script never writes into site-packages and
# never patches the installed tree. Whatever the image ships is what runs.
set -euo pipefail

MODE="${MODE:-container}"
LAUNCH_DIR="${LAUNCH_DIR:-/launch}"
MODEL_PATH="${MODEL_PATH:-/model/target}"
PYTHON_BIN="${PYTHON_BIN:-/opt/upgrade/.venv/bin/python}"
RUNTIME_ENV="${RUNTIME_ENV:-/opt/upgrade/runtime-env.sh}"

# 1. Machine-related variables.
MACHINE_CONFIG_FILE="${MACHINE_CONFIG_FILE:-$LAUNCH_DIR/cluster.env}"
[[ -f "$MACHINE_CONFIG_FILE" ]] || { echo "Missing $MACHINE_CONFIG_FILE" >&2; exit 2; }
source "$MACHINE_CONFIG_FILE"
[[ "$DP_SIZE" == 8 && "$TP_SIZE" == 4 && "$LOCAL_DP_SIZE" == 2 && "$MAX_MODEL_LEN" -ge 133120 ]] ||
  { echo "Deployment must preserve DP8/TP4/EP32 and 128K+2048 capacity." >&2; exit 2; }
: "${NODE_ID:?}" "${LOCAL_IP:?}" "${COMM_IFACE:?}"
[[ "$NODE_ID" =~ ^[0-3]$ && "$LOCAL_IP" == "${NODE_IPS[NODE_ID]}" ]] || exit 2
DP_START_RANK=$((NODE_ID * LOCAL_DP_SIZE))

# 2. Environment variables. Preserve the venv before the SDK script runs.
unset PYTHONPATH LD_PRELOAD VLLM_SERVER_DEV_MODE ASCEND_LAUNCH_BLOCKING
set +u
if [[ -f "$RUNTIME_ENV" ]]; then source "$RUNTIME_ENV"; fi
set -u
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 TASK_QUEUE_ENABLE=0 PYTHONUNBUFFERED=1
export ASCEND_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_USE_V2_MODEL_RUNNER=1 VLLM_ASCEND_ENABLE_FA3=1 VLLM_ASCEND_ENABLE_STEP5_FP8=1
# Keep the upstream optimistic host seq_lens for the target pass. Reading the
# post-rejection counts back blocks on the whole target forward (one full
# device pass per step) and is only needed by backends that read host lengths.
export VLLM_ASCEND_TARGET_EXACT_SEQ_LENS=0
export VLLM_ASCEND_MTP_EXACT_SEQ_LENS=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_ENGINE_READY_TIMEOUT_S=1800
export VLLM_KV_CACHE_LAYOUT=BLNHC VLLM_HOST_IP="$LOCAL_IP" HCCL_IF_IP="$LOCAL_IP"
export HCCL_SOCKET_IFNAME="$COMM_IFACE" GLOO_SOCKET_IFNAME="$COMM_IFACE" TP_SOCKET_IFNAME="$COMM_IFACE"
export HCCL_EXEC_TIMEOUT=240 HCCL_CONNECT_TIMEOUT=240 HCCL_BUFFSIZE=1024
export HCCL_OP_EXPANSION_MODE=CCU_MS HCCL_IF_BASE_PORT="${HCCL_BASE_PORT:-50000}"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True TRITON_ALL_BLOCKS_PARALLEL=1
export VLLM_USE_FASTOKENS=1 VLLM_PLUGINS=ascend,ascend_model,ascend_kv_connector,ascend_model_loader
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$LAUNCH_DIR/../torch-extensions}"
mkdir -p "$TORCH_EXTENSIONS_DIR"
# Cold JIT cache, matching the reference profiling environment. This only
# clears runtime compile artefacts (~/.cache, ~/.triton) so the run cannot
# inherit tiling from an older build; it does not touch installed packages or
# anything else the image ships. Set CLEAN_JIT_CACHE=0 to reuse them.
if [[ "${CLEAN_JIT_CACHE:-1}" == "1" ]]; then
  rm -rf /root/.cache /root/.triton/cache
fi
ulimit -l unlimited || true

# 3. vLLM command. FULL graph only; synthetic acceptance is for perf runs.
SPEC_CFG="$(mktemp)"
python3 - "$LAUNCH_DIR/speculative_config.json" "$SPEC_CFG" <<'PY'
import json, sys
from pathlib import Path
cfg = json.loads(Path(sys.argv[1]).read_text())
rate = cfg.pop("synthetic_acceptance_rate", None)
if rate is not None:
    drafts = cfg["num_speculative_tokens"]
    cfg["synthetic_acceptance_length"] = 1 + drafts * rate
Path(sys.argv[2]).write_text(json.dumps(cfg, separators=(",", ":")))
PY
SPECULATIVE_CONFIG_JSON="$(cat "$SPEC_CFG")"
rm -f "$SPEC_CFG"

args=("$MODEL_PATH" --trust-remote-code --served-model-name step5
  --load-format safetensors --safetensors-load-strategy prefetch --quantization ascend --dtype bfloat16
  --distributed-executor-backend mp --data-parallel-backend mp
  --tensor-parallel-size "$TP_SIZE" --pipeline-parallel-size 1 --enable-expert-parallel
  --data-parallel-size "$DP_SIZE" --data-parallel-size-local "$LOCAL_DP_SIZE"
  --data-parallel-address "$MASTER_IP" --data-parallel-rpc-port "$DP_RPC_PORT"
  --max-model-len "$MAX_MODEL_LEN" --max-num-seqs "$MAX_NUM_SEQS" --max-num-batched-tokens "$MAX_BATCHED_TOKENS"
  --gpu-memory-utilization "$MEMORY_UTILIZATION" --block-size 128 --no-enable-prefix-caching
  --speculative-config "$SPECULATIVE_CONFIG_JSON"
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","max_cudagraph_capture_size":512}'
  --profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"${PROFILER_DIR:-/run-cluster/profiler}\",\"torch_profiler_with_stack\":false,\"torch_profiler_with_memory\":false,\"ignore_frontend\":true,\"capture_torch_profiler\":false}"
  --additional-config '{"enable_cpu_binding":true,"enable_shared_expert_dp":true,"enable_flashcomm1":true,"multistream_overlap_shared_expert":true,"enable_force_eplb":false}'
)
if [[ "$NODE_ID" == 0 ]]; then
  args+=(--api-server-count 1 --host "$LOCAL_IP" --port "$API_PORT")
else
  args+=(--headless --data-parallel-start-rank "$DP_START_RANK")
fi

cd /tmp
# Prefer the `vllm` entry point when it exists (bare-metal installs), otherwise
# go through the interpreter that owns the package (container .venv).
if [[ "$MODE" == "host" ]] && command -v vllm >/dev/null 2>&1; then
  printf 'exec vllm serve '; printf '%q ' "${args[@]}"; printf '\n'
  exec vllm serve "${args[@]}"
fi
printf 'exec %q -m vllm.entrypoints.cli.main serve ' "$PYTHON_BIN"
printf '%q ' "${args[@]}"; printf '\n'
exec "$PYTHON_BIN" -m vllm.entrypoints.cli.main serve "${args[@]}"
