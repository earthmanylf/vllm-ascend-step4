#!/usr/bin/env bash
# Launch one node of the Step5 PP4/TP4 service.
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
# Two ranks' worth of nodes, and PP4/TP4 is what makes it a pipeline prefill
# cluster rather than the decode topology. A 128K request plus headroom must fit.
[[ "$PP_SIZE" == 4 && "$TP_SIZE" == 4 && "$DP_SIZE" == 1 && "$MAX_MODEL_LEN" -ge 133120 ]] ||
  { echo "Deployment must preserve PP4/TP4/DP1 and 128K+2048 capacity." >&2; exit 2; }
[[ "${#NODE_IPS[@]}" == 2 ]] || { echo "PP4/TP4 spans exactly two nodes." >&2; exit 2; }
: "${NODE_ID:?}" "${LOCAL_IP:?}" "${COMM_IFACE:?}" "${PP_LAYER_PARTITION:?}"
[[ "$NODE_ID" =~ ^[01]$ && "$LOCAL_IP" == "${NODE_IPS[NODE_ID]}" ]] || exit 2
# Stage 0 hosts PP ranks 0-1 and stage 1 hosts PP ranks 2-3: one node per half.

# 2. Environment variables. Preserve the venv before the SDK script runs.
unset PYTHONPATH LD_PRELOAD VLLM_SERVER_DEV_MODE ASCEND_LAUNCH_BLOCKING
set +u
if [[ -f "$RUNTIME_ENV" ]]; then source "$RUNTIME_ENV"; fi
set -u
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 TASK_QUEUE_ENABLE=0 PYTHONUNBUFFERED=1
export ASCEND_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_USE_V2_MODEL_RUNNER=1 VLLM_ASCEND_ENABLE_FA3=1 VLLM_ASCEND_ENABLE_STEP5_FP8=1
# Same readback policy as the decode recipe: the Step5 DSA/GBSA backends read KV
# lengths from device tensors, so the host-side exact-length readback is off.
# PP>1 with MTP is rejected unless the target pass keeps the exact seq_lens
# readback; 0 is only validated for single-stage pipelines. MTP readback keeps
# its default (off).
export VLLM_ASCEND_TARGET_EXACT_SEQ_LENS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_ENGINE_READY_TIMEOUT_S=1800
export VLLM_KV_CACHE_LAYOUT=BLNHC VLLM_HOST_IP="$LOCAL_IP" HCCL_IF_IP="$LOCAL_IP"
export HCCL_SOCKET_IFNAME="$COMM_IFACE" GLOO_SOCKET_IFNAME="$COMM_IFACE" TP_SOCKET_IFNAME="$COMM_IFACE"
export HCCL_EXEC_TIMEOUT=600 HCCL_CONNECT_TIMEOUT=300 HCCL_BUFFSIZE=1024
export HCCL_OP_EXPANSION_MODE=CCU_MS HCCL_IF_BASE_PORT="${HCCL_BASE_PORT:-53000}"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True TRITON_ALL_BLOCKS_PARALLEL=1
export VLLM_USE_FASTOKENS=1 VLLM_PLUGINS=ascend,ascend_model,ascend_kv_connector,ascend_model_loader
export VLLM_PP_LAYER_PARTITION
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

# Expert-parallel load balancing is a measurement knob, not an assumption:
# FORCE_EPLB=1 enables the forced eplb path so the same topology can be compared
# with and without it. One value drives the flag, so a comparison run differs in
# exactly this setting.
case "${FORCE_EPLB:-0}" in
  0) EPLB_JSON=false ;;
  1) EPLB_JSON=true ;;
  *) echo "FORCE_EPLB must be 0 or 1, got '$FORCE_EPLB'" >&2; exit 2 ;;
esac
ADDITIONAL_CONFIG_JSON="{\"enable_cpu_binding\":true,\"enable_shared_expert_dp\":true,\"enable_flashcomm1\":true,\"multistream_overlap_shared_expert\":true,\"enable_force_eplb\":$EPLB_JSON}"
echo "additional-config: $ADDITIONAL_CONFIG_JSON"

args=("$MODEL_PATH" --trust-remote-code --served-model-name step5
  --load-format safetensors --safetensors-load-strategy prefetch --quantization ascend --dtype bfloat16
  --distributed-executor-backend mp
  --tensor-parallel-size "$TP_SIZE" --pipeline-parallel-size "$PP_SIZE" --enable-expert-parallel
  --data-parallel-size "$DP_SIZE"
  --nnodes 2 --node-rank "$NODE_ID" --master-addr "$MASTER_IP" --master-port "$PP_MASTER_PORT"
  --max-model-len "$MAX_MODEL_LEN" --max-num-seqs "$MAX_NUM_SEQS" --max-num-batched-tokens "$MAX_BATCHED_TOKENS"
  --gpu-memory-utilization "$MEMORY_UTILIZATION" --block-size 128 --no-enable-prefix-caching
  --speculative-config "$SPECULATIVE_CONFIG_JSON"
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
  --profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"${PROFILER_DIR:-/run-cluster/profiler}\",\"torch_profiler_with_stack\":false,\"torch_profiler_with_memory\":false,\"ignore_frontend\":true,\"capture_torch_profiler\":false}"
  --additional-config "$ADDITIONAL_CONFIG_JSON"
)
if [[ "$NODE_ID" == 0 ]]; then
  args+=(--api-server-count 1 --host "$LOCAL_IP" --port "$API_PORT")
else
  args+=(--headless)
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
