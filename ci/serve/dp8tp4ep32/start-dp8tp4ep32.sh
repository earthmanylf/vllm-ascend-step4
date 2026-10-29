#!/usr/bin/env bash
# Start the Step5 DP8/TP4/EP32 service on four nodes.
#
#   ./start-dp8tp4ep32.sh --ips 10.0.0.1,10.0.0.2,10.0.0.3,10.0.0.4
#
# Node order matters: the first IP is node0 and hosts the OpenAI-compatible API
# server as well as DP ranks 0-1. The launcher runs over SSH, so the calling host
# must reach every node as root without a password prompt (directly, or via a
# ~/.ssh/config ProxyJump entry). Nothing is written outside $RUN_DIR on the nodes.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

IPS=""
IMAGE="${IMAGE:-}"   # empty => bare metal (`vllm serve` on the node)
CONTAINER_PREFIX="${CONTAINER_PREFIX:-step5-dp8tp4ep32}"
RUN_DIR="${RUN_DIR:-/data/step5-dp8tp4ep32}"
COMM_IFACE="${COMM_IFACE:-bond4}"
MEMORY_UTILIZATION="${MEMORY_UTILIZATION:-}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-}"
READY_TIMEOUT="${READY_TIMEOUT:-1800}"
SSH_USER="${SSH_USER:-root}"
DRY_RUN="${DRY_RUN:-0}"
SPEC_CONFIG="${SPEC_CONFIG:-synthetic}"   # synthetic | standard | <path to json>
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUNTIME_ENV="${RUNTIME_ENV:-}"

usage() {
  cat <<'USAGE'
Start Step5 on four nodes as DP8 x TP4 x EP32.

Required:
  --ips A,B,C,D          Four node IPs, node0 first.

Two start modes (pick one):
  (default)              Bare metal: run `vllm serve` directly on each node.
  --image REF            Container: run the service inside REF.

Optional:
  --image REF            Container image. Omit to run `vllm serve` on the host.
  --name PREFIX          Container name prefix (default: step5-dp8tp4ep32).
  --run-dir PATH         Node-local working dir (default: /data/step5-dp8tp4ep32).
  --iface NAME           HCCL/Gloo socket interface (default: bond4).
  --util FLOAT           gpu-memory-utilization override.
  --max-batched-tokens N Override the per-step token budget.
  --ready-timeout SEC    How long to wait for /health (default: 1800).
  --spec-config WHICH    MTP speculative config: `synthetic` (default, fixed
                         synthetic acceptance rate for perf runs), `standard`
                         (real rejection sampling, no acceptance rate), or a
                         path to your own JSON.
  --python PATH          Interpreter for bare metal (default: python3).
  --runtime-env PATH     SDK env script for bare metal (default: auto-detect).
  --dry-run              Print the plan without touching any node.
  -h, --help             This message.

Environment: IMAGE, CONTAINER_PREFIX, RUN_DIR, COMM_IFACE, MEMORY_UTILIZATION,
MAX_BATCHED_TOKENS, READY_TIMEOUT, SSH_USER, DRY_RUN, SPEC_CONFIG, PYTHON_BIN,
RUNTIME_ENV.

Notes:
  * 128K-context runs with large batches need more KV than 0.85 provides. 0.91
    is the highest utilization verified to start and still admit bs=144 at
    128K; higher values make HCCL init fail (`error code is 4`).
  * The launcher always starts node0 last so the DP coordinator exists before
    the headless ranks connect.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ips) IPS="${2:?--ips needs a value}"; shift 2 ;;
    --image) IMAGE="${2:?}"; shift 2 ;;
    --name) CONTAINER_PREFIX="${2:?}"; shift 2 ;;
    --run-dir) RUN_DIR="${2:?}"; shift 2 ;;
    --iface) COMM_IFACE="${2:?}"; shift 2 ;;
    --util) MEMORY_UTILIZATION="${2:?}"; shift 2 ;;
    --max-batched-tokens) MAX_BATCHED_TOKENS="${2:?}"; shift 2 ;;
    --ready-timeout) READY_TIMEOUT="${2:?}"; shift 2 ;;
    --spec-config) SPEC_CONFIG="${2:?}"; shift 2 ;;
    --python) PYTHON_BIN="${2:?}"; shift 2 ;;
    --runtime-env) RUNTIME_ENV="${2:?}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$IPS" ]] || { echo "Missing --ips; see --help" >&2; exit 2; }
IFS=',' read -r -a NODES <<< "${IPS// /}"
(( ${#NODES[@]} == 4 )) || { echo "--ips needs exactly four addresses, got ${#NODES[@]}" >&2; exit 2; }
for ip in "${NODES[@]}"; do
  [[ "$ip" =~ ^[0-9]{1,3}(\.[0-9]{1,3}){3}$ ]] || { echo "Not an IPv4 address: $ip" >&2; exit 2; }
  IFS='.' read -r -a octets <<< "$ip"
  for octet in "${octets[@]}"; do
    (( 10#$octet <= 255 )) || { echo "IPv4 octet out of range in $ip" >&2; exit 2; }
  done
done
if [[ "${NODES[0]}" == "${NODES[1]}" || "${NODES[0]}" == "${NODES[2]}" || "${NODES[0]}" == "${NODES[3]}" ||
      "${NODES[1]}" == "${NODES[2]}" || "${NODES[1]}" == "${NODES[3]}" || "${NODES[2]}" == "${NODES[3]}" ]]; then
  echo "--ips must list four distinct nodes" >&2
  exit 2
fi

# Resolve the MTP config: a bundled name, or a path to a JSON file.
case "$SPEC_CONFIG" in
  synthetic|standard)
    SPEC_CONFIG_FILE="$SCRIPT_DIR/container/speculative_config.$SPEC_CONFIG.json" ;;
  */*|*.json)
    SPEC_CONFIG_FILE="$SPEC_CONFIG" ;;
  *)
    echo "--spec-config must be 'synthetic', 'standard', or a path to a JSON file" >&2
    exit 2 ;;
esac
[[ -f "$SPEC_CONFIG_FILE" ]] || { echo "Spec config not found: $SPEC_CONFIG_FILE" >&2; exit 2; }
python3 - "$SPEC_CONFIG_FILE" <<'PYCHECK' || { echo "Invalid MTP config: $SPEC_CONFIG_FILE" >&2; exit 2; }
import json, sys
cfg = json.load(open(sys.argv[1]))
if cfg.get("method") != "mtp":
    raise SystemExit(f"expected method=mtp, got {cfg.get('method')!r}")
if "rejection_sample_method" not in cfg:
    raise SystemExit("missing rejection_sample_method")
if cfg["rejection_sample_method"] == "synthetic":
    if not any(k in cfg for k in ("synthetic_acceptance_rate",
                                  "synthetic_acceptance_length",
                                  "synthetic_acceptance_rates")):
        raise SystemExit("synthetic mode needs an acceptance rate/length")
PYCHECK
[[ -f "$SCRIPT_DIR/container/serve.sh" ]] || { echo "Missing container/serve.sh" >&2; exit 2; }
[[ -f "$SCRIPT_DIR/container/cluster.env.example" ]] || { echo "Missing container/cluster.env.example" >&2; exit 2; }

ssh_node() { local ip="$1"; shift; ssh -o BatchMode=yes -o StrictHostKeyChecking=no "${SSH_USER}@${ip}" "$@"; }
remote_sudo() { local ip="$1"; shift; ssh_node "$ip" "sudo -n $*"; }

say() { printf '==> %s\n' "$*"; }

MODE="container"; [[ -n "$IMAGE" ]] || MODE="host"
say "Plan"
printf '   mode       : %s\n' "$([[ "$MODE" == container ]] && echo "container ($IMAGE)" || echo "bare metal (vllm serve)")"
[[ "$MODE" == host ]] && printf '   python     : %s\n' "$PYTHON_BIN"
[[ "$MODE" == container ]] && printf '   containers : %s-node<0..3>\n' "$CONTAINER_PREFIX"
[[ "$MODE" == host ]] && printf '   pid files  : %s/launch/node<0..3>.pid\n' "$RUN_DIR"
case "$SPEC_CONFIG" in
  synthetic|standard) printf '   spec config: %s\n' "$SPEC_CONFIG" ;;
  *)                  printf '   spec config: %s (custom)\n' "$SPEC_CONFIG_FILE" ;;
esac
printf '   run dir    : %s\n' "$RUN_DIR"
printf '   iface      : %s\n' "$COMM_IFACE"
printf '   nodes      : %s\n' "${NODES[*]}"
[[ -n "$MEMORY_UTILIZATION" ]] && printf '   util       : %s\n' "$MEMORY_UTILIZATION"
[[ -n "$MAX_BATCHED_TOKENS" ]] && printf '   max tokens : %s\n' "$MAX_BATCHED_TOKENS"
if [[ "$DRY_RUN" == "1" ]]; then
  say "Dry run: nothing was started."
  exit 0
fi

# 1. Refuse to clobber a running service; the operator may still want its state.
for idx in 0 1 2 3; do
  ip="${NODES[$idx]}"; name="${CONTAINER_PREFIX}-node${idx}"
  if [[ "$MODE" == "container" ]]; then
    if remote_sudo "$ip" "docker container inspect $name" >/dev/null 2>&1; then
      echo "Container $name already exists on $ip; run stop-dp8tp4ep32.sh first" >&2
      exit 2
    fi
  else
    if ssh_node "$ip" "test -f ${RUN_DIR}/launch/node${idx}.pid && kill -0 \$(cat ${RUN_DIR}/launch/node${idx}.pid) 2>/dev/null"; then
      echo "A bare-metal service is already running on $ip (${RUN_DIR}/launch/node${idx}.pid)" >&2
      echo "Run stop-dp8tp4ep32.sh first." >&2
      exit 2
    fi
  fi
done

# 1b. The same tag may resolve to different content per node (hand-built
# derivatives). Print the resolved IDs and refuse a mixed cluster: silently
# running different code on different ranks is far worse than a failed start.
if [[ "$MODE" == "container" ]]; then
declare -a IMG_IDS=()
for idx in 0 1 2 3; do
  ip="${NODES[$idx]}"
  id="$(remote_sudo "$ip" "docker image inspect --format '{{.Id}}' $IMAGE" 2>/dev/null | tail -1)"
  [[ -n "$id" ]] || { echo "Image $IMAGE missing on ${ip}" >&2; exit 2; }
  IMG_IDS[$idx]="$id"
  printf '   node%d image: %s\n' "$idx" "${id:0:19}"
done
for idx in 1 2 3; do
  if [[ "${IMG_IDS[$idx]}" != "${IMG_IDS[0]}" ]]; then
    echo "Image digest differs across nodes for the same tag $IMAGE." >&2
    echo "  node0 ${IMG_IDS[0]}" >&2
    printf '  node%d %s\n' "$idx" "${IMG_IDS[$idx]}" >&2
    echo "Pin a single digest with --image repo@sha256:..., or rebuild the tag" >&2
    echo "per node from the same commit. Starting anyway would run different code" >&2
    echo "on different ranks." >&2
    exit 2
  fi
done
fi

# 2. Stage the launcher payload on every node.
tmp_env="$(mktemp)"
trap 'rm -f "$tmp_env"' EXIT
# Keys this launcher owns; the template must not also define them, otherwise
# sourcing order would decide the winner.
overrides="NODE_ID NODE_IPS MASTER_IP COMM_IFACE MEMORY_UTILIZATION MAX_BATCHED_TOKENS"
[[ -n "$MEMORY_UTILIZATION" ]] || MEMORY_UTILIZATION="$(sed -n 's/^MEMORY_UTILIZATION=//p' "$SCRIPT_DIR/container/cluster.env.example" | tail -1)"
[[ -n "$MAX_BATCHED_TOKENS" ]] || MAX_BATCHED_TOKENS="$(sed -n 's/^MAX_BATCHED_TOKENS=//p' "$SCRIPT_DIR/container/cluster.env.example" | tail -1)"
{
  echo "# Generated by start-dp8tp4ep32.sh -- do not edit."
  echo "NODE_IPS=(${NODES[*]})"
  echo 'MASTER_IP="${NODE_IPS[0]}"'
  echo "COMM_IFACE=$COMM_IFACE"
  echo "MEMORY_UTILIZATION=$MEMORY_UTILIZATION"
  echo "MAX_BATCHED_TOKENS=$MAX_BATCHED_TOKENS"
  grep -E '^[A-Z_]+=' "$SCRIPT_DIR/container/cluster.env.example" \
    | awk -v skip="$overrides" '
        BEGIN { n = split(skip, k, " "); for (i = 1; i <= n; i++) banned[k[i]] = 1 }
        { key = $0; sub(/=.*/, "", key); if (!(key in banned)) print }
      '
} > "$tmp_env"

for idx in 0 1 2 3; do
  ip="${NODES[$idx]}"
  say "Staging node${idx} (${ip})"
  ssh_node "$ip" "sudo -n mkdir -p ${RUN_DIR}/launch && sudo -n mkdir -p ${RUN_DIR}/profiler && sudo -n mkdir -p ${RUN_DIR}/torch-extensions"
  ssh_node "$ip" "cat > /tmp/.step5-spec.json" < "$SPEC_CONFIG_FILE"
  ssh_node "$ip" "cat > /tmp/.step5-serve.sh" < "$SCRIPT_DIR/container/serve.sh"
  ssh_node "$ip" "cat > /tmp/.step5-cluster.env" < "$tmp_env"
  ssh_node "$ip" "sudo -n install -m 0755 /tmp/.step5-serve.sh ${RUN_DIR}/launch/serve.sh && sudo -n install -m 0644 /tmp/.step5-cluster.env ${RUN_DIR}/launch/cluster.env && sudo -n install -m 0644 /tmp/.step5-spec.json ${RUN_DIR}/launch/speculative_config.json && rm -f /tmp/.step5-serve.sh /tmp/.step5-cluster.env /tmp/.step5-spec.json"
done

# 3. Launch. Node0 last: it owns the DP coordinator and the API server.
if [[ "$MODE" == "container" ]]; then
  for idx in 1 2 3 0; do
    ip="${NODES[$idx]}"; name="${CONTAINER_PREFIX}-node${idx}"
    say "Creating ${name} on ${ip}"
    remote_sudo "$ip" "docker run -d --name $name \
      --network host --privileged --ipc host --pid host \
      --shm-size 32g --ulimit memlock=-1:-1 --ulimit nofile=65536:65536 \
      --security-opt label=disable \
      -e NODE_ID=$idx -e COMM_IFACE=$COMM_IFACE \
      -e MODE=container -e LAUNCH_DIR=/launch -e MODEL_PATH=/model/target \
      -e PYTHON_BIN=/opt/upgrade/.venv/bin/python \
      -e PROFILER_DIR=/run-cluster/profiler \
      -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
      -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware:ro \
      -v /usr/local/dcmi:/usr/local/dcmi:ro \
      -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
      -v ${RUN_DIR}/launch:/launch:ro \
      -v ${RUN_DIR}/profiler:/run-cluster/profiler \
      -v ${RUN_DIR}/torch-extensions:/run-cluster/torch-extensions \
      --entrypoint /bin/bash \
      $IMAGE -c 'exec /launch/serve.sh'"
  done
else
  # Bare metal: `vllm serve` runs directly on the node, so no mounts and no
  # device passthrough are needed; the weights are read from MODEL_PATH_HOST.
  model_host="$(sed -n 's/^MODEL_PATH_HOST=//p' "$tmp_env" | tail -1)"
  [[ -n "$model_host" ]] || { echo "MODEL_PATH_HOST missing from cluster.env" >&2; exit 2; }
  for idx in 1 2 3 0; do
    ip="${NODES[$idx]}"
    say "Starting bare metal on node${idx} (${ip})"
    ssh_node "$ip" "sudo -n test -d '$model_host'" \
      || { echo "Weights not found on ${ip}: $model_host" >&2; exit 2; }
    # nohup + setsid keeps the service alive after the SSH session closes.
    ssh_node "$ip" "cd ${RUN_DIR}/launch && sudo -n env \
      NODE_ID=$idx COMM_IFACE=$COMM_IFACE \
      MODE=host LAUNCH_DIR=${RUN_DIR}/launch MODEL_PATH='$model_host' \
      PYTHON_BIN='$PYTHON_BIN' \
      RUNTIME_ENV='${RUNTIME_ENV:-\$(if [ -f /opt/upgrade/runtime-env.sh ]; then echo /opt/upgrade/runtime-env.sh; fi)}' \
      PROFILER_DIR=${RUN_DIR}/profiler \
      TORCH_EXTENSIONS_DIR=${RUN_DIR}/torch-extensions \
      setsid nohup bash ${RUN_DIR}/launch/serve.sh \
        > ${RUN_DIR}/launch/node${idx}.log 2>&1 & echo \$! > ${RUN_DIR}/launch/node${idx}.pid; sleep 2; cat ${RUN_DIR}/launch/node${idx}.pid"
  done
fi

# 4. Wait for the API server on node0.
api_port="$(grep -E '^API_PORT=' "$tmp_env" | tail -1 | cut -d= -f2)"
api_port="${api_port:-19082}"
api_ip="${NODES[0]}"
say "Waiting for http://${api_ip}:${api_port}/health (timeout ${READY_TIMEOUT}s)"
deadline=$(( $(date +%s) + READY_TIMEOUT ))
while (( $(date +%s) < deadline )); do
  code="$(curl -s -m 5 -o /dev/null -w '%{http_code}' "http://${api_ip}:${api_port}/health" || true)"
  if [[ "$code" == "200" ]]; then
    say "Service is ready: http://${api_ip}:${api_port}/v1"
    exit 0
  fi
  for idx in 0 1 2 3; do
    ip="${NODES[$idx]}"
    if [[ "$MODE" == "container" ]]; then
      alive="$(remote_sudo "$ip" "docker container inspect -f '{{.State.Running}}' ${CONTAINER_PREFIX}-node${idx}" 2>/dev/null | tail -1)"
      hint="ssh ${SSH_USER}@${ip} sudo docker logs ${CONTAINER_PREFIX}-node${idx}"
    else
      alive="$(ssh_node "$ip" "test -f ${RUN_DIR}/launch/node${idx}.pid && kill -0 \$(cat ${RUN_DIR}/launch/node${idx}.pid) 2>/dev/null && echo true" || true)"
      hint="ssh ${SSH_USER}@${ip} tail -50 ${RUN_DIR}/launch/node${idx}.log"
    fi
    if [[ "$alive" != "true" ]]; then
      echo "node${idx} (${ip}) exited before becoming ready; see:" >&2
      echo "  ${hint}" >&2
      exit 1
    fi
  done
  sleep 15
done

echo "Timed out after ${READY_TIMEOUT}s waiting for /health." >&2
echo "Check logs: ssh ${SSH_USER}@${api_ip} sudo docker logs ${CONTAINER_PREFIX}-node0" >&2
exit 1
