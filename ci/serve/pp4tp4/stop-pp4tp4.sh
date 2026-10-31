#!/usr/bin/env bash
# Stop the two-node Step5 PP4/TP4 service started by start-pp4tp4.sh.
#
#   ./stop-pp4tp4.sh --ips 10.0.0.1,10.0.0.2
#
# Add --remove to delete the containers afterwards (logs are lost then).
set -euo pipefail

IPS=""; CONTAINER_PREFIX="${CONTAINER_PREFIX:-step5-pp4tp4}"
IMAGE="${IMAGE:-}"; RUN_DIR="${RUN_DIR:-/data/step5-pp4tp4}"
REMOVE="${REMOVE:-0}"; SSH_USER="${SSH_USER:-root}"
DRY_RUN="${DRY_RUN:-0}"; STOP_TIMEOUT="${STOP_TIMEOUT:-60}"

usage() {
  cat <<'USAGE'
Stop the Step5 PP4/TP4 service, in either start mode.

Required:
  --ips A,B       The same two node IPs, node0 first.

Optional:
  --image REF     Container mode: must match what was started.
                  Omit for bare metal (stops the pid recorded by the starter).
  --name PREFIX   Container name prefix (default: step5-pp4tp4).
  --run-dir PATH  Node-local working dir (default: /data/step5-pp4tp4).
  --remove        Delete the containers after stopping them.
  --timeout SEC   Docker stop grace period (default: 60).
  --dry-run       Print the plan without touching any node.
  -h, --help      This message.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ips) IPS="${2:?}"; shift 2 ;;
    --name) CONTAINER_PREFIX="${2:?}"; shift 2 ;;
    --image) IMAGE="${2:?}"; shift 2 ;;
    --run-dir) RUN_DIR="${2:?}"; shift 2 ;;
    --remove) REMOVE=1; shift ;;
    --timeout) STOP_TIMEOUT="${2:?}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$IPS" ]] || { echo "Missing --ips; see --help" >&2; exit 2; }
IFS=',' read -r -a NODES <<< "${IPS// /}"
(( ${#NODES[@]} == 2 )) || { echo "--ips needs exactly two addresses, got ${#NODES[@]}" >&2; exit 2; }

ssh_node() { local ip="$1"; shift; ssh -o BatchMode=yes -o StrictHostKeyChecking=no "${SSH_USER}@${ip}" "$@"; }
LOCAL_IPS=" $(hostname -I 2>/dev/null || true) "
is_local() { [[ "$LOCAL_IPS" == *" $1 "* ]]; }
node_sh() { local ip="$1"; shift; if is_local "$ip"; then bash -c "$*"; else ssh_node "$ip" "$*"; fi; }
remote_sudo() { local ip="$1"; shift; if is_local "$ip"; then sudo -n bash -c "$*"; else ssh_node "$ip" "sudo -n $*"; fi; }

MODE="container"; [[ -n "$IMAGE" ]] || MODE="host"
echo "==> Mode: $([[ "$MODE" == container ]] && echo "container ($IMAGE)" || echo "bare metal (pid files)")"

# Stop node0 first so the API server and PP coordinator go away before the
# stage that feeds it starts logging connection errors.
for idx in 0 1; do
  ip="${NODES[$idx]}"; name="${CONTAINER_PREFIX}-node${idx}"
  if [[ "$MODE" == "container" ]]; then
    if [[ "$DRY_RUN" == "1" ]]; then
      echo "would stop $name on $ip (remove=$REMOVE)"
      continue
    fi
    echo "==> Stopping ${name} on ${ip}"
    remote_sudo "$ip" "docker stop -t ${STOP_TIMEOUT} ${name} >/dev/null 2>&1 || true"
    if [[ "$REMOVE" == "1" ]]; then
      remote_sudo "$ip" "docker rm -f ${name} >/dev/null 2>&1 || true"
    fi
  else
    pidfile="${RUN_DIR}/launch/node${idx}.pid"
    if [[ "$DRY_RUN" == "1" ]]; then
      echo "would stop the process in $pidfile on $ip"
      continue
    fi
    echo "==> Stopping bare-metal service on ${ip} (${pidfile})"
    # Run the teardown from a file: nested quoting through ssh breaks easily,
    # and `pkill -f <path>` would match (and kill) the shell that runs it, so
    # the pattern below is written as a bracket expression that cannot match
    # its own command line.
    node_sh "$ip" "cat > /tmp/.step5-stop.sh" <<STOP
#!/usr/bin/env bash
set -u
pidfile='${pidfile}'
pattern='${RUN_DIR}/launch/ser[v]e.sh'
p="\$(cat \"\$pidfile\" 2>/dev/null || true)"
if [ -n "\$p" ] && kill -0 "\$p" 2>/dev/null; then
  kill -TERM "\$p" 2>/dev/null || true
fi
for _ in \$(seq 1 ${STOP_TIMEOUT}); do
  [ -z "\$p" ] && break
  kill -0 "\$p" 2>/dev/null || break
  sleep 1
done
if [ -n "\$p" ] && kill -0 "\$p" 2>/dev/null; then
  echo "  pid \$p ignored TERM; sending KILL"
  kill -KILL "\$p" 2>/dev/null || true
fi
# Sweep leftover workers from an interrupted start.
for q in \$(pgrep -f "\$pattern" 2>/dev/null || true); do
  [ "\$q" = "\$\$" ] && continue
  kill -KILL "\$q" 2>/dev/null || true
done
rm -f "\$pidfile"
echo "  stopped"
STOP
    remote_sudo "$ip" "bash /tmp/.step5-stop.sh; rm -f /tmp/.step5-stop.sh" || true
    rm -f /tmp/.step5-stop.sh 2>/dev/null || true
  fi
done
echo "Done."
