#!/usr/bin/env bash
set -euo pipefail
CI_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
INPUTS="${CI_UPGRADE_INPUTS_DIR:-$CI_DIR/cache/tailring-cann9.2.0-143f3649d8}"
CONTEXT=""
IMAGE="quay.io/ascend/vllm-ascend:v0.30.0rc1-a5-openeuler-step5-cann9.2.0-tailring-143f3649d8"
RECEIPT=""
DRY_RUN=0
CHECK_ONLY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --inputs-dir|--context-dir|--image|--receipt)
      [[ $# -ge 2 && "$2" != --* ]] || { echo "$1 needs a value" >&2; exit 2; }
      case "$1" in
        --inputs-dir) INPUTS="$2" ;;
        --context-dir) CONTEXT="$2" ;;
        --image) IMAGE="$2" ;;
        --receipt) RECEIPT="$2" ;;
      esac
      shift 2 ;;
    --check-only) CHECK_ONLY=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --keep|--source-rebase|--fast) shift ;;
    --backend)
      [[ "${2:-}" == docker ]] || { echo "Only docker backend is supported" >&2; exit 2; }
      shift 2 ;;
    -h|--help)
      echo "TailRing source rebase on verified formal CANN 9.2.0 base"
      echo "ci/build.sh [--inputs-dir DIR] [--context-dir NEW_DIR] [--image REF] [--receipt FILE]"
      echo "[--check-only|--dry-run] [--keep] [--backend docker]"
      echo "Rebuilds vLLM-Ascend native code; retains same-CANN Flash/Step5 custom ops."
      exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
if [[ "$CHECK_ONLY" == 1 ]]; then
  python3 "$CI_DIR/scripts/prepare-tailring-source-rebase.py" --ci-dir "$CI_DIR" --inputs-dir "$INPUTS" --check-only
  exit 0
fi
[[ -n "$CONTEXT" ]] || { echo "An explicit new --context-dir is required" >&2; exit 2; }
python3 "$CI_DIR/scripts/prepare-tailring-source-rebase.py" --ci-dir "$CI_DIR" --inputs-dir "$INPUTS" --output "$CONTEXT"
[[ "$DRY_RUN" == 0 ]] || exit 0
[[ $(uname -m) == aarch64 ]] || { echo "Native aarch64 required" >&2; exit 2; }
BASE="quay.io/ascend/vllm-ascend:v0.30.0rc1-a5-openeuler-step5-cann9.2.0-317c7fb99e"
LOCKED="sha256:0a16c96cc8be9bacdd5028d35bed3623297b5313870d87229984c08062033483"
[[ $(docker image inspect "$BASE" --format '{{.Id}}') == "$LOCKED" ]] || { echo "Base config mismatch" >&2; exit 2; }
DOCKER_BUILDKIT=0 docker build --network host \
  --add-host pypi.org:151.101.192.223 --add-host files.pythonhosted.org:151.101.192.223 \
  --add-host mirrors.huaweicloud.com:120.46.2.67 \
  --build-arg "BASE_IMAGE=$BASE" -f "$CONTEXT/Dockerfile" -t "$IMAGE" "$CONTEXT" 2>&1 | tee "$CONTEXT/build.log"
docker image inspect "$IMAGE" > "$CONTEXT/built-image-inspect.json"
docker run --rm --network none --entrypoint /bin/bash "$IMAGE" -c \
  'source /opt/upgrade/cann92-env.sh; /opt/upgrade/.venv/bin/python /opt/upgrade/probe_environment.py' > "$CONTEXT/runtime-identity.json"
RECEIPT="${RECEIPT:-$CONTEXT/build-receipt.json}"
python3 "$CI_DIR/scripts/write-tailring-receipt.py" --ci-dir "$CI_DIR" --context "$CONTEXT" --image "$IMAGE" --output "$RECEIPT"
