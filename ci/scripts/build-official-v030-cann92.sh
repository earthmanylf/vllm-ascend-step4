#!/usr/bin/env bash
# Native ARM64 full build. Context checks may run on the x86 jump host.
set -euo pipefail
CI_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
REPO="$(cd "$CI_DIR/.." && pwd -P)"
# shellcheck source=lib.sh
. "$CI_DIR/scripts/lib.sh"
INPUT_ROOT="${CI_UPGRADE_INPUTS_DIR:-$CI_DIR/cache/tailring-cann9.2.0-143f3649d8}"
CONTEXT="${CONTEXT_DIR:-}"
IMAGE="${IMAGE_REF:-step5-fp8-a5:ci-tailring-cann9.2.0-143f3649d8}"
DOCKER="${CI_DOCKER_BIN:-docker}"
BACKEND="${BACKEND:-docker}"
COMMIT="${COMMIT:-143f3649d8}"
DRY_RUN=0
CHECK_ONLY=0
KEEP="${KEEP:-0}"
RECEIPT="${RECEIPT_FILE:-$CI_DIR/ci-build-receipt-official-v030-cann92.json}"
usage() {
  cat <<'HELP'
Usage: ci/build.sh [--dry-run|--check-only] [--inputs-dir DIR] [--context-dir NEW_DIR]
                   [--image REF] [--commit 143f3649d8|HEAD] [--keep] [--receipt FILE]
                   [--backend docker] [--fast]

Default input cache: ci/cache/tailring-cann9.2.0-143f3649d8
Default: full official v0.30.0rc1 + formal CANN 9.2.0 recipe from DSH session b5f168ee.
--check-only checks hashes of present inputs and reports missing inputs without building.
--dry-run requires every input and creates a verified context, but does not build.
Actual builds require native aarch64 and the exact official rc1 base config locally.
--fast falls back to this full rebuild: old native reuse is unsafe for this upgrade.
ci/scripts/seed-shared-cache.py --source EXISTING_ARCHIVE_ROOT
ci/scripts/fetch-cann92-payload.py --image quay.io/ascend/cann:9.2.0-950-openeuler24.03-py3.12-devel
Shared cache archives use Git LFS; obtain published objects with git lfs pull.
ci/build.sh --legacy [old flags] opts into the historical candidate build.
HELP
}
need_value() { [[ $# -ge 2 && "$2" != --* ]] || die "$1 requires a value"; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --inputs-dir) need_value "$@"; INPUT_ROOT="$2"; shift 2 ;;
    --context-dir) need_value "$@"; CONTEXT="$2"; shift 2 ;;
    --image) need_value "$@"; IMAGE="$2"; shift 2 ;;
    --commit) need_value "$@"; COMMIT="$2"; shift 2 ;;
    --receipt) need_value "$@"; RECEIPT="$2"; shift 2 ;;
    --backend) need_value "$@"; BACKEND="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --check-only) CHECK_ONLY=1; shift ;;
    --keep) KEEP=1; shift ;;
    --fast) warn "CANN/base upgrade requires native rebuild; --fast uses the same full recipe"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "Unknown official-profile option: $1 (historical flags require --legacy)" ;;
  esac
done
[[ "$BACKEND" == docker ]] || die "Official session profile requires --backend docker"
need_cmd python3; need_cmd git
EXPECTED="$(git -C "$REPO" rev-parse '143f3649d8^{commit}')"
SELECTED="$(git -C "$REPO" rev-parse --verify "$COMMIT^{commit}")"
[[ "$SELECTED" == "$EXPECTED" ]] || die "Exact session profile pins 143f3649d8, not $SELECTED"
if [[ "$CHECK_ONLY" == 1 ]]; then
  python3 "$CI_DIR/scripts/prepare-official-v030-cann92.py" \
    --ci-dir "$CI_DIR" --inputs-dir "$INPUT_ROOT" --check-only
  exit 0
fi
if [[ "$DRY_RUN" == 0 ]]; then
  [[ "$(uname -m)" == aarch64 ]] || die "Native build requires aarch64; host is $(uname -m)"
fi
TEMP_PARENT=""
if [[ -z "$CONTEXT" ]]; then
  TEMP_PARENT="$(mktemp -d "${TMPDIR:-/tmp}/step5-official-ci.XXXXXX")"
  CONTEXT="$TEMP_PARENT/context"
fi
cleanup() {
  local rc=$?
  if [[ -n "$TEMP_PARENT" ]]; then
    if [[ "$rc" == 0 && "$KEEP" == 0 && "$DRY_RUN" == 0 ]]; then
      rm -rf -- "${TEMP_PARENT:?}"
    else
      warn "Context retained: $CONTEXT"
    fi
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
python3 "$CI_DIR/scripts/prepare-official-v030-cann92.py" \
  --ci-dir "$CI_DIR" --inputs-dir "$INPUT_ROOT" --output "$CONTEXT"
CONTEXT="$(cd "$CONTEXT" && pwd -P)"
if [[ "$DRY_RUN" == 1 ]]; then
  log "Dry-run complete: $CONTEXT; no build/daemon/device operation"
  exit 0
fi
need_cmd "$DOCKER"
BASE="quay.io/ascend/vllm-ascend:v0.30.0rc1-a5-openeuler"
ACTUAL="$("$DOCKER" image inspect "$BASE" --format '{{.Id}}')"
LOCKED="$(json_get "$CI_DIR/pins/official-v030-cann92.json" base_config_digest)"
[[ "$ACTUAL" == "$LOCKED" ]] || die "Official rc1 base config mismatch: $ACTUAL != $LOCKED"
# Retain session network/Python/DNS behavior. No old native wheels or implicit base pull.
DOCKER_BUILDKIT=0 "$DOCKER" build --network host \
  --add-host pypi.org:151.101.192.223 \
  --add-host files.pythonhosted.org:151.101.192.223 \
  --add-host ascend-triton-open.obs.cn-north-4.myhuaweicloud.com:122.9.24.7 \
  --build-arg "BASE_IMAGE=$BASE" \
  -f "$CONTEXT/Dockerfile" -t "$IMAGE" "$CONTEXT" 2>&1 | tee "$CONTEXT/build.log"
"$DOCKER" image inspect "$IMAGE" > "$CONTEXT/built-image-inspect.json"
# Image-only probe: no host NPU, weights, source overlays, or existing services.
"$DOCKER" run --rm --network none --entrypoint /bin/bash "$IMAGE" -c \
  'source /opt/upgrade/cann92-env.sh &&
   test "$ASCEND_HOME_PATH" = /usr/local/Ascend/cann-9.2.0 &&
   test -f "$ASCEND_OPP_PATH/vendors/step5_ops_transformer/op_api/lib/libcust_opapi.so" &&
   python3 /opt/upgrade/probe_environment.py' > "$CONTEXT/runtime-identity.json"
python3 - "$CI_DIR" "$CONTEXT" "$IMAGE" "$RECEIPT" <<'PY'
import datetime
import json
import pathlib
import sys

ci, context = map(pathlib.Path, sys.argv[1:3])
image = json.loads((context / 'built-image-inspect.json').read_text())[0]
runtime = json.loads((context / 'runtime-identity.json').read_text())
assert runtime['machine'] == 'aarch64'
assert runtime['packages']['torch']['version'] == '2.10.0+cpu'
assert runtime['packages']['torch-npu']['version'] == '2.10.0.post4'
assert runtime['packages']['vllm']['version'] == '0.29.1rc1.dev2194+gca5e43ddcd.empty'
assert runtime['packages']['vllm-ascend']['version'] == '0.19.1rc2.dev2977+g143f3649d8'
assert runtime['packages']['triton-ascend']['version'] == '3.6.0+dev20261013182112'
assert image['Config']['Labels']['step5.cann.version'] == '9.2.0'
receipt = {
    'schema_version': 1,
    'built_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'image_ref': sys.argv[3], 'image_config_digest': image['Id'],
    'profile': json.loads((ci / 'pins/official-v030-cann92.json').read_text()),
    'context': json.loads((context / 'context-manifest.json').read_text()),
    'runtime': runtime,
    'matches_session_config': image['Id'] ==
        'sha256:0a16c96cc8be9bacdd5028d35bed3623297b5313870d87229984c08062033483',
    'device_validation': 'not run; separate 950DT numerical/inference acceptance required',
}
output = pathlib.Path(sys.argv[4]).expanduser().absolute()
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(receipt, indent=2) + '\n')
print('Build receipt:', output)
print('pip check diagnostic returncode:', runtime['pip_check']['returncode'])
PY
log "Build complete; device/inference validation has NOT been performed"
