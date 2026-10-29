#!/usr/bin/env bash
# Build an image from the code that is checked out right now.
#
#   ./build.sh                     # default: python-overlay, fast, offline
#   ./build.sh --full              # native rebuild (needs aarch64 + CANN payload)
#   ./build.sh --image REF         # tag to produce
#   ./build.sh --check-only        # verify inputs, build nothing
#   ./build.sh --dry-run           # prepare the context, build nothing
#
# Two modes exist because they answer different questions:
#
#   python-overlay (default)
#     The base image already carries CANN 9.2.0, torch-npu, flash-attn and the
#     compiled vllm_ascend_C extension; this replaces the Python layer with the
#     current HEAD. Minutes, no network, no toolchain. Correct only while csrc/
#     is unchanged, which is verified against the base image's source commit
#     before anything is written -- otherwise it stops and points at --full.
#
#   --full
#     The native recipe: rebuilds the extension and custom ops from source. Needs
#     an aarch64 host, the 4.8 GB CANN payload and the LFS-hydrated archives.
set -euo pipefail

CI_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
DOCKER="${CI_DOCKER_BIN:-docker}"
MODE=overlay
IMAGE=""
BASE_IMAGE="${CI_BASE_IMAGE:-}"
CHECK_ONLY=0
DRY_RUN=0
KEEP=0
SOURCE_REF="${CI_SOURCE_REF:-HEAD}"
EXTRA=()

usage() {
  cat <<'HELP'
Build an image from the code that is checked out right now.

  ./build.sh                  default: python-overlay (fast, offline)
  ./build.sh --full           native rebuild (aarch64 + CANN payload)
  ./build.sh --check-only     verify the build can happen, build nothing
  ./build.sh --dry-run        prepare the context, build nothing

Modes:
  python-overlay (default)
    The base image already carries CANN 9.2.0, torch-npu, flash-attn and the
    compiled vllm_ascend_C extension, so only the Python layer is replaced with
    the current HEAD. Minutes, no network, no toolchain. Valid only while csrc/
    is unchanged; that is checked against the base image's source commit before
    anything is written, and the script stops pointing at --full if it differs.

  --full
    Native recipe: rebuilds the extension and custom ops from source. Needs an
    aarch64 host, the 4.8 GB CANN payload and the LFS-hydrated archives.

Options:
  --image REF         Tag to produce.
  --base-image REF    Base image for the overlay (default: newest local image
                      that carries a step5 source label).
  --source-ref REF    git ref to package (default: HEAD).
  --keep              Keep the generated context.
HELP
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --full) MODE=full; shift ;;
    --image) IMAGE="${2:?--image needs a value}"; shift 2 ;;
    --base-image) BASE_IMAGE="${2:?--base-image needs a value}"; shift 2 ;;
    --source-ref) SOURCE_REF="${2:?--source-ref needs a value}"; shift 2 ;;
    --check-only) CHECK_ONLY=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --keep) KEEP=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --fast) MODE=overlay; shift ;;
    --backend|--inputs-dir|--context-dir|--receipt|--commit)
      # Native-only flags: they mean nothing to the overlay build, so keep them
      # and let the --full branch decide, but refuse them in overlay mode below.
      EXTRA+=("$1" "${2:?$1 needs a value}"); shift 2 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

REPO="$(cd -- "$CI_DIR/.." && pwd -P)"

pick_base_image() {
  command -v "$DOCKER" >/dev/null 2>&1 || {
    echo "docker not found (set CI_DOCKER_BIN)" >&2; return 1; }
  local ref base_label mode_label
  local candidates=()
  while IFS= read -r ref; do
    base_label="$("$DOCKER" image inspect "$ref" --format \
      '{{index .Config.Labels "org.stepfun.step5.source_base"}}' 2>/dev/null || true)"
    [[ -n "$base_label" && "$base_label" != "<no value>" ]] || continue
    # Docker labels are inherited, so an overlay image still advertises the
    # native base commit it was layered on. Picking it would stack overlay on
    # overlay and accumulate one build's defects into the next, so only native
    # images qualify as a base.
    mode_label="$("$DOCKER" image inspect "$ref" --format \
      '{{index .Config.Labels "step5.build.mode"}}' 2>/dev/null || true)"
    [[ "$mode_label" == "python-overlay" ]] && continue
    candidates+=("$ref")
  done < <("$DOCKER" images --format '{{.Repository}}:{{.Tag}}' | grep -v '<none>' || true)
  ((${#candidates[@]})) || {
    echo "No local native image carries a step5 source label." >&2
    echo "Pass --base-image REF, or build one with --full." >&2
    return 1; }
  # Newest first: the most recently built native image is the one to extend.
  for ref in "${candidates[@]}"; do
    echo "$("$DOCKER" image inspect "$ref" --format '{{.Created}}') $ref"
  done | sort -r | head -1 | cut -d' ' -f2-
}

if [[ "$MODE" == full ]]; then
  echo "==> Native rebuild (aarch64 + CANN payload + LFS archives required)"
  exec bash "$CI_DIR/scripts/build-official-v030-cann92.sh" \
    --commit "$SOURCE_REF" ${IMAGE:+--image "$IMAGE"} "${EXTRA[@]+"${EXTRA[@]}"}"
fi

if ((${#EXTRA[@]})); then
  echo "Native-only options are not valid for the overlay build: ${EXTRA[*]}" >&2
  echo "Use --full if you need them." >&2
  exit 2
fi

[[ -n "$BASE_IMAGE" ]] || BASE_IMAGE="$(pick_base_image)"
[[ -n "$IMAGE" ]] || IMAGE="step5-ascend950:overlay-$(git -C "$REPO" rev-parse --short=10 "$SOURCE_REF")"
echo "==> python-overlay"
echo "    base : $BASE_IMAGE"
echo "    image: $IMAGE"
echo "    src  : $SOURCE_REF"

PREPARE=(python3 "$CI_DIR/scripts/prepare-python-overlay.py"
  --ci-dir "$CI_DIR" --base-image "$BASE_IMAGE"
  --source-ref "$SOURCE_REF" --docker "$DOCKER")
((CHECK_ONLY)) && PREPARE+=(--check-only)

if ((CHECK_ONLY)); then
  "${PREPARE[@]}"
  exit
fi

CONTEXT="$(mktemp -d "${TMPDIR:-/tmp}/step5-overlay.XXXXXX")/context"
cleanup() {
  if ((KEEP)); then echo "    context kept: $CONTEXT"; else rm -rf -- "$(dirname "$CONTEXT")"; fi
}
trap cleanup EXIT

PREPARE+=(--output "$CONTEXT")
"${PREPARE[@]}"

if ((DRY_RUN)); then
  echo "==> dry run: context prepared, nothing built"
  exit 0
fi

VERSION="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["version"])' "$CONTEXT/source-version.json")"
COMMIT="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["source_commit"])' "$CONTEXT/source-version.json")"
BASE_VERSION="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["base_version"])' "$CONTEXT/source-version.json")"

DOCKER_BUILDKIT=0 "$DOCKER" build --network host \
  --build-arg "BASE_IMAGE=$BASE_IMAGE" \
  --build-arg "VLLM_ASCEND_VERSION=$VERSION" \
  --build-arg "VLLM_ASCEND_COMMIT=$COMMIT" \
  --build-arg "VLLM_ASCEND_BASE_VERSION=$BASE_VERSION" \
  --build-arg "VLLM_ASCEND_BASE_IMAGE=$BASE_IMAGE" \
  -f "$CONTEXT/Dockerfile" -t "$IMAGE" "$CONTEXT"

echo
echo "==> built $IMAGE"
"$DOCKER" image inspect "$IMAGE" --format \
  '{{index .Config.Labels "step5.vllm-ascend.version"}} @ {{index .Config.Labels "step5.vllm-ascend.commit"}}'
