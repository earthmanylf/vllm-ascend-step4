#!/usr/bin/env bash
# Build an image from the code that is checked out right now.
#
#   ./build.sh                     # default: fused CANN-only A5 native build
#   ./build.sh --full              # native rebuild (needs aarch64 + CANN payload)
#   ./build.sh --image REF         # tag to produce
#   ./build.sh --check-only        # verify inputs, build nothing
#   ./build.sh --dry-run           # prepare the context, build nothing
#
# Two modes exist because they answer different questions:
#
#   python-overlay (explicit --python-overlay)
#     The base image already carries CANN 9.2.0, torch-npu, flash-attn and the
#     compiled vllm_ascend_C extension; this replaces the Python layer with the
#     current HEAD. Minutes, no network, no toolchain. Correct only while csrc/
#     is unchanged, which is verified against the base image's source commit
#     before anything is written -- otherwise it stops and points at --full.
#
#   --full
#     The native recipe: rebuilds the extension and custom ops from source. Needs
#     an aarch64 host, the CANN-only base, and LFS-hydrated source archives.
set -euo pipefail

CI_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
DOCKER="${CI_DOCKER_BIN:-docker}"
MODE=full
IMAGE=""
REGISTRY="${CI_IMAGE_REGISTRY:-UNIHUB}"
PUSH="${CI_PUSH:-0}"
RELEASE=""
REUSE_BASE_ASSETS="${REUSE_BASE_ASSETS:-0}"
WITH_UNION_OPS="${WITH_UNION_OPS:-0}"
VLLM_COMMIT="${VLLM_COMMIT:-}"
VLLM_TAG="${VLLM_TAG:-}"
GITLAB_ACCESS_TOKEN="${GITLAB_ACCESS_TOKEN:-${CI_GITLAB_TOKEN:-}}"
BASE_IMAGE="${CI_BASE_IMAGE:-}"
CHECK_ONLY=0
DRY_RUN=0
KEEP=0
SOURCE_REF="${CI_SOURCE_REF:-HEAD}"
EXTRA=()

usage() {
  cat <<'HELP'
Build an image from the code that is checked out right now.

  ./build.sh                  default: fused CANN-only A5 native build
  ./build.sh --full           same native CANN-only recipe
  ./build.sh --python-overlay explicit overlay for an already verified native image
  ./build.sh --check-only     verify the build can happen, build nothing
  ./build.sh --dry-run        prepare the context, build nothing

Modes:
  python-overlay (explicit --python-overlay)
    The base image already carries CANN 9.2.0, torch-npu, flash-attn and the
    compiled vllm_ascend_C extension, so only the Python layer is replaced with
    the current HEAD. Minutes, no network, no toolchain. Valid only while csrc/
    is unchanged; that is checked against the base image's source commit before
    anything is written, and the script stops pointing at --full if it differs.

  --full
    Native recipe: rebuilds the extension and custom ops from source. Needs an
    aarch64 host, the CANN-only base, and LFS-hydrated source archives.

Image naming: <registry>/stepcast:<release>-<vllm>-<ascend>-<date>-<time>
  e.g. hub.i.basemind.com/stepcast/stepcast:
       0.30.0rc2-b7042c5887ba-500440ee3b52-20261029-093000
  The release comes from the derived version, so the tag cannot drift from what
  was compiled.

Options:
  --image REF         Override the generated tag.
  --registry NAME     Registry shortcut (UNIHUB, SH-ALI, SH-VOLCANO-MLP,
                      SH-VOLCANO-SYS, BJ-BAIDU) or a full registry path.
                      Default UNIHUB -> hub.i.basemind.com/stepcast.
  --push              Push after building. Without it the image stays local.
  --release X.Y.Z     Release stamp for the tag (default: derived).
  --base-image REF    CANN-only base for native builds (default: pinned CANN ref);
                      verified native base when --python-overlay is explicit.
  --source-ref REF    git ref to package (default: HEAD).
  --reuse-base-assets Retired: a CANN-only base does not contain vLLM/ops assets.
  --vllm-commit SHA   Exact vllm commit to clone (default: the source pin).
  --with-union-ops    Native only: install union-mode custom ops from
                      ci/third-party/prebuilt/ (see ci/third-party/README.md).
  --max-jobs N        Native compile parallelism (default 16).
  --cargo-build-jobs N Cargo parallelism (default 16).
  --keep              Keep the generated context.
HELP
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --full) MODE=full; shift ;;
    --python-overlay) MODE=overlay; shift ;;
    --image) IMAGE="${2:?--image needs a value}"; shift 2 ;;
    --base-image) BASE_IMAGE="${2:?--base-image needs a value}"; shift 2 ;;
    --source-ref) SOURCE_REF="${2:?--source-ref needs a value}"; shift 2 ;;
    --check-only) CHECK_ONLY=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --keep) KEEP=1; shift ;;
    --registry) REGISTRY="${2:?--registry needs a value}"; shift 2 ;;
    --reuse-base-assets) echo "CANN-only recipe requires native source assets; --reuse-base-assets is no longer supported" >&2; exit 2 ;;
    --with-union-ops) WITH_UNION_OPS=1; shift ;;
    --vllm-commit) VLLM_COMMIT="${2:?}"; shift 2 ;;
    --vllm-tag) VLLM_TAG="${2:?}"; shift 2 ;;
    --push) PUSH=1; shift ;;
    --release) RELEASE="${2:?--release needs a value}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    --backend)
      # Accepted for compatibility with the previous entry point, but validated
      # here: only the docker backend exists, and the overlay path used to
      # swallow the flag entirely, so `--backend buildctl` silently built anyway.
      [[ "${2:-}" == docker ]] || {
        echo "--backend only supports 'docker', got '${2:-<none>}'" >&2; exit 2; }
      EXTRA+=("$1" "$2"); shift 2 ;;
    --fast|--source-rebase|--inputs-dir|--context-dir|--receipt|--max-jobs|--cargo-build-jobs)
      # Same compatibility, value-taking flags pass through to the native recipe.
      if [[ "$1" == --inputs-dir || "$1" == --context-dir || "$1" == --receipt || "$1" == --max-jobs || "$1" == --cargo-build-jobs ]]; then
        EXTRA+=("$1" "${2:?$1 needs a value}"); shift 2
      else
        EXTRA+=("$1"); shift
      fi ;;
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
  echo "==> Fused CANN-only A5 native build (local CANN base + source archives)"
  # The native recipe derives its own release tag from the context it prepares,
  # so an explicit --image is only needed to override it.
  NATIVE_ARGS=(--commit "$SOURCE_REF" --registry "$REGISTRY")
  ((CHECK_ONLY)) && NATIVE_ARGS+=(--check-only)
  ((DRY_RUN)) && NATIVE_ARGS+=(--dry-run)
  ((KEEP)) && NATIVE_ARGS+=(--keep)
  [[ -n "$IMAGE" ]] && NATIVE_ARGS+=(--image "$IMAGE")
  [[ -n "$BASE_IMAGE" ]] && NATIVE_ARGS+=(--base-image "$BASE_IMAGE")
  [[ -n "${VLLM_COMMIT:-}" ]] && NATIVE_ARGS+=(--vllm-commit "$VLLM_COMMIT")
  [[ -n "${VLLM_TAG:-}" ]] && NATIVE_ARGS+=(--vllm-tag "$VLLM_TAG")
  [[ -n "${GITLAB_ACCESS_TOKEN:-}" ]] && NATIVE_ARGS+=(--vllm-token "$GITLAB_ACCESS_TOKEN")
  [[ "${REUSE_BASE_ASSETS:-0}" == "1" ]] && NATIVE_ARGS+=(--reuse-base-assets)
  [[ "${WITH_UNION_OPS:-0}" == "1" ]] && NATIVE_ARGS+=(--with-union-ops)
  bash "$CI_DIR/scripts/build-official-v030-cann92.sh" \
    "${NATIVE_ARGS[@]}" "${EXTRA[@]+"${EXTRA[@]}"}"
  if ((PUSH)); then
    # build-official-v030-cann92.sh records the tag it produced in its receipt.
    RECEIPT="${RECEIPT_FILE:-$CI_DIR/ci-build-receipt-official-v030-cann92.json}"
    BUILT="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["image_ref"])' "$RECEIPT")"
    echo "==> pushing to $REGISTRY"
    "$DOCKER" push "$BUILT"
    echo "==> pushed $BUILT"
  fi
  exit
fi

[[ -n "$BASE_IMAGE" ]] || BASE_IMAGE="$(pick_base_image)"
echo "==> python-overlay"
echo "    base : $BASE_IMAGE"
echo "    image: ${IMAGE:-<derived after context>}"
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

# Release-style tag, as the released stepfun scheme does:
#   <registry>/stepcast:<release>-<vllm>-<ascend>-<date>-<time>
# The release comes from the derived version, so the tag cannot drift from what
# was actually compiled.
if [[ -z "$IMAGE" ]]; then
  VLLM_COMMIT="$("$DOCKER" image inspect "$BASE_IMAGE" --format \
    '{{index .Config.Labels "step5.vllm.commit"}}' 2>/dev/null || true)"
  [[ -n "$VLLM_COMMIT" && "$VLLM_COMMIT" != "<no value>" ]] || VLLM_COMMIT="unknown"
  NAME_ARGS=(--registry "$REGISTRY"
             --vllm-commit "$VLLM_COMMIT"
             --ascend-commit "$(git -C "$REPO" rev-parse "$SOURCE_REF")"
             --version-file "$CONTEXT/source-version.json")
  [[ -n "$RELEASE" ]] && NAME_ARGS+=(--release "$RELEASE")
  IMAGE="$(python3 "$CI_DIR/scripts/image_name.py" "${NAME_ARGS[@]}")"
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
if ((PUSH)); then
  echo "==> pushing to $REGISTRY"
  "$DOCKER" push "$IMAGE"
  echo "==> pushed $IMAGE"
fi
echo
"$DOCKER" image inspect "$IMAGE" --format \
  '{{index .Config.Labels "step5.vllm-ascend.version"}} @ {{index .Config.Labels "step5.vllm-ascend.commit"}}'
