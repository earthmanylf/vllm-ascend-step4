#!/usr/bin/env bash
# =============================================================================
# Step5 / Ascend 950DT 一键构建入口（ci/build.sh）
#
# 用法：
#   ./build.sh                      # 复现封存 r2 镜像（默认 pin commit）
#   ./build.sh --commit HEAD        # 构建当前 checkout HEAD
#   ./build.sh --commit <sha>       # 构建任意本仓库 commit
#   ./build.sh --dry-run            # 只做预检 + context 准备，不构建
#   ./build.sh --deep-check         # 资产校验含 16GB OCI 全量 sha256
#   ./build.sh --import-base        # base 镜像缺失时自动从 OCI 导入
#   ./build.sh --fast               # 快速路径：Python 级变更直接源码替换，跳过 native 重编译
#   ./build_fast.sh                 # 等价于 ./build.sh --fast
#   ./build.sh --help
#
# 流程：预检 -> 资产校验 -> 生成 context（仓库外）-> 确保 base 镜像 -> 构建 -> 校验+回执
# 纪律：不在交付目录内生成本地缓存；不创建/重配置任何 daemon；不触网。
# =============================================================================
set -euo pipefail

CI_CI_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CI_REPO_ROOT="$(cd "$CI_CI_DIR/.." && pwd)"
CI_PINS_DIR="$CI_CI_DIR/pins"
export CI_CI_DIR CI_REPO_ROOT CI_PINS_DIR

# 参数默认值（与环境变量共存：同名环境变量优先，再被 CLI 参数覆盖）
COMMIT="${COMMIT:-}"             # 空 = pins/source-input.json 的封存 commit
BASE="${BASE:-}"                 # commit 模式的 base commit（默认 pins 的 base_commit）
WHEEL_VERSION="${WHEEL_VERSION:-}"
BACKEND="${BACKEND:-}"           # 空 = build.params 默认
IMAGE_REF="${IMAGE_REF:-}"
DELIVERY_ROOT_OVERRIDE=""       # 仅来自 --delivery-root（严格）；CI_DELIVERY_ROOT env 宽容降级
CONTEXT_DIR="${CONTEXT_DIR:-}"   # 空 = 交付外临时目录（用完即清）
RECEIPT_FILE="${RECEIPT_FILE:-}" # 空 = ci/ci-build-receipt.json
KEEP="${KEEP:-0}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_ASSET_CHECK="${SKIP_ASSET_CHECK:-0}"
DEEP_CHECK="${DEEP_CHECK:-0}"
IMPORT_BASE="${IMPORT_BASE:-0}"
FAST="${FAST:-0}"               # 1 = 快速源码替换路径（Dockerfile.fast）

usage() { sed -n '2,18p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

# shellcheck source=scripts/lib.sh
. "$CI_CI_DIR/scripts/lib.sh"

# 取值参数统一校验：防止 "$2: unbound variable" 与 flag 被当值吃掉
need_value() {
  [ $# -ge 2 ] || die "参数 $1 需要一个值（见 --help）"
}

while [ $# -gt 0 ]; do
  case "$1" in
    --commit)         need_value "$@"; COMMIT="$2"; shift 2 ;;
    --base)           need_value "$@"; BASE="$2"; shift 2 ;;
    --wheel-version)  need_value "$@"; WHEEL_VERSION="$2"; shift 2 ;;
    --backend)        need_value "$@"; BACKEND="$2"; shift 2 ;;
    --image)          need_value "$@"; IMAGE_REF="$2"; shift 2 ;;
    --delivery-root)  need_value "$@"; DELIVERY_ROOT_OVERRIDE="$2"; shift 2 ;;
    --context-dir)    need_value "$@"; CONTEXT_DIR="$2"; shift 2 ;;
    --receipt)        need_value "$@"; RECEIPT_FILE="$2"; shift 2 ;;
    --keep)           KEEP=1; shift ;;
    --dry-run)        DRY_RUN=1; shift ;;
    --skip-asset-check) SKIP_ASSET_CHECK=1; shift ;;
    --deep-check)     DEEP_CHECK=1; shift ;;
    --import-base)    IMPORT_BASE=1; shift ;;
    --fast)           FAST=1; shift ;;
    -h|--help)        usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

# --delivery-root 覆盖必须先于 delivery.env：delivery.env 用 CI_DELIVERY_ROOT
# 派生全部资产路径，晚于它设置不会重算（M1）。
# --delivery-root（flag，严格：必须存在）与 CI_DELIVERY_ROOT（env，宽容：不存在则告警降级）
if [ -n "$DELIVERY_ROOT_OVERRIDE" ]; then
  [ -d "$DELIVERY_ROOT_OVERRIDE" ] || die "--delivery-root 目录不存在: $DELIVERY_ROOT_OVERRIDE"
  CI_DELIVERY_ROOT="$(cd "$DELIVERY_ROOT_OVERRIDE" && pwd)"
elif [ -n "${CI_DELIVERY_ROOT:-}" ]; then
  # env 形式：存在则规范化，不存在则保留原值（下方 DELIVERY_PRESENT 判 0 并告警）
  if [ -d "$CI_DELIVERY_ROOT" ]; then
    CI_DELIVERY_ROOT="$(cd "$CI_DELIVERY_ROOT" && pwd)"
  fi
else
  CI_DELIVERY_ROOT="$(cd "$CI_REPO_ROOT/.." && pwd)"
fi
export CI_DELIVERY_ROOT
if [ -d "$CI_DELIVERY_ROOT/build" ]; then
  DELIVERY_PRESENT=1
else
  DELIVERY_PRESENT=0
  warn "交付目录不可用（$CI_DELIVERY_ROOT）：仅使用仓库内 vendor 输入；大文件按 ci/inputs/REMOTE-SOURCES.md 从 URL 获取"
fi
export DELIVERY_PRESENT

set -a   # 后续配置对所有子脚本导出
# shellcheck source=config/delivery.env
. "$CI_CI_DIR/config/delivery.env"
# shellcheck source=config/build.params
. "$CI_CI_DIR/config/build.params"
set +a

BACKEND="${BACKEND:-$CI_BACKEND_DEFAULT}"
case "$BACKEND" in buildctl|docker) ;; *) die "未知 backend: $BACKEND（buildctl|docker）" ;; esac

PIN_COMMIT="$(json_get "$CI_PINS_DIR/source-input.json" commit)"
export PIN_COMMIT
EFFECTIVE_COMMIT="${COMMIT:-$PIN_COMMIT}"
# 统一解析为完整 40 位 sha：容器内 probe 断言 FEATURE_COMMIT == pin commit（完整 sha）
EFFECTIVE_COMMIT="$(git -C "$CI_REPO_ROOT" rev-parse --verify "$EFFECTIVE_COMMIT^{commit}" 2>/dev/null)" \
  || die "本仓库找不到 commit: $EFFECTIVE_COMMIT"

# 镜像 ref / 回执 / Dockerfile 的默认值（fast 模式用独立配方与 tag，需在 commit 解析后计算）
if [ "$FAST" = "1" ]; then
  CI_DOCKERFILE="Dockerfile.fast.legacy"
  DEFAULT_IMAGE_REF="step5-fp8-a5:fast-${EFFECTIVE_COMMIT:0:12}"
  DEFAULT_RECEIPT="$CI_CI_DIR/ci-build-receipt-fast.json"
else
  CI_DOCKERFILE="Dockerfile"
  DEFAULT_IMAGE_REF="$CI_IMAGE_REF_DEFAULT"
  DEFAULT_RECEIPT="$CI_CI_DIR/ci-build-receipt.json"
fi
export CI_DOCKERFILE
IMAGE_REF="${IMAGE_REF:-$DEFAULT_IMAGE_REF}"
# 回执默认写在 ci/ 内（被 ci/.gitignore 覆盖），避免弄脏仓库根目录或交付目录
RECEIPT_FILE="${RECEIPT_FILE:-$DEFAULT_RECEIPT}"

# context 目录：默认交付外的临时父目录（context 由 prepare-context 创建）；
# 显式 --context-dir 必须尚不存在
if [ -z "$CONTEXT_DIR" ]; then
  CONTEXT_PARENT="$(mktemp -d "${TMPDIR:-/tmp}/step5-ci-context.XXXXXX")"
  CONTEXT_DIR="$CONTEXT_PARENT/context"
  RESULT_DIR="$CONTEXT_PARENT"
  CONTEXT_IS_TEMP=1
else
  case "$CONTEXT_DIR" in /*) ;; *) CONTEXT_DIR="$PWD/$CONTEXT_DIR" ;; esac
  { [ -e "$CONTEXT_DIR" ] || [ -L "$CONTEXT_DIR" ]; } && die "context 目录已存在: $CONTEXT_DIR"
  RESULT_DIR="$(dirname "$CONTEXT_DIR")"
  [ -d "$RESULT_DIR" ] || die "context 父目录不存在: $RESULT_DIR"
  CONTEXT_IS_TEMP=0
fi

# 清理策略：成功且未要求保留才删临时 context；失败时保留现场并打印路径；
# 捕获 INT/TERM/HUP，保证中断也能清理或保留。
CONTEXT_KEEP_REASON=""
cleanup() {
  local rc=$?
  if [ "$rc" -ne 0 ]; then
    CONTEXT_KEEP_REASON="构建失败（退出码 $rc），保留现场便于排查"
  elif [ "$KEEP" = "1" ]; then
    CONTEXT_KEEP_REASON="--keep 指定保留"
  elif [ "$DRY_RUN" = "1" ]; then
    CONTEXT_KEEP_REASON="dry-run 保留（请手动清理 ${CONTEXT_PARENT:-$CONTEXT_DIR}）"
  fi
  if [ "$CONTEXT_IS_TEMP" = "1" ]; then
    if [ -n "$CONTEXT_KEEP_REASON" ]; then
      warn "$CONTEXT_KEEP_REASON: $CONTEXT_DIR"
    else
      rm -rf "$CONTEXT_PARENT"
      log "已清理临时 context"
    fi
  else
    log "保留 context: $CONTEXT_DIR"
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

PREPARE_RESULT="$RESULT_DIR/prepare-result.json"
METADATA_FILE="$CONTEXT_DIR/build-metadata.json"

log "============================================================="
log "Step5 / Ascend 950DT CI build"
log "  repo        : $CI_REPO_ROOT"
log "  delivery    : $CI_DELIVERY_ROOT"
log "  backend     : $BACKEND"
log "  commit      : $EFFECTIVE_COMMIT (pin: $PIN_COMMIT)"
log "  image       : $IMAGE_REF"
log "  context     : $CONTEXT_DIR"
log "============================================================="

# --- 1. 工具预检（只用 command -v，不做全盘扫描） ---
need_cmd git; need_cmd python3; need_cmd sha256sum
case "$BACKEND" in
  buildctl) need_cmd "$CI_BUILDCTL_BIN"; need_cmd "$CI_CTR_BIN" ;;
  docker)   need_cmd "$CI_DOCKER_BIN" ;;
esac
log "preflight ok (backend=$BACKEND)"

# --- 2. 资产校验 ---
if [ "$SKIP_ASSET_CHECK" = "1" ]; then
  warn "跳过资产校验（--skip-asset-check）"
else
  if [ "$DEEP_CHECK" = "1" ]; then
    CI_LOG_TAG=assets "$CI_CI_DIR/scripts/check-assets.sh" --deep
  else
    CI_LOG_TAG=assets "$CI_CI_DIR/scripts/check-assets.sh"
  fi
fi

# --- 3. 生成 context ---
PREPARE_ARGS=(--repo "$CI_REPO_ROOT" --ci-dir "$CI_CI_DIR"
              --commit "$EFFECTIVE_COMMIT" --output "$CONTEXT_DIR")
[ "$DELIVERY_PRESENT" = "1" ] && PREPARE_ARGS+=(--delivery "$CI_DELIVERY_ROOT")
[ -n "$BASE" ] && PREPARE_ARGS+=(--base "$BASE")
[ -n "$WHEEL_VERSION" ] && PREPARE_ARGS+=(--wheel-version "$WHEEL_VERSION")
CI_LOG_TAG=prepare python3 "$CI_CI_DIR/scripts/prepare-context.py" "${PREPARE_ARGS[@]}" | tee "$PREPARE_RESULT"
PIN_MATCH="$(python3 - "$PREPARE_RESULT" <<'PY'
import json, sys
try:
    print('true' if json.load(open(sys.argv[1]))['pin_match'] else 'false')
except Exception as exc:
    sys.exit('读取 prepare 结果失败: %s' % exc)
PY
)"
SOURCE_TREE="$(python3 - "$PREPARE_RESULT" <<'PY'
import json, sys
try:
    print(json.load(open(sys.argv[1]))['source_tree'])
except Exception as exc:
    sys.exit('读取 prepare 结果失败: %s' % exc)
PY
)"
if [ "$PIN_MATCH" = "true" ]; then
  log "context 与 recipe-origin.json 六件套 SHA 完全一致（封存复现）"
else
  warn "commit 模式：context 为新建 commit 生成，未做封存 pin 校验"
fi

# --- 3b. fast 门禁：仅允许 Python 级变更走源码替换路径 ---
if [ "$FAST" = "1" ]; then
  CI_LOG_TAG=fast-gate python3 - "$CONTEXT_DIR" <<'PY'
import json, sys
from pathlib import Path
pin = json.loads((Path(sys.argv[1]) / 'source-input.json').read_text())
changed = pin.get('changed_paths') or []
if not changed:
    sys.exit('FAST GATE REFUSED: source-input.json 无 changed_paths')
forbidden = ('csrc/', 'setup.py', 'pyproject.toml', 'CMakeLists.txt', '.gitmodules',
             'vllm_ascend/csrc/')
allowed = ('.py', '.json', '.yaml', '.yml', '.txt', '.cfg', '.toml', '.md')
bad = [p for p in changed if p.startswith(forbidden)]
bad += [p for p in changed if p.startswith('vllm_ascend/') and not p.endswith(allowed)]
if bad:
    print('FAST GATE REFUSED: 变更触及构建/二进制文件，源码替换不安全：', file=sys.stderr)
    for p in bad:
        print('  ' + p, file=sys.stderr)
    print('请使用完整构建：./build.sh（native 重编译）', file=sys.stderr)
    sys.exit(1)
print('fast gate ok: {} 个变更文件均为 Python 级'.format(len(changed)))
PY
  # fast 配方与替换脚本进入 context（6 个封存文件之外的新增件，不影响 pin 校验）
  cp "$CI_CI_DIR/docker/Dockerfile.fast.legacy" "$CONTEXT_DIR/Dockerfile.fast.legacy"
  cp "$CI_CI_DIR/scripts/fast_replace.py" "$CONTEXT_DIR/fast_replace.py"
  log "fast 模式：context 增加 Dockerfile.fast 与 fast_replace.py"
fi

if [ "$DRY_RUN" = "1" ]; then
  log "dry-run 完成，未执行构建"
  exit 0
fi

# --- 4. base 镜像 ---
CI_LOG_TAG=base-image "$CI_CI_DIR/scripts/ensure-base-image.sh" "$BACKEND" "$IMPORT_BASE"

# --- 5. 构建 ---
CI_LOG_TAG=build "$CI_CI_DIR/scripts/run-build.sh" "$BACKEND" "$CONTEXT_DIR" "$IMAGE_REF" \
  "$EFFECTIVE_COMMIT" "$SOURCE_TREE" "$METADATA_FILE"

# --- 6. 校验 + 回执 ---
CI_LOG_TAG=verify python3 "$CI_CI_DIR/scripts/verify-build.py" \
  --backend "$BACKEND" --metadata "$METADATA_FILE" --image "$IMAGE_REF" \
  --commit "$EFFECTIVE_COMMIT" --tree "$SOURCE_TREE" \
  --pin-match "$PIN_MATCH" \
  --context-files "$(python3 - "$PREPARE_RESULT" <<'PY'
import json, sys
try:
    print(json.dumps(json.load(open(sys.argv[1]))['context_files']))
except Exception as exc:
    sys.exit('读取 prepare 结果失败: %s' % exc)
PY
)" \
  --receipt "$RECEIPT_FILE"

log "============================================================="
log "构建完成：$IMAGE_REF"
log "回执：$RECEIPT_FILE"
log "注意：重新编译的 wheel/镜像不承诺逐 bit 相同，设备验收需另行执行。"
log "============================================================="
