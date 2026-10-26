#!/usr/bin/env bash
# 确保构建后端存在 base 镜像（按 digest 精确匹配）。
#   buildctl 后端：检查 containerd namespace；缺失时打印导入命令，
#                  传 --import-base 才真正执行导入（16GB，显式动作）。
#   docker  后端：检查本地 docker；缺失时提示 docker load。
# 本脚本不创建、不重配置任何 daemon。
set -euo pipefail
CI_LOG_TAG=base-image
# shellcheck source=lib.sh
. "$(dirname "$0")/lib.sh"

[ $# -ge 2 ] || die "usage: ensure-base-image.sh <buildctl|docker> <0|1 import-base>"
BACKEND="$1"
IMPORT_BASE="$2"    # 1 = 允许自动导入

# 支持独立运行：缺省时自行推导路径与配置
CI_PINS_DIR="${CI_PINS_DIR:-$(cd "$(dirname "$0")/../pins" && pwd)}"
[ -n "${CI_DELIVERY_ROOT:-}" ] || die "请通过 ci/build.sh 运行（或先导出 CI_DELIVERY_ROOT）"
CI_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CI_CI_DIR="${CI_CI_DIR:-$CI_DIR}"
[ -f "$CI_DIR/config/build.params" ] && . "$CI_DIR/config/build.params"
[ -f "$CI_DIR/config/delivery.env" ] && . "$CI_DIR/config/delivery.env"

BASE_REF="$(json_get "$CI_PINS_DIR/source-input.json" base.image)"
BASE_MANIFEST="$(json_get "$CI_PINS_DIR/source-input.json" base.manifest_digest)"
log "base image: $BASE_REF @ $BASE_MANIFEST"

case "$BACKEND" in
  buildctl)
    # ctr 解析：裸名走 PATH；绝对路径直接用（950DT 上 /opt/basemind 目录对普通用户
    # 不可 stat，command -v 会误报找不到，但 sudo -n 下可用）
    case "$CI_CTR_BIN" in
      /*) CTR_ABS="$CI_CTR_BIN" ;;
      *) CTR_ABS="$(command -v "$CI_CTR_BIN")" || die "找不到 ctr: $CI_CTR_BIN" ;;
    esac
    CTR_CMD=(${CI_CTR_SUDO:-} "$CTR_ABS")
    SUDO_FLAG=""
    [ -n "${CI_CTR_SUDO:-}" ] && SUDO_FLAG="--sudo"

    base_present() {
      # 按 digest 唯一匹配（REF 形式在不同 ctr 版本/导入方式下可能规范化，digest 不会）
      # 退出码：0=在位；1=不在；2=查询失败（ctr 本身报错）
      local rows stderr_file
      stderr_file="$(mktemp "${TMPDIR:-/tmp}/ctr-ls-stderr.XXXXXX")"
      set +e
      rows="$("${CTR_CMD[@]}" -n "$CI_CTR_NAMESPACE" images ls 2>"$stderr_file")"
      local rc=$?
      set -e
      if [ "$rc" -ne 0 ]; then
        warn "ctr images ls 失败（退出码 $rc）：$(head -3 "$stderr_file" 2>/dev/null)"
        rm -f "$stderr_file"
        return 2
      fi
      rm -f "$stderr_file"
      printf '%s\n' "$rows" | awk '{print $3}' | grep -qxF "$BASE_MANIFEST"
    }

    set +e
    base_present
    base_rc=$?
    set -e
    if [ "$base_rc" -eq 0 ]; then
      log "base image present in containerd namespace $CI_CTR_NAMESPACE"
      exit 0
    elif [ "$base_rc" -eq 2 ]; then
      die "无法查询 containerd namespace $CI_CTR_NAMESPACE（见上方 ctr 错误）"
    fi
    warn "containerd namespace $CI_CTR_NAMESPACE 中缺少 base 镜像"
    echo "  导入命令（会先校验 16GB OCI 的 sha256）：" >&2
    echo "    python3 '$CI_CI_DIR/scripts/import-oci.py' --delivery '$CI_DELIVERY_ROOT' \\" >&2
    echo "      --ctr '$CTR_ABS' --namespace '$CI_CTR_NAMESPACE' $SUDO_FLAG --apply" >&2
    if [ "$IMPORT_BASE" = "1" ]; then
      log "--import-base：开始导入（16GB，耗时较长）..."
      # shellcheck disable=SC2086
      python3 "$CI_CI_DIR/scripts/import-oci.py" --delivery "$CI_DELIVERY_ROOT" \
        --ctr "$CTR_ABS" --namespace "$CI_CTR_NAMESPACE" $SUDO_FLAG --apply
      # 导入后复查：import 退出 0 不等于 namespace 可见
      if base_present; then
        log "base image 导入完成并复查就位"
      else
        die "导入后仍未在 namespace $CI_CTR_NAMESPACE 见到 base digest"
      fi
    else
      die "base 镜像缺失；重试时加 --import-base，或先手动执行上面的导入命令"
    fi
    ;;
  docker)
    need_cmd "$CI_DOCKER_BIN"
    if "$CI_DOCKER_BIN" image inspect "$BASE_REF@$BASE_MANIFEST" >/dev/null 2>&1; then
      log "base image present in local docker"
      exit 0
    fi
    OCI_TARS=("$CI_IMAGES_DIR"/*.oci.tar)
    [ "${#OCI_TARS[@]}" -eq 1 ] || die "images/ 期望恰好 1 个 *.oci.tar，实际 ${#OCI_TARS[@]}"
    warn "本地 docker 缺少 base 镜像"
    echo "  导入命令（16GB，docker load 会同时装入父镜像与候选镜像）：" >&2
    echo "    $CI_DOCKER_BIN load -i '${OCI_TARS[0]}'" >&2
    if [ "$IMPORT_BASE" = "1" ]; then
      log "--import-base：docker load（16GB，耗时较长）..."
      "$CI_DOCKER_BIN" load -i "${OCI_TARS[0]}"
      log "docker load 完成"
    else
      die "base 镜像缺失；重试时加 --import-base，或先手动 docker load"
    fi
    ;;
  *)
    die "unknown backend: $BACKEND"
    ;;
esac
