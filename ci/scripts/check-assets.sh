#!/usr/bin/env bash
# 资产预检。
#   1) 仓库内 vendor 的输入（ci/inputs/）——强制校验（sha256 对照 pins）；
#   2) 交付目录资产——仅当 CI_DELIVERY_ROOT 存在时校验（没有交付目录访问权的人
#      跳过，大文件均可用 URL 重新获取，见 ci/inputs/REMOTE-SOURCES.md）。
# 用法：check-assets.sh [--deep]   --deep 额外对交付的 16GB OCI 做全量 sha256（约 3 分钟）
set -euo pipefail
shopt -s nullglob   # glob 不匹配时得到空数组，避免 ls 管道失败被 set -e 静默吞掉
CI_LOG_TAG=assets
# shellcheck source=lib.sh
. "$(dirname "$0")/lib.sh"

DEEP=0
[ "${1:-}" = "--deep" ] && DEEP=1

# 支持独立运行：缺省时自行推导路径与配置
CI_PINS_DIR="${CI_PINS_DIR:-$(cd "$(dirname "$0")/../pins" && pwd)}"
CI_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CI_INPUTS_DIR="$CI_DIR/inputs"
[ -n "${CI_DELIVERY_ROOT:-}" ] || CI_DELIVERY_ROOT="$(cd "$CI_DIR/../.." && pwd)"
[ -f "$CI_DIR/config/build.params" ] && . "$CI_DIR/config/build.params"
[ -f "$CI_DIR/config/delivery.env" ] && . "$CI_DIR/config/delivery.env"

PIN_JSON="$CI_PINS_DIR/source-input.json"
RECIPE_JSON="$CI_PINS_DIR/recipe-origin.json"
BUNDLE_SHA="$(json_get "$PIN_JSON" bundle_sha256)"
WHEEL_SHA="$(json_get "$RECIPE_JSON" wheel_sha256)"

# ---------- 1) 仓库内 vendor 输入（强制） ----------
log "vendored inputs: $CI_INPUTS_DIR"
check_file "$CI_INPUTS_DIR/source.bundle" "$BUNDLE_SHA" "inputs/source.bundle (Git 恢复 bundle)"

OPS_ART="$CI_INPUTS_DIR/cann-ops/ops-artifacts.json"
check_file "$OPS_ART" "" "inputs/cann-ops/ops-artifacts.json"
python3 - "$OPS_ART" "$CI_INPUTS_DIR/cann-ops" <<'PY'
import json, os, sys
art = json.load(open(sys.argv[1]))
base = sys.argv[2]
for row in art:
    p = os.path.join(base, row['file'])
    if not os.path.isfile(p):
        sys.exit('missing vendored ops artifact: ' + p)
    size = os.path.getsize(p)
    if size != row['bytes']:
        sys.exit('size mismatch: %s (%d != %d)' % (p, size, row['bytes']))
    print('[assets] asset ok: ' + p)
PY

for f in official-image-lock.json upgrade-platform.sh verify-platform-layers.py step5-env.sh \
         5f0fafc82f4b5f2deb0c1aeb4f3da674af66f014e8479f5f9961cdfc88516ae2.tar.gz \
         f7145efeeea1c8e065d9d8a298b3827ee6371c0059358170590154141f28cbe1.tar.gz; do
  check_file "$CI_INPUTS_DIR/cann-platform/$f" "" "inputs/cann-platform/$f"
done
check_file "$CI_INPUTS_DIR/cann-platform/adaptation/normalize-platform.py" "" "inputs/cann-platform/adaptation/normalize-platform.py"
for f in ascend-thirdparty-archive-lock.json ascend-thirdparty-files-manifest.json ops-thirdparty-archive-lock.json; do
  check_file "$CI_INPUTS_DIR/source-assets-locks/$f" "" "inputs/source-assets-locks/$f"
done
check_file "$CI_INPUTS_DIR/REMOTE-SOURCES.md" "" "inputs/REMOTE-SOURCES.md"
log "vendored inputs ok"

# ---------- 2) 交付目录资产（可选） ----------
if [ ! -d "$CI_DELIVERY_ROOT/build" ]; then
  warn "交付目录不可达（$CI_DELIVERY_ROOT），跳过交付侧校验；大文件按 ci/inputs/REMOTE-SOURCES.md 从 URL 获取"
  log "ALL ASSET CHECKS PASSED (vendored inputs only)"
  exit 0
fi
log "delivery root: $CI_DELIVERY_ROOT（存在，执行完整校验）"

# --- 构建输入（六件套除 source.tar 外） ---
check_file "$CI_BUILD_DIR/inputs/source.bundle" "$BUNDLE_SHA" "delivery source.bundle"
check_file "$CI_BUILD_DIR/Dockerfile" "" "delivery build/Dockerfile"
check_file "$CI_BUILD_DIR/recipe-origin.json" "" "delivery recipe-origin.json"
check_file "$CI_BUILD_DIR/source-lock.json" "" "delivery source-lock.json"

# --- wheelhouse（封存 wheel 与 ops 产物；输入 wheel 可 URL 获取，仅点数） ---
check_file "$CI_WHEELHOUSE_DIR/generated/vllm_ascend-0.30.0rc2.dev23+g7ae69ef36e5027a379ebbd1464503135a9a62034-cp312-cp312-linux_aarch64.whl" "$WHEEL_SHA" "封存 vllm_ascend wheel"
WHEEL_INPUTS=("$CI_WHEELHOUSE_DIR"/input/*.whl)
[ "${#WHEEL_INPUTS[@]}" -eq 8 ] || die "wheelhouse/input 期望 8 个 wheel，实际 ${#WHEEL_INPUTS[@]}"
log "asset ok: wheelhouse/input (${#WHEEL_INPUTS[@]} wheels)"

# --- CANN 离线资产 ---
PLATFORM_TARS=("$CI_CANN_PLATFORM_DIR"/payload/*.tar.gz)
[ "${#PLATFORM_TARS[@]}" -eq 3 ] || die "cann-platform/payload 期望 3 个 tar.gz，实际 ${#PLATFORM_TARS[@]}"
check_file "$CI_CANN_PLATFORM_DIR/payload/official-image-lock.json" "" "official-image-lock.json"
check_file "$CI_CANN_PLATFORM_DIR/adaptation/normalize-platform.py" "" "normalize-platform.py"

# --- 源码资产 ---
for d in ascend-thirdparty flash-attention-npu ops-thirdparty ops-transformer; do
  [ -d "$CI_SOURCE_ASSETS_DIR/$d" ] || die "missing source-assets/$d"
  log "asset ok: source-assets/$d"
done

# --- 镜像 OCI（期望字节数从 final-manifest.json 派生，不硬编码） ---
OCI_TARS=("$CI_IMAGES_DIR"/*.oci.tar)
[ "${#OCI_TARS[@]}" -eq 1 ] || die "images/ 期望恰好 1 个 *.oci.tar，实际 ${#OCI_TARS[@]}"
OCI_TAR="${OCI_TARS[0]}"
OCI_BYTES="$(stat -c %s "$OCI_TAR")"
EXPECTED_OCI_BYTES="$(python3 - "$CI_FINAL_MANIFEST" "$OCI_TAR" <<'PY'
import json, os, sys
manifest = json.load(open(sys.argv[1]))
rel = os.path.relpath(os.path.realpath(sys.argv[2]), os.path.dirname(os.path.realpath(sys.argv[1])))
for row in manifest['files']:
    if row['path'] == rel:
        print(row['bytes']); break
else:
    sys.exit('OCI not found in final-manifest.json: ' + rel)
PY
)"
[ "$OCI_BYTES" -eq "$EXPECTED_OCI_BYTES" ] || die "OCI 大小异常: $OCI_BYTES (期望 $EXPECTED_OCI_BYTES)"
log "asset ok: $OCI_TAR ($OCI_BYTES bytes)"
if [ "$DEEP" = "1" ]; then
  log "deep check: 计算 OCI sha256（顺序读 16GB，约 3 分钟）..."
  EXPECTED_OCI_SHA="$(python3 - "$CI_FINAL_MANIFEST" "$OCI_TAR" <<'PY'
import json, os, sys
manifest = json.load(open(sys.argv[1]))
rel = os.path.relpath(os.path.realpath(sys.argv[2]), os.path.dirname(os.path.realpath(sys.argv[1])))
for row in manifest['files']:
    if row['path'] == rel:
        print(row['sha256']); break
else:
    sys.exit('OCI not found in final-manifest.json: ' + rel)
PY
)"
  ACTUAL_OCI_SHA="$(sha256_file "$OCI_TAR")"
  [ "$ACTUAL_OCI_SHA" = "$EXPECTED_OCI_SHA" ] || die "OCI sha256 不匹配: $ACTUAL_OCI_SHA != $EXPECTED_OCI_SHA"
  log "asset ok: OCI sha256 == $ACTUAL_OCI_SHA"
else
  log "跳过 OCI 全量 sha256（--deep 可开启）"
fi

# --- 两个源码 checkout ---
PIN_COMMIT="${PIN_COMMIT:-$(json_get "$PIN_JSON" commit)}"
CI_REPO_ROOT="${CI_REPO_ROOT:-$(cd "$CI_DIR/.." && pwd)}"
VLLM_COMMIT="e177c621f2b8ae1be06190bc34eef84646e54f11"
[ -d "$CI_VLLM_DIR/.git" ] || die "missing vllm repo: $CI_VLLM_DIR"
git_has_commit "$CI_VLLM_DIR" "$VLLM_COMMIT" || die "vllm repo 缺少 commit $VLLM_COMMIT"
log "asset ok: vllm repo @ $VLLM_COMMIT"
git_has_commit "$CI_REPO_ROOT" "$PIN_COMMIT" || die "vllm-ascend repo 缺少封存 commit $PIN_COMMIT"
log "asset ok: vllm-ascend repo 包含封存 commit $PIN_COMMIT"

log "ALL ASSET CHECKS PASSED"
