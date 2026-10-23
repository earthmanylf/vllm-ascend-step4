#!/usr/bin/env bash
# CI 公共函数库。被 build.sh 与各步骤脚本 source。
# 约定：所有脚本 set -euo pipefail；日志统一前缀；错误带步骤号退出。

CI_LOG_TAG="${CI_LOG_TAG:-ci}"

log()  { printf "[%s] %s\n" "$CI_LOG_TAG" "$*"; }
warn() { printf "[%s] WARN: %s\n" "$CI_LOG_TAG" "$*" >&2; }
die()  { printf "[%s] ERROR: %s\n" "$CI_LOG_TAG" "$*" >&2; exit 1; }

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "required command not found on PATH: $1"
}

sha256_file() {
  local f="$1"
  [ -f "$f" ] || die "sha256_file: not a regular file: $f"
  local out
  out="$(sha256sum "$f")" || die "sha256sum failed: $f"
  printf '%s' "$out" | cut -d" " -f1
}

# 校验文件存在且（可选）sha256 匹配。用法：check_file <path> [expected_sha] [描述]
check_file() {
  local path="$1" expected="${2:-}" desc="${3:-$1}"
  [ -f "$path" ] || die "missing asset: $desc ($path)"
  if [ -n "$expected" ]; then
    local actual
    actual="$(sha256_file "$path")"
    [ "$actual" = "$expected" ] || die "sha256 mismatch for $desc ($path): expected $expected got $actual"
  fi
  log "asset ok: $desc"
}

# 从 JSON 文件读取字段（python3，无外部依赖；支持 a.b.c 点号嵌套路径）
json_get() {
  local file="$1" key="$2"
  [ -f "$file" ] || die "json_get: not a regular file: $file"
  python3 - "$file" "$key" <<'PY' || die "json_get 失败: $file 缺少字段 $key"
import json, sys
data = json.load(open(sys.argv[1]))
for part in sys.argv[2].split("."):
    data = data[part]
print(data)
PY
}

# git 对象存在性检查（不触网、不做全盘扫描）
git_has_commit() {
  local repo="$1" commit="$2"
  git -C "$repo" cat-file -e "$commit^{commit}" 2>/dev/null
}
