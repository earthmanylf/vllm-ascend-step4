#!/usr/bin/env bash
# 执行构建。参数：run-build.sh <backend> <context-dir> <image-ref> <commit> <tree> <metadata-file>
# buildctl：与原交付构建逐参数一致（dockerfile.v0 前端、target、platform、cache mount 由 Dockerfile 声明）。
# docker  ：buildx build，x86 主机经 QEMU 模拟 arm64，仅作应急后备，速度慢。
set -euo pipefail
CI_LOG_TAG=build
# shellcheck source=lib.sh
. "$(dirname "$0")/lib.sh"

[ $# -eq 6 ] || die "usage: run-build.sh <backend> <context-dir> <image-ref> <commit> <tree> <metadata-file>"
BACKEND="$1"; CONTEXT="$2"; IMAGE_REF="$3"; COMMIT="$4"; TREE="$5"; METADATA="$6"
# 支持独立运行：缺省时自行加载配置（build.sh 已 source 过，重复 source 幂等）
CI_DIR="$(cd "$(dirname "$0")/.." && pwd)"
[ -f "$CI_DIR/config/build.params" ] && . "$CI_DIR/config/build.params"
[ -f "$CI_DIR/config/delivery.env" ] && . "$CI_DIR/config/delivery.env"
DOCKERFILE_NAME="${CI_DOCKERFILE:-Dockerfile}"
[ -f "$CONTEXT/$DOCKERFILE_NAME" ] || die "context 缺少 $DOCKERFILE_NAME: $CONTEXT"
[ -f "$CONTEXT/source.tar" ] || die "context 缺少 source.tar: $CONTEXT"
rm -f "$METADATA"

case "$BACKEND" in
  buildctl)
    need_cmd "$CI_BUILDCTL_BIN"
    log "buildctl build: addr=$CI_BUILDKIT_ADDR image=$IMAGE_REF dockerfile=$DOCKERFILE_NAME"
    log "context=$CONTEXT commit=$COMMIT tree=$TREE"
    "$CI_BUILDCTL_BIN" --addr "$CI_BUILDKIT_ADDR" build \
      --frontend dockerfile.v0 \
      --local "context=$CONTEXT" --local "dockerfile=$CONTEXT" \
      --opt "filename=$DOCKERFILE_NAME" \
      --opt "target=$CI_TARGET" \
      --opt "platform=$CI_PLATFORM" \
      --opt image-resolve-mode=local \
      --opt "build-arg:FEATURE_COMMIT=$COMMIT" \
      --opt "build-arg:FEATURE_TREE=$TREE" \
      --output "type=image,name=$IMAGE_REF,push=false" \
      --metadata-file "$METADATA" --progress plain
    ;;
  docker)
    need_cmd "$CI_DOCKER_BIN"
    warn "docker 后端：x86 主机将通过 QEMU 模拟 $CI_PLATFORM（应急后备）"
    log "docker buildx build: image=$IMAGE_REF dockerfile=$DOCKERFILE_NAME"
    "$CI_DOCKER_BIN" buildx build \
      --platform "$CI_PLATFORM" \
      --build-arg "FEATURE_COMMIT=$COMMIT" \
      --build-arg "FEATURE_TREE=$TREE" \
      --target "$CI_TARGET" \
      -t "$IMAGE_REF" \
      -f "$CONTEXT/$DOCKERFILE_NAME" \
      --provenance=false --sbom=false \
      --progress plain \
      --load \
      "$CONTEXT"
    # buildx 不产生 buildctl 风格 metadata-file；用 image id 代替。
    "$CI_DOCKER_BIN" image inspect "$IMAGE_REF" --format '{{.Id}}' > "$METADATA"
    log "docker 后端完成，image id 写入 $METADATA"
    ;;
  *)
    die "unknown backend: $BACKEND"
    ;;
esac
