#!/usr/bin/env bash
set -euo pipefail
ROOT="${EXPORT_ROOT:-/data/chensiyu/step5-tailring-final-export-20261023}"
IMAGE="${IMAGE_REF:-quay.io/ascend/vllm-ascend:v0.30.0rc1-a5-openeuler-step5-cann9.2.0-tailring-143f3649d8-final}"
mkdir -p "$ROOT"
[[ ! -e "$ROOT/image.tar.gz" && ! -e "$ROOT/image.tar.gz.part" ]] || { echo "Refusing to overwrite export" >&2; exit 2; }
docker image inspect "$IMAGE" > "$ROOT/export-image-inspect.json"
docker save "$IMAGE" | pigz -p 16 -1 > "$ROOT/image.tar.gz.part"
mv "$ROOT/image.tar.gz.part" "$ROOT/image.tar.gz"
sha256sum "$ROOT/image.tar.gz" > "$ROOT/image.sha256"
docker image inspect "$IMAGE" --format '{{.Id}}' > "$ROOT/image.id"
ls -lh "$ROOT/image.tar.gz"
