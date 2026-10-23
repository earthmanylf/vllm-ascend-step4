#!/usr/bin/env bash
set -euo pipefail
node=${1:?node index 0..3}
[[ "$node" =~ ^[0-3]$ ]] || exit 2
ROOT="${SERVICE_ROOT:-/data/chensiyu/step5-tailring-service-20261023}"
IMAGE="${IMAGE_REF:-quay.io/ascend/vllm-ascend:v0.30.0rc1-a5-openeuler-step5-cann9.2.0-tailring-143f3649d8}"
NAME="${CONTAINER_NAME:-step5-tailring-dp8tp4ep32-node${node}-20261023}"
DRAFT_EAGER="${MTP_DRAFT_EAGER:-false}"
[[ -f "$ROOT/serve-tailring-dp8tp4.sh" ]] || { echo "Missing service script" >&2; exit 2; }
if docker container inspect "$NAME" >/dev/null 2>&1; then
  echo "Refusing to replace existing container: $NAME" >&2
  exit 2
fi
docker image inspect "$IMAGE" > "$ROOT/image-inspect-node${node}.json"
python3 - "$ROOT/image-inspect-node${node}.json" <<'PY'
import json
import sys
from pathlib import Path

j = json.loads(Path(sys.argv[1]).read_text())[0]
labels = j["Config"]["Labels"]
if j["Architecture"] != "arm64" or labels["step5.source.branch"] != "step5-preview-tailring-npu":
    raise ValueError("Wrong image architecture/source branch")
if labels["step5.mtp.fix"] != "fia-full-prebuilder-and-eager-real-prefix":
    raise ValueError("Image is missing MTP repairs")
PY
npu-smi info > "$ROOT/npu-before-node${node}.txt"
python3 - "$ROOT/npu-before-node${node}.txt" <<'PY'
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text()
if not all("No running processes found in NPU " + str(i) in text for i in range(8)):
    raise RuntimeError("NPU busy; stop the authorized old service first")
PY
[[ -z $(ss -H -ltn 'sport = :13345') ]] || { echo "DP RPC port busy" >&2; exit 2; }
if [[ "$node" == 0 ]]; then
  [[ -z $(ss -H -ltn 'sport = :19082') ]] || { echo "API port busy" >&2; exit 2; }
fi
docker run -d --name "$NAME" --network host --privileged --security-opt label=disable \
  --shm-size 32g --ulimit nofile=65536:65536 -e "MTP_DRAFT_EAGER=$DRAFT_EAGER" --entrypoint /bin/bash \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware:ro \
  -v /usr/local/dcmi:/usr/local/dcmi:ro -v /etc/hixlep:/etc/hixlep:ro \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
  -v /data/models/Step-5-Preview-w8a8-mxfp8:/model/target:ro -v "$ROOT:/run-cluster" \
  "$IMAGE" -c "exec bash /run-cluster/serve-tailring-dp8tp4.sh $node > /run-cluster/server-node${node}.log 2>&1"
