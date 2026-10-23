#!/usr/bin/env bash
# Run on the Ascend950DT host. Never replace an existing container.
set -euo pipefail
CTR=${CTR:-/opt/basemind/containerd/bin/ctr}
NS=${NS:-step5-fp8-review-20261014}
CID=${CID:-step5-rebase-20261021}
IMAGE=${IMAGE:-docker.io/library/step5-fp8-a5:modelslim-a2ac-wheel-20261016}
WORK=${WORK:-/home/gpfs-ops/a5work}
SOURCE_ROOT=${SOURCE_ROOT:-/home/liuyi/step5-rebase}
WEIGHTS=${WEIGHTS:-/mnt/nvme0n1/chensiyu/Eco-Tech/Step-5-Preview-w8a8-mxfp8}
containers=$(sudo -n "$CTR" -n "$NS" containers list -q)
while IFS= read -r existing; do
    if [[ "$existing" == "$CID" ]]; then
        echo "Container $CID already exists; refusing to replace it. Use tasks exec to enter it." >&2
        exit 1
    fi
done <<< "$containers"
for path in "$WORK" "$SOURCE_ROOT" "$WEIGHTS" /usr/local/Ascend/driver; do
    [[ -d "$path" ]] || { echo "Missing host directory: $path" >&2; exit 1; }
done
[[ -x /usr/local/sbin/npu-smi ]] || { echo 'Missing /usr/local/sbin/npu-smi' >&2; exit 1; }
args=(run -d --privileged --net-host --cap-add CAP_IPC_LOCK)
for dev in /dev/davinci_manager /dev/hisi_hdc /dev/devmm_svm /dev/ummu/tid /dev/ubagg \
    /dev/ubmempfd /dev/unified_bus /dev/hisi_soc_cache_mgmt /dev/ubcore/ubcore; do
    [[ ! -e "$dev" ]] || args+=(--device "$dev")
done
for index in {0..7}; do
    [[ -e /dev/davinci$index ]] || { echo "Missing NPU device $index" >&2; exit 1; }
    args+=(--device "/dev/davinci$index")
done
for dev in /dev/uburma/*; do [[ ! -e "$dev" ]] || args+=(--device "$dev"); done
for lib in /usr/lib64/liburma.so.0.0.3 /usr/lib64/liburma_common.so.0.0.1 /usr/lib64/libummu.so.1.0.5; do
    [[ ! -e "$lib" ]] || args+=(--mount "type=bind,src=$(readlink -f "$lib"),dst=$lib,options=rbind:ro")
done
[[ ! -d /usr/lib64/urma ]] || args+=(--mount type=bind,src=/usr/lib64/urma,dst=/usr/lib64/urma,options=rbind:ro)
args+=(
    --mount type=bind,src=/usr/local/Ascend/driver,dst=/usr/local/Ascend/driver,options=rbind:ro
    --mount type=bind,src=/usr/local/sbin/npu-smi,dst=/usr/local/bin/npu-smi,options=rbind:ro
    --mount "type=bind,src=$WORK,dst=/work,options=rbind"
    --mount "type=bind,src=$SOURCE_ROOT,dst=$SOURCE_ROOT,options=rbind"
    --mount "type=bind,src=$WEIGHTS,dst=/models/step5-w8a8-mxfp8,options=rbind:ro"
    --mount type=tmpfs,dst=/dev/shm,options=size=16g
    --env ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    --env SOC_VERSION=ascend950dt_9582
    --env TORCH_DEVICE_BACKEND_AUTOLOAD=0
    "$IMAGE" "$CID" /bin/bash -c 'sleep infinity'
)
sudo -n "$CTR" -n "$NS" "${args[@]}"
printf 'Created %s. Install the rebase packages before serving; see README.md.\n' "$CID"
