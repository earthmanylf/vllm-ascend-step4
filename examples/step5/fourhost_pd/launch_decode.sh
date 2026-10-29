#!/usr/bin/env bash
set -eo pipefail
node=${1:?node rank}
case "$node" in 0) expected=gpu-a950-0026.qpct.sh.istep.fun;; 1) expected=gpu-a950-0027.qpct.sh.istep.fun;; *) exit 2;; esac
test "$(hostname)" = "$expected"
name=step5-dp4tp4-072-node$node
root=/home/liuyi/step5-pd-twohost-20261027
image=hub.i.basemind.com/stepcast/stepcast:0.30.0rc2-step5-20261016-vllm-e177c621f2-ascend-a2ac749ee3
test "$(npu-smi info | grep -c 'No running processes found in NPU')" -eq 8
if sudo -n docker ps -a --format '{{.Names}}' | grep -Fxq "$name"; then exit 2; fi
if ss -ltn | awk '{print $4}' | grep -Eq ':(19172|29672|13472)$'; then exit 3; fi
bash -n "$root/serve_dp4tp4_072.sh"
args=(sudo -n docker run -d --name "$name" --privileged --network host
 --shm-size 64g --ulimit memlock=-1:-1
 --mount type=bind,src=/usr/local/Ascend/driver,dst=/usr/local/Ascend/driver,readonly
 --mount type=bind,src=/data/models/Step-5-Preview-w8a8-mxfp8,dst=/work/model-target,readonly
 --mount "type=bind,src=$root,dst=/work/run")
for path in /usr/lib64/urma /usr/lib64/liburma.so.0.0.3 /usr/lib64/liburma_common.so.0.0.1 /usr/lib64/libummu.so.1.0.5; do
 test ! -e "$path" || args+=(--mount "type=bind,src=$(readlink -f "$path"),dst=$path,readonly")
done
"${args[@]}" --entrypoint /bin/bash "$image" -c \
 "exec bash /work/run/serve_dp4tp4_072.sh $node > /work/run/serve-dp4tp4-072-node$node.log 2>&1"
