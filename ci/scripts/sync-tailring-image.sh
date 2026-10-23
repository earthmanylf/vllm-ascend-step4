#!/usr/bin/env bash
# The credential is provided only through the invoking process environment.
set -euo pipefail
HOST=${1:?destination host}
case "$HOST" in 177.42.26.1|177.42.27.1|177.42.28.1) ;; *) exit 2 ;; esac
case "$HOST" in
  177.42.26.1) PEER=172.27.9.106 ;;
  177.42.27.1) PEER=172.27.9.107 ;;
  177.42.28.1) PEER=172.27.9.108 ;;
esac
ROOT="${EXPORT_ROOT:-/data/chensiyu/step5-tailring-final-export-20261023}"
IMAGE="${IMAGE_REF:-quay.io/ascend/vllm-ascend:v0.30.0rc1-a5-openeuler-step5-cann9.2.0-tailring-143f3649d8-final}"
export SSH_ASKPASS=/data/chensiyu/step5-dp8tp4ep32-20261022/ssh-askpass.sh SSH_ASKPASS_REQUIRE=force DISPLAY=:0
: "${CLOUD_SSH_PASSWORD:?credential must be supplied in memory}"
OPTIONS=(-o NumberOfPasswordPrompts=1 -o ConnectTimeout=15 -o ServerAliveInterval=30 -o ServerAliveCountMax=6
  -o "HostKeyAlias=$HOST"
  -o UserKnownHostsFile=/data/chensiyu/step5-dp8tp4ep32-20261022/known-hosts
  -o StrictHostKeyChecking=yes -o UpdateHostKeys=no)
EXPECTED_SHA=$(cut -d' ' -f1 "$ROOT/image.sha256")
EXPECTED_ID=$(cat "$ROOT/image.id")
ssh "${OPTIONS[@]}" "root@$PEER" "mkdir -p '$ROOT'"
rsync -v --partial --append-verify -e "ssh ${OPTIONS[*]}" "$ROOT/image.tar.gz" "root@$PEER:$ROOT/"
ssh "${OPTIONS[@]}" "root@$PEER" "set -e;
  echo '$EXPECTED_SHA  $ROOT/image.tar.gz' | sha256sum -c -;
  docker load -i '$ROOT/image.tar.gz';
  test \"\$(docker image inspect '$IMAGE' --format '{{.Id}}')\" = '$EXPECTED_ID';
  printf '%s\n' '$EXPECTED_ID' > '$ROOT/verified-image-id.txt';
  echo IMAGE_SYNC_VERIFIED"
