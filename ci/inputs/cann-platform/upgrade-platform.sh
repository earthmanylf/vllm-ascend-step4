#!/usr/bin/env bash
# Executes inside the PR61 image only, with immutable official layers mounted RO.
set -eo pipefail
[[ "$(uname -m)" == aarch64 ]]
[[ -f /cann92/official-image-lock.json ]]
python3 /cann92/verify-platform-layers.py
mkdir -p /opt/step5-cann92-evidence
mkdir -p /opt/step5/bin
cp /cann92/official-image-lock.json /opt/step5-cann92-evidence/
cp /cann92/step5-env.sh /opt/step5/bin/step5-env.sh
# These are image filesystem paths; no host Ascend installation is mounted.
rm -rf /usr/local/Ascend /etc/Ascend
for digest in \
  9735e4842a98fd25d906d3bd349b3461fe5c3e631d485101861295ee325a98da \
  5f0fafc82f4b5f2deb0c1aeb4f3da674af66f014e8479f5f9961cdfc88516ae2; do
  tar --extract --gzip --file="/cann92/$digest.tar.gz" --directory=/ --no-same-owner
 done
# The official profile layer also contains OS setup; reproduce only the CANN
# environment via our shared helper. Keep PR61's original system/Python files.
[[ "$(readlink -f /usr/local/Ascend/cann)" == /usr/local/Ascend/cann-9.2.0-beta.2 ]]
[[ ! -e /usr/local/Ascend/cann-9.1.0 ]]
[[ -f /etc/Ascend/ascend_cann_install.info ]]
[[ -f /usr/local/Ascend/cann/share/info/ascendnpu-ir/bin/set_env.sh ]]
source /opt/step5/bin/step5-env.sh
[[ "$ASCEND_HOME_PATH" == /usr/local/Ascend/cann-9.2.0-beta.2 ]]
[[ "$ASCEND_OPP_PATH" == /usr/local/Ascend/cann-9.2.0-beta.2/opp ]]
python3 - <<'PY'
from pathlib import Path
import sys,json,os,importlib.metadata as m,hashlib
assert sys.version_info[:3]==(3,12,13),sys.version
versions={n:m.version(n) for n in ['torch','torch-npu','torchvision','torchaudio']}
assert versions=={'torch':'2.10.0+cpu','torch-npu':'2.10.0.post4','torchvision':'0.25.0+cpu','torchaudio':'2.10.0+cpu'},versions
record={'python':sys.version,'executable':sys.executable,'versions':versions,'environment':{k:v for k,v in os.environ.items() if k in ['ASCEND_HOME_PATH','ASCEND_OPP_PATH','ASCEND_TOOLKIT_HOME','ATB_HOME_PATH','ASCEND_HOME','PATH','LD_LIBRARY_PATH','PYTHONPATH']},'install_info':Path('/etc/Ascend/ascend_cann_install.info').read_text()}
Path('/opt/step5-cann92-evidence/platform-probe.json').write_text(json.dumps(record,indent=2)+'\n')
print('CANN92_PLATFORM_FILES_INSTALLED',json.dumps(record),flush=True)
PY
