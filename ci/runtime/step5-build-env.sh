# Source only during image builds. Runtime uses step5-env.sh and real driver libs.
source /opt/step5/bin/step5-env.sh
TASK_STEP5_BUILD_HAL_STUB="$ASCEND_HOME_PATH/aarch64-linux/devlib/libascend_hal.so"
test -f "$TASK_STEP5_BUILD_HAL_STUB" || { echo "Missing CANN build HAL stub: $TASK_STEP5_BUILD_HAL_STUB" >&2; return 2; }
TASK_STEP5_BUILD_STUB_DIR=/tmp/step5-build-hal
mkdir -p "$TASK_STEP5_BUILD_STUB_DIR" /opt/step5/evidence
ln -sfn "$TASK_STEP5_BUILD_HAL_STUB" "$TASK_STEP5_BUILD_STUB_DIR/libascend_hal.so"
# Prepend only HAL: the entire devlib directory also contains incompatible stubs
# that shadow CANN's real liberror_manager and break torch_npu import.
export LD_LIBRARY_PATH="$TASK_STEP5_BUILD_STUB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TASK_STEP5_BUILD_HAL_STUB
python3 - <<'PY'
from pathlib import Path
import json,hashlib,os
p=Path(os.environ['TASK_STEP5_BUILD_HAL_STUB'])
Path('/opt/step5/evidence/build-hal-stub.json').write_text(json.dumps({'source':'Pinned official CANN9.2beta2 devlib; no host driver copied','path':str(p),'realpath':str(p.resolve()),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'scope':'Temporary build RUN LD_LIBRARY_PATH only; runtime environment excludes this stub'},indent=2)+'\n')
PY
