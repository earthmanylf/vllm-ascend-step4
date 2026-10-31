#!/bin/bash
source /opt/upgrade/cann92-env.sh
# Locate the HAL link-time stub within this base's CANN layout. Never expose
# the full devlib directory to runtime; the temporary link is build-only.
hal=
for candidate in \
    "$ASCEND_HOME_PATH/aarch64-linux/devlib/linux/aarch64/libascend_hal.so" \
    "$ASCEND_HOME_PATH/aarch64-linux/devlib/libascend_hal.so"; do
    [[ ! -f "$candidate" ]] || { hal="$candidate"; break; }
done
[[ -n "$hal" ]] || { echo 'CANN developer base lacks the libascend_hal build stub' >&2; return 1; }
mkdir -p /tmp/step5-cann92-build-hal
ln -sf "$hal" /tmp/step5-cann92-build-hal/libascend_hal.so
export LD_LIBRARY_PATH="/tmp/step5-cann92-build-hal:$LD_LIBRARY_PATH"
