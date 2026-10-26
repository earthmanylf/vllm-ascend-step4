#!/bin/bash
source /opt/upgrade/cann92-env.sh
# Only libascend_hal link-time stub, not an entire toolkit devlib search path.
mkdir -p /tmp/step5-cann92-build-hal
ln -sf /usr/local/Ascend/cann-9.2.0/aarch64-linux/devlib/linux/aarch64/libascend_hal.so /tmp/step5-cann92-build-hal/libascend_hal.so
export LD_LIBRARY_PATH="/tmp/step5-cann92-build-hal:$LD_LIBRARY_PATH"
