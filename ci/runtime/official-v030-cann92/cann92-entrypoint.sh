#!/bin/bash
set -eo pipefail
source /opt/upgrade/cann92-env.sh
set -u
[[ -f "$ASCEND_OPP_PATH/vendors/step5_ops_transformer/op_api/lib/libcust_opapi.so" ]] || { echo 'Missing Step5 vendor' >&2; exit 2; }
exec "$@"
