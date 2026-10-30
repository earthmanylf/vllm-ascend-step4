#!/bin/bash
# Explicit official CANN9.2 release selection; never use toolkit devlib at runtime.
source /usr/local/Ascend/cann-9.2.0/set_env.sh
source /usr/local/Ascend/cann-9.2.0/share/info/ascendnpu-ir/bin/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=1
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.2.0
export ASCEND_TOOLKIT_HOME=$ASCEND_HOME_PATH ASCEND_AICPU_PATH=$ASCEND_HOME_PATH
export ASCEND_OPP_PATH=$ASCEND_HOME_PATH/opp
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
for key in PATH LD_LIBRARY_PATH PYTHONPATH CMAKE_PREFIX_PATH ASCEND_CUSTOM_OPP_PATH; do
    clean=
    IFS=: read -r -a items <<< "${!key-}"
    for item in "${items[@]}"; do
        case "$item" in
            *cann-9.1.0*|*cann-9.2.0-beta.2*|*/devlib|*/devlib/*|'') continue ;;
        esac
        clean+="${clean:+:}$item"
    done
    export "$key=$clean"
done
# CANN Python CLIs have a 3.12.15 shebang; use wrappers with retained official rc1 Python.
export PATH="/opt/upgrade/cann-cli/bin:$PATH"
vendor="$ASCEND_OPP_PATH/vendors/step5_ops_transformer"
if [[ -d "$vendor" ]]; then
    export ASCEND_CUSTOM_OPP_PATH="$vendor${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
    export LD_LIBRARY_PATH="$vendor/op_api/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

# Union mode installs two more vendor packages (union GBSA + UnionSparseMeta).
# They are opt-in because they do not merely add ops: all four aclnn entry
# points union GBSA exports -- aclnnGenericBlockSparseAttention and its
# metadata call, plus their GetWorkspaceSize -- also exist in
# step5_ops_transformer, and dlopen("libcust_opapi.so") resolves to the first
# match on this path. Exporting them by default therefore *replaces* the
# production GBSA kernel with the union fork, which measured 12-33% slower
# decode at 128K in the bs1..bs144 sweep. Set VLLM_ASCEND_UNION_OPS=1 to opt in;
# union GBSA must then stay in front of step5_ops, or it would never load.
if [[ "${VLLM_ASCEND_UNION_OPS:-0}" == "1" ]]; then
    for v in custom_nn_nn union_test_transformer; do
        d="$ASCEND_OPP_PATH/vendors/$v"
        if [[ -d "$d/op_api/lib" ]]; then
            export ASCEND_CUSTOM_OPP_PATH="$d${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
            export LD_LIBRARY_PATH="$d/op_api/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        fi
    done
fi
