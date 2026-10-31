#!/bin/bash
# Canonicalized from the CANN-only base by normalize-cann92.py.
source /usr/local/Ascend/cann-9.2.0/set_env.sh
ir=/usr/local/Ascend/cann-9.2.0/share/info/ascendnpu-ir/bin/set_env.sh
[[ ! -f "$ir" ]] || source "$ir"
atb=/usr/local/Ascend/nnal/atb/set_env.sh
[[ ! -f "$atb" ]] || source "$atb" --cxx_abi=1
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
export PATH="/opt/upgrade/.venv/bin:/opt/upgrade/cann-cli/bin:$PATH"
# The CANN base ships its own cann_ops_transformer under the SDK python path and
# set_env.sh puts that path ahead of site-packages, so it silently wins over the
# copy this recipe builds (which is the one carrying the Step5 ops such as
# indexer_region_score). Give the venv its normal precedence instead.
venv_site="/opt/upgrade/.venv/lib/python3.12/site-packages"
if [[ -d "$venv_site" ]]; then
    export PYTHONPATH="$venv_site${PYTHONPATH:+:$PYTHONPATH}"
fi
# Only real driver directories are eligible at runtime, not SDK HAL stubs.
for driver in /usr/local/Ascend/driver/lib64/common /usr/local/Ascend/driver/lib64/driver; do
    [[ ! -d "$driver" ]] || export LD_LIBRARY_PATH="$driver${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
done
vendor="$ASCEND_OPP_PATH/vendors/step5_ops_transformer"
if [[ -d "$vendor" ]]; then
    export ASCEND_CUSTOM_OPP_PATH="$vendor"
    export LD_LIBRARY_PATH="$vendor/op_api/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
# Union packages remain explicit opt-in; their presence must not replace GBSA.
if [[ "${VLLM_ASCEND_UNION_OPS:-0}" == "1" ]]; then
    for v in custom_nn_nn union_test_transformer; do
        d="$ASCEND_OPP_PATH/vendors/$v"
        if [[ -d "$d/op_api/lib" ]]; then
            export ASCEND_CUSTOM_OPP_PATH="$d${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
            export LD_LIBRARY_PATH="$d/op_api/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        fi
    done
fi
