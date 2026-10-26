# One environment for build, CPU capture, chroot numeric and eight-card runtime.
: "${CANN_SET_ENV_PATH:=/usr/local/Ascend/cann-9.2.0-beta.2/set_env.sh}"
[[ "$(readlink -f "$CANN_SET_ENV_PATH")" == /usr/local/Ascend/cann-9.2.0-beta.2/set_env.sh ]] || { echo 'Unexpected CANN environment target' >&2; return 2; }
# Remove only obsolete, versioned CANN entries inherited from the PR base.
# Generic Ascend aliases now resolve to the complete 9.2 tree; driver and
# non-CANN paths retain their original order.
for TASK_STEP5_PATH_VAR in PATH LD_LIBRARY_PATH PYTHONPATH CMAKE_PREFIX_PATH ASCEND_CUSTOM_OPP_PATH; do
    TASK_STEP5_FILTERED_PATH=
    IFS=: read -r -a TASK_STEP5_PATH_ITEMS <<< "${!TASK_STEP5_PATH_VAR-}"
    for TASK_STEP5_PATH_ITEM in "${TASK_STEP5_PATH_ITEMS[@]}"; do
        case "$TASK_STEP5_PATH_ITEM" in
            /usr/local/Ascend/cann-9.1.0|/usr/local/Ascend/cann-9.1.0/*|/usr/local/Ascend/ascend-toolkit/9.1.0|/usr/local/Ascend/ascend-toolkit/9.1.0/*) continue ;;
        esac
        [[ -n "$TASK_STEP5_PATH_ITEM" ]] || continue
        TASK_STEP5_FILTERED_PATH+="${TASK_STEP5_FILTERED_PATH:+:}$TASK_STEP5_PATH_ITEM"
    done
    export "$TASK_STEP5_PATH_VAR=$TASK_STEP5_FILTERED_PATH"
done
source "$CANN_SET_ENV_PATH" || return $?
[[ -f "$ASCEND_HOME_PATH/share/info/ascendnpu-ir/bin/set_env.sh" ]] || return 2
source "$ASCEND_HOME_PATH/share/info/ascendnpu-ir/bin/set_env.sh" || return $?
[[ -f /usr/local/Ascend/nnal/atb/set_env.sh ]] || return 2
source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=1 || return $?
export ATB_CXX_ABI=1
[[ "$ASCEND_HOME_PATH" == /usr/local/Ascend/cann-9.2.0-beta.2 && "$ASCEND_OPP_PATH" == "$ASCEND_HOME_PATH/opp" ]] || return 2
export TASK_STEP5_VENDOR="$ASCEND_OPP_PATH/vendors/step5_ops_transformer"
if [[ -d "$TASK_STEP5_VENDOR" ]]; then
    export ASCEND_CUSTOM_OPP_PATH="$TASK_STEP5_VENDOR${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
    export LD_LIBRARY_PATH="$TASK_STEP5_VENDOR/op_api/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
