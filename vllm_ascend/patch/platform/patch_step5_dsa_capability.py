# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Patch target: vllm/models/step5/nvidia/model.py
# - `Step5Attention.__init__` gates the DSA backend on
#
#       capability = current_platform.get_device_capability()
#       if capability is None or not (
#           Step5DSAAttentionBackend.supports_compute_capability(capability)
#       ):
#           raise NotImplementedError(...)
#
#   Ascend NPU has no SM version: `AscendPlatform.get_device_capability`
#   returns None, so the `capability is None` short-circuit fires and model
#   construction dies with
#   `NotImplementedError: Step5 DSA requires SM90 or SM103; got None`
#   before any Ascend code runs.
#
#   Note the short-circuit: patching `supports_compute_capability` alone is
#   useless, because that function is never called when capability is None.
#
# Why a platform patch:
#   The gate lives in upstream vllm's step5 model code and runs during model
#   construction inside the worker process. The actual attention work is done
#   by `AscendStep5DSABackend` in vllm-ascend, so the CUDA compute-capability
#   precondition simply does not apply here.
#
# How:
#   `current_platform` is a module-level name binding in the step5 model
#   module (line 81). Rebinding only that module's name to a delegating proxy
#   keeps the change surgical -- the ~20 other vllm call sites of
#   `get_device_capability` (compilation fusion, distributed all-reduce,
#   other models) keep seeing the real Ascend platform and its None.
#   The proxy returns DeviceCapability(9, 0) only for
#   `get_device_capability`, which is exactly what the DSA guard needs;
#   every other attribute delegates untouched.
#
#   Line 1254's `current_platform.is_device_capability(103)` is gated on
#   `not self.use_dsa_backend`, so it is False either way and unaffected.
#
# Related PR (if no, explain why):
#   No upstream PR: upstream step5 DSA is CUDA-only. Ascend support is
#   provided by vllm-ascend's own backend.
#
# Future Plan:
#   Remove when upstream gates the DSA backend on a backend capability query
#   rather than a CUDA compute capability.

import importlib

from vllm.logger import logger

_MODULE = "vllm.models.step5.nvidia.model"
_SPARSE_MODULE = "vllm.models.step5.nvidia.sparse_attention"
_PROXY_ATTR = "_ascend_capability_proxy"


class _AscendPlatformProxy:
    """Delegating proxy that reports an SM90 compute capability.

    Only `get_device_capability` is overridden. Everything else -- including
    `device_name` ("npu") and `device_type` -- is forwarded to the real
    Ascend platform so unrelated behavior is unchanged.
    """

    def __init__(self, real):
        object.__setattr__(self, "_real", real)

    def get_device_capability(self, device_id: int = 0):
        from vllm.platforms.interface import DeviceCapability

        return DeviceCapability(9, 0)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_real"), name)

    def __setattr__(self, name, value):
        setattr(object.__getattribute__(self, "_real"), name, value)


def _patch_module(module_name: str) -> bool:
    """Rebind one step5 module's `current_platform` to the capability proxy."""
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        logger.info("%s not available; skipping DSA capability patch", module_name)
        return False

    if getattr(module, _PROXY_ATTR, None) is not None:
        return True

    real_platform = getattr(module, "current_platform", None)
    if real_platform is None:
        logger.info("%s has no current_platform binding; skipping", module_name)
        return False

    proxy = _AscendPlatformProxy(real_platform)
    module.current_platform = proxy
    setattr(module, _PROXY_ATTR, proxy)
    logger.info(
        "Patched %s.current_platform to report SM90 for the step5 DSA capability gate",
        module_name,
    )
    return True


def _apply_patch() -> None:
    # Two step5 modules carry the identical DSA capability guard, each with its
    # own module-level `current_platform` binding:
    #   model.py:1017           (Step5Attention.__init__)
    #   sparse_attention.py:292 (Step5DSAAttentionImpl.__init__)
    # Both must be patched or construction still dies on the second one.
    for module_name in (_MODULE, _SPARSE_MODULE):
        _patch_module(module_name)


_apply_patch()
