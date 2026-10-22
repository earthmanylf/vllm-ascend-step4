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
#   `current_platform.get_device_capability()` returning a CUDA compute
#   capability in (SM90, SM103). Ascend NPU has no SM version, so
#   `vllm_ascend.platform.get_device_capability` returns None and the model
#   raises `NotImplementedError: Step5 DSA requires SM90 or SM103; got None`
#   before any Ascend code runs.
#   The actual attention work is done by the Ascend backend
#   (`AscendStep5DSABackend` in vllm_ascend), so the capability gate is a
#   CUDA-specific precondition that does not apply here.
#
# Why a platform patch:
#   The gate lives in upstream vllm's step5 model code and runs during model
#   construction inside the worker process, long after CLI parsing. Patching
#   `Step5DSAAttentionBackend.supports_compute_capability` keeps the upstream
#   tree untouched while letting the DSA path proceed on Ascend.
#
# How:
#   Wrap `supports_compute_capability` so it also returns True when the active
#   platform is Ascend. Non-Ascend platforms keep the original SM90/SM103
#   check exactly.
#
# Related PR (if no, explain why):
#   No upstream PR: upstream step5 DSA is CUDA-only. Ascend support is
#   provided by vllm-ascend's own backend.
#
# Future Plan:
#   Remove when upstream gates the DSA backend on a backend capability query
#   rather than a CUDA compute capability.

from vllm.logger import logger


def _apply_patch() -> None:
    try:
        from vllm.models.step5.nvidia import sparse_attention as _sa
    except ImportError:
        logger.info("step5 sparse_attention not available; skipping DSA capability patch")
        return

    backend_cls = getattr(_sa, "Step5DSAAttentionBackend", None)
    if backend_cls is None:
        logger.info("Step5DSAAttentionBackend not found; skipping DSA capability patch")
        return

    if getattr(backend_cls, "_ascend_dsa_capability_patched", False):
        return

    original = backend_cls.supports_compute_capability.__func__

    def supports_compute_capability(cls, capability):
        try:
            from vllm.platforms import current_platform

            # AscendPlatform.device_name is the literal "npu" (see
            # vllm_ascend/platform.py); CUDA/ROCm report their own names.
            if current_platform.device_name == "npu":
                return True
        except Exception:  # noqa: BLE001 - never block model construction on a probe
            pass
        return original(cls, capability)

    backend_cls.supports_compute_capability = classmethod(supports_compute_capability)
    backend_cls._ascend_dsa_capability_patched = True
    logger.info("Patched Step5DSAAttentionBackend.supports_compute_capability for Ascend")


_apply_patch()
