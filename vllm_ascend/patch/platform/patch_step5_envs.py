"""Default the upstream Step5 fused QKV+gate GEMM on for Ascend.

Upstream gates the packed qkvg_proj projection (the head-wise attention
gate logits fused into the qkv GEMM, replacing a standalone g_proj GEMM
launched after attention) behind VLLM_STEP5_ENABLE_QKVG_PROJ, whose
vllm-side default is off. Ascend deployments want it on: export the
switch at patch time. Both the v1 engine core (core.py, load_general_plugins
before enable_envs_cache) and each multiproc worker (worker construction
before the executor's enable_envs_cache) apply these platform patches
before vllm envs are frozen, so the default lands in every process that
later reads it. An explicitly exported VLLM_STEP5_ENABLE_QKVG_PROJ always
wins.

Long term this default belongs upstream (flipping the vllm-side default
once the fused path is validated on Ascend), after which this patch can
be dropped.
"""

import os

os.environ.setdefault("VLLM_STEP5_ENABLE_QKVG_PROJ", "1")
