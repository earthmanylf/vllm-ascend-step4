# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# This file is a part of the vllm-ascend project.
"""Ascend Step5 router helper.

The TailRing port removed the SWF scratch pool and undo journal, so the only
surviving helper is the FP32-accumulating router projection.
"""

from __future__ import annotations

import torch
from vllm.model_executor.layers.linear import ReplicatedLinear


class Step5FP32RouterLinear(ReplicatedLinear):
    """Step5 router projection with FP32 accumulation."""

    def forward(self, x: torch.Tensor):
        assert self.params_dtype == torch.float32
        return super().forward(x.to(torch.float32))


__all__ = [
    "Step5FP32RouterLinear",
]
