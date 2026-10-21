# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# This file is a part of the vllm-ascend project.
"""Ascend scratch/router helpers and the fixed SWF summary-cache contract.

The summary cache itself is imported from the fixed SWF implementation; only
its NPU dispatch and Ascend owning-KV layout are adapted in summary_cache.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from vllm.model_executor.layers.linear import ReplicatedLinear


@dataclass
class Step5DSAScratchWorkspace:
    """Scratch storage shared by all DSA attention layers in one model."""

    tensor_buffers_by_engine: dict[int, dict[str, torch.Tensor]] = field(default_factory=dict)
    order_tokens_by_engine: dict[int, torch.Tensor] = field(default_factory=dict)
    memory_profiled: bool = False
    allocations_locked: bool = False

    def __post_init__(self) -> None:
        for virtual_engine, buffers in self.tensor_buffers_by_engine.items():
            if virtual_engine in self.order_tokens_by_engine:
                continue
            device = next((buffer.device for buffer in buffers.values()), None)
            if device is not None:
                self.order_tokens_by_engine[int(virtual_engine)] = torch.zeros((1,), device=device, dtype=torch.int64)

    def get_order_token(
        self,
        virtual_engine: int,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        virtual_engine = int(virtual_engine)
        token = self.order_tokens_by_engine.get(virtual_engine)
        if token is None or token.device != device or token.dtype is not torch.int64 or tuple(token.shape) != (1,):
            if self.allocations_locked:
                raise RuntimeError(
                    "Step5 DSA ordering-token capacity was exceeded after bind: "
                    f"virtual_engine={virtual_engine}, device={device}."
                )
            token = torch.zeros((1,), device=device, dtype=torch.int64)
            self.order_tokens_by_engine[virtual_engine] = token
        return token

    @torch.inference_mode()
    def reset_runtime_state(self) -> int:
        num_buffers = 0
        for buffers in self.tensor_buffers_by_engine.values():
            for buffer in buffers.values():
                buffer.zero_()
                num_buffers += 1
        for token in self.order_tokens_by_engine.values():
            token.zero_()
        return num_buffers


class Step5FP32RouterLinear(ReplicatedLinear):
    """Step5 router projection with FP32 accumulation."""

    def forward(self, x: torch.Tensor):
        assert self.params_dtype == torch.float32
        return super().forward(x.to(torch.float32))


__all__ = [
    "Step5DSAScratchWorkspace",
    "Step5FP32RouterLinear",
]
