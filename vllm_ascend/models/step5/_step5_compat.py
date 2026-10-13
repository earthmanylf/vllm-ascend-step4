# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# This file is a part of the vllm-ascend project.
"""Local re-implementations of Step5 helpers removed by the Inferact/swf
``step-5-preview-tracking`` fork from ``vllm.models.step5``.

The Ascend Step5 DSA / sparse-summary-cache code owns the usage of these
pure-Python (torch-only) helpers, so we vendor them here instead of importing
from the (now restructured) upstream fork. They are copied from the
stepfun/water18 vLLM fork (``vllm/models/step5/sparse_summary_cache.py``) where
vllm-ascend was previously validated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from vllm.config import ModelConfig
from vllm.model_executor.layers.linear import ReplicatedLinear


@dataclass
class Step5DSAScratchWorkspace:
    """Scratch storage shared by all DSA attention layers in one model."""

    tensor_buffers_by_engine: dict[int, dict[str, torch.Tensor]] = field(
        default_factory=dict
    )
    order_tokens_by_engine: dict[int, torch.Tensor] = field(default_factory=dict)
    memory_profiled: bool = False
    allocations_locked: bool = False

    def __post_init__(self) -> None:
        for virtual_engine, buffers in self.tensor_buffers_by_engine.items():
            if virtual_engine in self.order_tokens_by_engine:
                continue
            device = next((buffer.device for buffer in buffers.values()), None)
            if device is not None:
                self.order_tokens_by_engine[int(virtual_engine)] = torch.zeros(
                    (1,), device=device, dtype=torch.int64
                )

    def get_order_token(
        self,
        virtual_engine: int,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        virtual_engine = int(virtual_engine)
        token = self.order_tokens_by_engine.get(virtual_engine)
        if (
            token is None
            or token.device != device
            or token.dtype is not torch.int64
            or tuple(token.shape) != (1,)
        ):
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
        sentinel_minus_one = {
            "csa_prefill_scratch_region_ids",
            "csa_prefill_scratch_row_map",
        }
        num_buffers = 0
        for buffers in self.tensor_buffers_by_engine.values():
            for name, buffer in buffers.items():
                if name in sentinel_minus_one:
                    buffer.fill_(-1)
                else:
                    buffer.zero_()
                num_buffers += 1
        for token in self.order_tokens_by_engine.values():
            token.zero_()
        return num_buffers


class Step5SparseSummaryCacheConfig:
    num_pages: int
    page_size: int
    region_block_size: int
    num_kv_heads: int
    proxy_dim: int

    def __post_init__(self) -> None:
        for name in (
            "num_pages",
            "page_size",
            "region_block_size",
            "num_kv_heads",
            "proxy_dim",
        ):
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}.")

    @staticmethod
    def max_region_fragments_per_block(
        *,
        block_size: int,
        region_block_size: int,
    ) -> int:
        block_size = int(block_size)
        region_block_size = int(region_block_size)
        if block_size <= 0 or region_block_size <= 0:
            raise ValueError(
                "Step5 sparse requires positive block_size and "
                "region_block_size, "
                f"got block_size={block_size}, "
                f"region_block_size={region_block_size}."
            )
        common = math.gcd(block_size, region_block_size)
        return (
            block_size + region_block_size - common + region_block_size - 1
        ) // region_block_size

    @property
    def max_region_fragments_per_page(self) -> int:
        return self.max_region_fragments_per_block(
            block_size=self.page_size,
            region_block_size=self.region_block_size,
        )

    @property
    def summaries_per_page(self) -> int:
        return self.max_region_fragments_per_page

    @property
    def sum_shape(self) -> tuple[int, int, int, int]:
        return (
            self.num_pages,
            self.summaries_per_page,
            self.num_kv_heads,
            self.proxy_dim,
        )

    @property
    def count_shape(self) -> tuple[int, int, int]:
        return (self.num_pages, self.summaries_per_page, self.num_kv_heads)


def step5_sparse_summary_cache_bytes_per_block(
    *,
    block_size: int,
    region_block_size: int,
    proxy_dim: int,
    num_kv_heads: int = 1,
    sum_dtype: torch.dtype | None = None,
    count_dtype: torch.dtype | None = None,
) -> int:
    proxy_dim = int(proxy_dim)
    num_kv_heads = int(num_kv_heads)
    if proxy_dim <= 0 or num_kv_heads <= 0:
        raise ValueError(
            "Step5 sparse summary cache requires positive proxy_dim and "
            f"num_kv_heads, got proxy_dim={proxy_dim}, "
            f"num_kv_heads={num_kv_heads}."
        )
    fragments = Step5SparseSummaryCacheConfig.max_region_fragments_per_block(
        block_size=block_size, region_block_size=region_block_size
    )
    del sum_dtype, count_dtype
    return int(fragments * num_kv_heads * proxy_dim)


class Step5FP32RouterLinear(ReplicatedLinear):
    """Step5 router projection with FP32 accumulation."""

    def forward(self, x: torch.Tensor):
        assert self.params_dtype == torch.float32
        return super().forward(x.to(torch.float32))


def _require_resolved_valid_vocab_size(model_config: ModelConfig) -> int:
    if model_config.valid_vocab_size is None:
        raise ValueError(
            "Step5 requires valid_vocab_size to be resolved from the tokenizer "
            "before model construction. Call VllmConfig.resolve_valid_vocab_size() "
            "or pass --valid-vocab-size when tokenizer initialization is skipped."
        )
    return model_config.get_valid_vocab_size()



__all__ = [
    "Step5DSAScratchWorkspace",
    "Step5FP32RouterLinear",
    "_require_resolved_valid_vocab_size",
    "Step5SparseSummaryCacheConfig",
    "step5_sparse_summary_cache_bytes_per_block",
]
