# SPDX-License-Identifier: Apache-2.0
"""Register PreFilledDecodeBenchConnector via KVConnectorFactory.

This patch injects the ``PreFilledDecodeBenchConnector`` into vLLM's
``KVConnectorFactory`` registry at platform-patch time (before the engine
starts), so no modification to the upstream vllm tree is needed.

The connector pre-fills the entire KV cache pool at worker init and claims
all tokens except the model's prefix-replay suffix on the scheduler side,
enabling pure-decode performance testing on Step4 (and other models with
``prefix_cache_recompute_tokens > 0``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1 import (
    KVConnectorBase_V1,
    KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.decode_bench_connector import (
    DecodeBenchConnector,
    DecodeBenchConnectorScheduler,
)
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)

_FLOAT8_DTYPES = {
    dtype
    for dtype_name in (
        "float8_e4m3fn",
        "float8_e5m2",
        "float8_e4m3fnuz",
        "float8_e5m2fnuz",
    )
    if (dtype := getattr(torch, dtype_name, None)) is not None
}


class PreFilledDecodeBenchConnector(DecodeBenchConnector):
    """KV Connector that pre-fills the entire KV cache pool at init.

    Worker: pre-fills entire KV pool at ``register_kv_caches``; ``start_fill_kv``
    is a no-op.
    Scheduler: claims every skippable token as matched, leaving only the model's
    prefix-replay suffix (``prefix_cache_recompute_tokens``) for a single fast
    local prefill chunk.
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(vllm_config, role, kv_cache_config)

        if role == KVConnectorRole.WORKER:
            self.connector_worker = _PreFilledWorker(vllm_config)
        elif role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = _ReplayAwareScheduler(
                vllm_config, kv_cache_config
            )

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        assert self.connector_worker is not None
        self.connector_worker.register_kv_caches(kv_caches)


class _ReplayAwareScheduler(DecodeBenchConnectorScheduler):
    """Scheduler that claims all tokens except the prefix-replay suffix.

    A model-side replay contract (Step4 DSA sets
    ``prefix_cache_recompute_tokens = block_size``) makes the scheduler decline
    any connector hit longer than
    ``num_tokens - prefix_cache_recompute_tokens`` (ext_tokens clamped to 0).
    Claim up to that ceiling and leave the replay suffix for local compute, so
    each request does one small prefill chunk and then decodes.
    """

    def __init__(
        self, vllm_config: "VllmConfig", kv_cache_config: "KVCacheConfig"
    ):
        super().__init__(vllm_config)
        self.prefix_cache_recompute_tokens = max(
            (
                group.kv_cache_spec.prefix_cache_recompute_tokens
                for group in kv_cache_config.kv_cache_groups
            ),
            default=0,
        )

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int, bool]:
        req_id = request.request_id
        if req_id in self._filled_requests:
            return 0, False

        num_uncomputed = request.num_tokens - num_computed_tokens
        replay_suffix = max(1, self.prefix_cache_recompute_tokens)
        num_to_claim = num_uncomputed - replay_suffix
        if num_to_claim <= 0:
            return 0, False

        return num_to_claim, False


class _PreFilledWorker:
    """Worker that pre-fills the entire KV cache pool at registration time.

    After init, ``start_fill_kv`` is a no-op: blocks are already filled.
    """

    def __init__(self, vllm_config: "VllmConfig"):
        self.vllm_config = vllm_config
        self.block_size = vllm_config.cache_config.block_size

        kv_transfer_config = vllm_config.kv_transfer_config
        assert kv_transfer_config is not None
        self.fill_mean = kv_transfer_config.get_from_extra_config(
            "fill_mean", 0.015
        )
        self.fill_std = kv_transfer_config.get_from_extra_config(
            "fill_std", 0.0
        )

        self.kv_caches: dict[str, torch.Tensor] | None = None
        self._pool_filled = False

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        self.kv_caches = kv_caches
        if not self._pool_filled:
            self._fill_entire_pool()

    def _make_fill_values(self, shape, dtype, device):
        gen_dtype = (
            torch.float16 if dtype in _FLOAT8_DTYPES else dtype
        )
        if self.fill_std > 0:
            values = torch.normal(
                mean=self.fill_mean,
                std=self.fill_std,
                size=shape,
                dtype=gen_dtype,
                device=device,
            )
        else:
            values = torch.full(
                shape, self.fill_mean, dtype=gen_dtype, device=device
            )
        return values.to(dtype) if gen_dtype != dtype else values

    def _fill_entire_pool(self):
        assert self.kv_caches is not None
        logger.info(
            "PreFilledDecodeBenchConnector: Pre-filling entire KV cache pool "
            "(%d layers, mean=%.6f, std=%.6f)",
            len(self.kv_caches),
            self.fill_mean,
            self.fill_std,
        )
        for _layer_name, kv_cache in self.kv_caches.items():
            if isinstance(kv_cache, torch.Tensor):
                dtype = kv_cache.dtype
                if dtype in _FLOAT8_DTYPES:
                    fill = self._make_fill_values(
                        kv_cache.shape, dtype, kv_cache.device
                    )
                    kv_cache.view(torch.uint8).copy_(fill.view(torch.uint8))
                else:
                    if self.fill_std > 0:
                        kv_cache.normal_(
                            mean=self.fill_mean, std=self.fill_std
                        )
                    else:
                        kv_cache.fill_(self.fill_mean)
            elif isinstance(kv_cache, (list, tuple)):
                for t in kv_cache:
                    if isinstance(t, torch.Tensor):
                        if t.dtype in _FLOAT8_DTYPES:
                            fill = self._make_fill_values(
                                t.shape, t.dtype, t.device
                            )
                            t.view(torch.uint8).copy_(
                                fill.view(torch.uint8)
                            )
                        elif self.fill_std > 0:
                            t.normal_(
                                mean=self.fill_mean, std=self.fill_std
                            )
                        else:
                            t.fill_(self.fill_mean)
        self._pool_filled = True
        logger.info(
            "PreFilledDecodeBenchConnector: KV cache pool pre-fill complete."
        )

    def start_fill_kv(self, metadata):
        logger.debug(
            "PreFilledDecodeBenchConnector: start_fill_kv called "
            "(no-op, pool pre-filled)"
        )


_CONNECTOR_NAME = "PreFilledDecodeBenchConnector"
_MODULE_PATH = "vllm_ascend.patch.platform.patch_decode_bench_connector"
_CLASS_NAME = "PreFilledDecodeBenchConnector"

if _CONNECTOR_NAME not in KVConnectorFactory._registry:
    KVConnectorFactory.register_connector(
        _CONNECTOR_NAME, _MODULE_PATH, _CLASS_NAME
    )
    logger.info(
        "Registered %s via vllm-ascend platform patch", _CONNECTOR_NAME
    )
