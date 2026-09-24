# SPDX-License-Identifier: Apache-2.0
"""Step5's page-aligned Mooncake V1 handoff contract."""

from typing import Any

import torch

STEP5_PD_VERSION = 1
STEP5_BUFFER_ALIGNMENT = 2 * 1024 * 1024


def allocate_step5_summary_cache(num_blocks: int, page_bytes: int, device: Any) -> torch.Tensor:
    num_bytes = num_blocks * page_bytes
    raw = torch.zeros(num_bytes + STEP5_BUFFER_ALIGNMENT - 1, dtype=torch.uint8, device=device)
    offset = (-raw.data_ptr()) % STEP5_BUFFER_ALIGNMENT
    return raw[offset : offset + num_bytes].view(num_blocks, page_bytes)


def is_step5_model(vllm_config: Any) -> bool:
    return vllm_config.model_config.hf_text_config.model_type in ("step5", "step5_vl", "step5_mtp")


def validate_step5_pd_config(vllm_config: Any) -> None:
    """Limit the initial protocol to matching TP and page ownership."""
    parallel = vllm_config.parallel_config
    if vllm_config.cache_config.block_size != 128:
        raise ValueError("Step5 Mooncake V1 requires block_size=128 on P and D.")
    if any(
        getattr(parallel, name) != 1
        for name in (
            "pipeline_parallel_size",
            "prefill_context_parallel_size",
            "decode_context_parallel_size",
        )
    ):
        raise ValueError("Step5 Mooncake V1 currently requires PP=PCP=DCP=1.")
    if vllm_config.speculative_config is not None:
        raise ValueError("Step5 Mooncake V1 does not yet support speculative decoding.")
    transfer = vllm_config.kv_transfer_config
    for side in ("prefill", "decode"):
        config = transfer.get_from_extra_config(side, {})
        if config.get("tp_size") != parallel.tensor_parallel_size or config.get("pp_size", 1) != 1:
            raise ValueError("Step5 Mooncake V1 requires matching P/D TP and PP=1 in kv_connector_extra_config.")


def step5_prefix_tokens(num_tokens: int, block_size: int) -> int:
    """Leave at least one page for D to rebuild its CSA active-slot state."""
    return max(0, (num_tokens - block_size) // block_size * block_size)


def prepare_step5_prefill(request: Any, block_size: int) -> None:
    """Cut P's prompt before prefix lookup, keeping SWA's tail at the cut."""
    params = request.kv_transfer_params
    if not params or not params.get("do_remote_decode") or "step5_pd_version" in params:
        return
    if getattr(request, "mm_features", None) or getattr(request, "resumable", False):
        raise ValueError("Step5 Mooncake V1 currently supports non-streaming text prompts only.")
    original_length = request.num_prompt_tokens
    prefix_tokens = step5_prefix_tokens(original_length, block_size)
    params.update(
        step5_pd_version=STEP5_PD_VERSION,
        step5_original_prompt_tokens=original_length,
        step5_prefix_tokens=prefix_tokens,
    )
    # An empty prompt is invalid. Short prompts run locally on D; P publishes
    # no cache blocks and therefore needs no delayed-free acknowledgement.
    if prefix_tokens == 0:
        return
    if request.prompt_token_ids is not None:
        del request.prompt_token_ids[prefix_tokens:]
    if request.prompt_embeds is not None:
        request.prompt_embeds = request.prompt_embeds[:prefix_tokens]
    if getattr(request, "prompt_is_token_ids", None) is not None:
        request.prompt_is_token_ids = request.prompt_is_token_ids[:prefix_tokens]
    del request._all_token_ids[prefix_tokens:]
    request.num_prompt_tokens = prefix_tokens
    request.max_tokens = 1


def step5_remote_prefix_tokens(request: Any, block_size: int) -> int:
    params = request.kv_transfer_params
    expected = step5_prefix_tokens(request.num_prompt_tokens, block_size)
    if (
        params.get("step5_pd_version") != STEP5_PD_VERSION
        or params.get("step5_original_prompt_tokens") != request.num_prompt_tokens
        or params.get("step5_prefix_tokens") != expected
        or params.get("remote_block_size") != block_size
    ):
        raise ValueError("Step5 P/D handoff contract mismatch; both peers must use the Step5 Mooncake V1 adapter.")
    return expected


def add_step5_summary_caches(
    kv_caches: dict[str, Any], forward_context: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """Append opaque summary pages to connector views, leaving model K/V intact."""
    transfer_caches = dict(kv_caches)
    summaries = {}
    for name, caches in kv_caches.items():
        owner = forward_context[name]
        sidecar = getattr(owner, "sparse_summary_cache", None)
        if sidecar is None:
            if getattr(owner, "kv_cache_requires_zeroing", False):
                raise ValueError(f"Step5 DSA owner {name} is missing its summary side storage.")
            continue
        backing = sidecar.kv_cache
        if not isinstance(backing, torch.Tensor) or backing.ndim != 2:
            raise ValueError(f"Step5 summary cache for {name} has not been bound.")
        if not isinstance(caches, (list, tuple)) or len(caches) != 2:
            raise ValueError(f"Step5 owner {name} must expose split K/V caches.")
        if caches[0].shape[0] != backing.shape[0] or caches[0].shape[2] != 1:
            raise ValueError("Step5 summary transfer requires one local KV head and one scheduler page per KV block.")
        if backing.data_ptr() % STEP5_BUFFER_ALIGNMENT or not backing.is_contiguous():
            raise ValueError("Step5 summary transfer requires contiguous, 2 MiB aligned backing.")
        transfer_caches[name] = (*caches, backing)
        summaries[name] = backing
    return transfer_caches, summaries
