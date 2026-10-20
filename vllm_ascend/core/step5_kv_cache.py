# SPDX-License-Identifier: Apache-2.0
"""Contiguous Step5 K/V views over the upstream shared group backing.

The byte permutation from page-interleaved K/V to K-then-V is identical
for every layer slot. It preserves exclusive block ownership across groups
only when all layers have identical, unpadded storage geometry.
"""

import torch
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)


def _layer_specs(groups):
    result = {}
    for group in groups:
        spec = group.kv_cache_spec
        for name in group.layer_names:
            if name in result:
                return None
            result[name] = (
                spec.kv_cache_specs[name]
                if isinstance(spec, UniformTypeKVCacheSpecs)
                else spec
            )
    return result


def supports_step5_shared_kv_backing(vllm_config, kv_cache_groups) -> bool:
    """Shared predicate for the worker budget and the MRv2 allocator.

    Other models, runners, padded/heterogeneous pages and distributed KV
    transport retain their existing allocation and budget behavior.
    Side-storage budgets are metadata, intentionally absent from geometry.
    """
    if not getattr(vllm_config, "use_v2_model_runner", False):
        return False
    model_config = vllm_config.model_config
    if getattr(model_config.hf_text_config, "model_type", None) != "step5":
        return False
    if vllm_config.kv_transfer_config is not None:
        return False
    additional = getattr(vllm_config, "additional_config", None) or {}
    if additional.get("enable_kvpp", False):
        return False
    if additional.get("sparse_kv_offload_config", {}).get("enabled", False):
        return False
    if (
        getattr(getattr(vllm_config, "attention_config", None), "hisparse_config", None)
        is not None
    ):
        return False
    parallel = vllm_config.parallel_config
    if any(
        getattr(parallel, field, 1) != 1
        for field in ("decode_context_parallel_size", "prefill_context_parallel_size")
    ):
        return False
    layout = vllm_config.cache_config.get_resolved_kv_cache_layout()
    if not (layout.is_layer_compact and layout.is_block_compact):
        return False
    if any(getattr(group, "host_resident", False) for group in kv_cache_groups):
        return False
    specs = _layer_specs(kv_cache_groups)
    if not specs:
        return False
    geometries = set()
    for spec in specs.values():
        # Exact types deliberately exclude MLA, quantized and compressed specs.
        if type(spec) not in (FullAttentionSpec, SlidingWindowSpec):
            return False
        head_size_v = getattr(spec, "head_size_v", spec.head_size)
        if (
            spec.block_size != 128
            or spec.head_size != 192
            or head_size_v != spec.head_size
            or spec.dtype != torch.bfloat16
            or spec.num_kv_heads <= 0
            or getattr(spec, "store_on_host", False)
            or getattr(spec, "kv_quant_mode", 0) != 0
            or getattr(spec, "num_head_slots", None) is not None
            or getattr(spec, "state_content_bytes", None) is not None
            or getattr(spec, "tokens_per_state", 1) != 1
            or getattr(spec, "page_size_padded", None) is not None
        ):
            return False
        dense_page = 2 * spec.block_size * spec.num_kv_heads * spec.head_size * 2
        if spec.page_size_bytes != dense_page:
            return False
        geometries.add(
            (spec.block_size, spec.num_kv_heads, spec.head_size, spec.dtype, dense_page)
        )
    return len(geometries) == 1


def allocate_step5_shared_kv_backing(kv_cache_config, device):
    """Validate descriptors first, then allocate one backing and K/V half views.

    Each returned pair contains byte tensors and follows the existing tuple
    reshape path. Native PA/GBSA and reshape-and-cache thus keep dense K/V.
    No summary pages or request journals alias this main backing.
    """
    specs = _layer_specs(kv_cache_config.kv_cache_groups)
    if not specs or kv_cache_config.num_blocks <= 0:
        raise ValueError("Step5 shared KV requires layers and a positive block pool.")
    if kv_cache_config.needs_base_kv_cache_zeroing:
        raise ValueError(
            "Step5 contiguous K/V backing does not support descriptor-based main-page zeroing."
        )
    sizes = {descriptor.size for descriptor in kv_cache_config.kv_cache_tensors}
    if len(sizes) != 1:
        raise ValueError("Step5 KV descriptors must reference one common backing size.")
    backing_size = sizes.pop()
    layer_size = kv_cache_config.num_blocks * next(iter(specs.values())).page_size_bytes
    if backing_size <= 0 or backing_size % layer_size:
        raise ValueError(
            "Step5 shared KV backing must contain positive, complete layer slots."
        )
    regions = {}
    for descriptor in kv_cache_config.kv_cache_tensors:
        for index, name in enumerate(descriptor.layers):
            if name not in specs or name in regions:
                raise ValueError(
                    f"Step5 shared KV descriptor has invalid layer {name!r}."
                )
            spec = specs[name]
            layer_size = kv_cache_config.num_blocks * spec.page_size_bytes
            start = descriptor.offset + index * descriptor.layer_stride
            if (
                descriptor.layer_stride != layer_size
                or descriptor.block_stride != spec.page_size_bytes
                or start < 0
                or start % layer_size
                or start + layer_size > backing_size
                or getattr(descriptor, "host_resident", False)
            ):
                raise ValueError(
                    f"Step5 shared KV descriptor is not a dense layer slot: {name!r}."
                )
            regions[name] = (start, layer_size)
    if set(regions) != set(specs):
        raise ValueError(
            "Step5 shared KV descriptors must cover every group layer exactly once."
        )
    for group in kv_cache_config.kv_cache_groups:
        starts = [regions[name][0] for name in group.layer_names]
        if len(starts) != len(set(starts)):
            raise ValueError("Step5 layers within one group cannot alias a KV slot.")
    backing = torch.zeros(backing_size, dtype=torch.int8, device=device)
    return {
        name: (
            backing[start : start + size // 2],
            backing[start + size // 2 : start + size],
        )
        for name, (start, size) in regions.items()
    }
