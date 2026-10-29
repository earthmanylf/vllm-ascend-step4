# SPDX-License-Identifier: Apache-2.0
"""Step5 native K/V views over upstream descriptor-backed shared storage.

BLNHC retains real layer pages inside global block tiles, including aligned PD
storage. The legacy compact permutation requires identical unpadded pages.
"""

import torch
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
    compute_layer_kv_cache_shape_bytes,
    compute_layout_strides,
)
from vllm.v1.kv_cache_layout import KVCacheLayout


def uses_step5_blnhc(vllm_config):
    """Select V2 BLNHC, including locally validated homogeneous PD profiles."""
    if not getattr(vllm_config, "use_v2_model_runner", False):
        return False
    if getattr(vllm_config.model_config.hf_text_config, "model_type", None) != "step5":
        return False
    additional = getattr(vllm_config, "additional_config", None) or {}
    parallel = vllm_config.parallel_config
    if (
        vllm_config.scheduler_config.disable_hybrid_kv_cache_manager
        or additional.get("enable_kvpp", False)
        or additional.get("sparse_kv_offload_config", {}).get("enabled", False)
        or getattr(getattr(vllm_config, "attention_config", None), "hisparse_config", None) is not None
        or any(
            getattr(parallel, field, 1) != 1
            for field in ("prefill_context_parallel_size", "decode_context_parallel_size")
        )
    ):
        raise ValueError("Step5 BLNHC currently requires V2 local KV with PCP/DCP=1 and no offload")
    validate_step5_kv_transfer_config(vllm_config)
    return True


def _validate_step5_blnhc_spec(spec):
    if type(spec) not in (FullAttentionSpec, SlidingWindowSpec, CircularBufferSpec):
        raise ValueError("Step5 BLNHC does not support this cache spec")
    if getattr(spec, "num_head_slots", None) is not None or getattr(spec, "state_content_bytes", None) is not None:
        raise ValueError("Step5 BLNHC does not support packed head slots or state content")
    ratio = getattr(spec, "tokens_per_state", 1)
    if spec.block_size != 128 or getattr(spec, "kv_quant_mode", 0) != 0:
        raise ValueError("Step5 BLNHC requires unquantized 128-token manager blocks")
    if getattr(spec, "page_size_padded", None) is not None:
        raise ValueError("Step5 BLNHC must retain each layer's real page size")
    if type(spec) is CircularBufferSpec:
        valid = spec.dtype == torch.bfloat16 and spec.head_size_v == 0 and ratio == 1
    elif spec.head_size_v == 0:
        valid = type(spec) is FullAttentionSpec and spec.dtype == torch.uint8 and spec.head_size == 256 and ratio == 8
    else:
        valid = spec.dtype == torch.bfloat16 and spec.head_size == spec.head_size_v == 192 and ratio == 1
    if not valid or spec.num_kv_heads != 1:
        raise ValueError("Step5 BLNHC cache geometry is unsupported")


def validate_step5_blnhc_descriptors(config, layout):
    """Validate upstream block tiles before allocation, without replanning."""
    if layout is not KVCacheLayout.BLNHC:
        raise ValueError("Step5 V2 requires the upstream BLNHC layout")
    specs = _layer_specs(config.kv_cache_groups)
    if not specs or type(config.num_blocks) is not int or config.num_blocks <= 0:
        raise ValueError("Step5 BLNHC requires layers and positive blocks")
    # The DSA summary layer replaces its BLNHC-strided 4 KiB pages with a dense
    # mirror (see models/step5/summary_cache._summary_table). That mirror only
    # becomes this layer's `kv_cache`, so a freshly allocated block is cleared
    # through the runner's KVBlockZeroer rather than by this layer. Step5's
    # bf16 main KV plus uint8 summaries already make the cache mixed-precision,
    # which is what enables zeroing today; fail loudly if that ever changes,
    # because a reused block would then serve a previous owner's summaries.
    if not getattr(config, "needs_kv_cache_zeroing", False):
        raise ValueError(
            "Step5 BLNHC requires worker-side KV block zeroing "
            "(KVCacheConfig.needs_kv_cache_zeroing); the dense summary mirror "
            "depends on it to clear reused blocks."
        )
    if any(getattr(group, "host_resident", False) for group in config.kv_cache_groups):
        raise ValueError("Step5 BLNHC does not support host-resident groups")
    regions = {}
    tile_stride = None
    for descriptor in config.kv_cache_tensors:
        if not descriptor.layers or getattr(descriptor, "host_resident", False):
            raise ValueError("Step5 BLNHC descriptor must contain device layers")
        stride = descriptor.block_stride
        if type(stride) is not int or stride <= 0 or descriptor.size != config.num_blocks * stride:
            raise ValueError("Step5 BLNHC descriptor size must match whole global block tiles")
        if tile_stride is not None and tile_stride != stride:
            raise ValueError("Step5 BLNHC descriptors must share one global block stride")
        tile_stride = stride
        for index, name in enumerate(descriptor.layers):
            if name not in specs or name in regions:
                raise ValueError("Step5 BLNHC descriptors contain unknown or duplicate layers")
            spec = specs[name]
            _validate_step5_blnhc_spec(spec)
            # Check native row alignment independently of the collected spec:
            # a missing backend customization must fail before allocation, not
            # later in GBSA's PA_BBND tiling parser.
            if spec.head_size_v > 0:
                native_alignment = spec.num_kv_heads * spec.head_size * spec.dtype.itemsize
                if stride % native_alignment:
                    raise ValueError(
                        f"Step5 BLNHC main KV {name} block stride {stride} bytes "
                        f"is not divisible by native row alignment {native_alignment} bytes"
                    )
            page = spec.page_size_bytes
            offset = descriptor.offset + index * descriptor.layer_stride
            if (
                type(descriptor.offset) is not int
                or descriptor.offset < 0
                or descriptor.layer_stride != page
                or offset < 0
                or offset + page > stride
                or offset % spec.dtype.itemsize
                or stride % spec.dtype.itemsize
                or stride % spec.get_block_stride_alignment()
            ):
                raise ValueError("Step5 BLNHC descriptor exceeds or misaligns its global block tile")
            regions[name] = (offset, offset + page)
    if set(regions) != set(specs):
        raise ValueError("Step5 BLNHC descriptors must cover every layer exactly once")
    # All groups address block b within [b*stride, (b+1)*stride).
    # Groups may overlay only the SAME block ID; layers within one group may not.
    for group in config.kv_cache_groups:
        spans = sorted(regions[name] for name in group.layer_names)
        if any(left[1] > right[0] for left, right in zip(spans, spans[1:])):
            raise ValueError("Step5 BLNHC layers in one manager group cannot overlap")
    return specs


def allocate_step5_pd_blnhc(config, device, layout, kernel_block_sizes):
    """Canonical upstream views on one aligned PD slab; never replan tiles."""
    specs = validate_step5_blnhc_descriptors(config, layout)
    if len(kernel_block_sizes) != len(config.kv_cache_groups) or any(size != 128 for size in kernel_block_sizes):
        raise ValueError("Step5 BLNHC currently requires 128-token kernel blocks for every group")
    size = config.kv_cache_tensors[0].size
    raw = torch.zeros(size + STEP5_PD_BUFFER_ALIGNMENT, dtype=torch.int8, device=device)
    # Offsets are absolute within the raw storage, including the aligned prefix.
    base_offset = (-raw.data_ptr()) % STEP5_PD_BUFFER_ALIGNMENT
    caches = {}
    for descriptor in config.kv_cache_tensors:
        for index, name in enumerate(descriptor.layers):
            spec = specs[name]
            shape = compute_layer_kv_cache_shape_bytes(spec, config.num_blocks, 128)
            strides = compute_layout_strides(
                spec,
                config.num_blocks,
                len(descriptor.layers),
                layout,
                128,
                fixed_strides=(descriptor.layer_stride, descriptor.block_stride, None, None, None),
            )
            offset = base_offset + descriptor.offset + index * descriptor.layer_stride
            caches[name] = raw.as_strided(shape, strides[1:], offset).view(spec.dtype)
    return caches


def step5_blnhc_layer_views(caches, specs):
    """Zero-copy native K/V planes inside each upstream BLNHC layer page.

    CANN FIA permits gaps only between blocks, not interleaved token rows.
    Reinterpret the content of each main page as dense K then dense V, while
    retaining the planner's block/layer offsets and its shared global-ID tile.
    Summary and ring pages retain their canonical content, including in PD.
    """
    if set(caches) != set(specs):
        raise ValueError("Step5 BLNHC allocation does not cover every layer")
    result = {}
    for name, cache in caches.items():
        spec = specs[name]
        expected = (
            cache.shape[0],
            spec.num_kv_heads,
            spec.block_size // getattr(spec, "tokens_per_state", 1),
            spec.head_size + spec.head_size_v,
        )
        if cache.dtype != spec.dtype or tuple(cache.shape) != expected:
            raise ValueError(f"Step5 BLNHC canonical cache shape mismatch: {name}")
        page_elements = expected[1] * expected[2] * expected[3]
        # BLNHC rows are dense; singleton strides do not affect addresses.
        inner_strides = (expected[3], expected[1] * expected[3], 1)
        if cache.stride(0) < page_elements or any(
            size > 1 and cache.stride(dim) != stride
            for dim, (size, stride) in enumerate(zip(expected[1:], inner_strides), 1)
        ):
            raise ValueError(f"Step5 BLNHC canonical cache blocks must be dense and non-overlapping: {name}")
        if spec.head_size_v == 0:
            result[name] = cache
        else:
            shape = (cache.shape[0], spec.block_size, spec.num_kv_heads, spec.head_size)
            block_stride = cache.stride(0)
            strides = (block_stride, spec.num_kv_heads * spec.head_size, spec.head_size, 1)
            offset = cache.storage_offset()
            k = cache.as_strided(shape, strides, offset)
            v = cache.as_strided(shape, strides, offset + spec.block_size * spec.num_kv_heads * spec.head_size)
            result[name] = (k, v)
    return result


STEP5_PD_BUFFER_ALIGNMENT = 2 * 1024 * 1024


def validate_step5_kv_transfer_config(vllm_config) -> None:
    """Local capability boundary; peer topology is checked by the handshake."""
    transfer = vllm_config.kv_transfer_config
    if transfer is None or not transfer.is_kv_transfer_instance:
        return
    if transfer.kv_connector not in ("MooncakeConnectorV2", "MooncakePullConnector") or (
        transfer.kv_connector_module_path is not None
    ):
        raise ValueError(
            "Step5 KV transfer requires the registered Ascend MooncakeConnectorV2 or "
            "MooncakePullConnector implementation; other connectors and module overrides "
            "do not implement the main KV, summary and ring transfer contract."
        )
    if transfer.kv_role not in ("kv_producer", "kv_consumer"):
        raise ValueError(
            "Step5 KV transfer requires kv_producer or kv_consumer; kv_both is not supported by Mooncake pull."
        )
    parallel = vllm_config.parallel_config
    pp_producer = (
        transfer.kv_role == "kv_producer"
        and getattr(parallel, "pipeline_parallel_size", None) == 4
        and getattr(parallel, "tensor_parallel_size", None) == 4
        and getattr(parallel, "data_parallel_size", None) == 1
        and getattr(parallel, "data_parallel_size_local", None) == 1
        and getattr(parallel, "enable_expert_parallel", False)
        and getattr(vllm_config, "use_v2_model_runner", False)
    )
    if parallel.pipeline_parallel_size != 1 and not pp_producer:
        raise ValueError("Step5 KV transfer requires pipeline_parallel_size=1 or the PP4/TP4 producer profile.")
    for field in ("prefill_context_parallel_size", "decode_context_parallel_size"):
        if getattr(parallel, field) != 1:
            raise ValueError(f"Step5 KV transfer requires {field}=1.")
    spec = vllm_config.speculative_config
    if spec is not None:
        if (
            not getattr(vllm_config, "use_v2_model_runner", False)
            or getattr(spec, "method", None) != "mtp"
            or getattr(spec, "num_speculative_tokens", None) != 3
        ):
            raise ValueError(
                "Step5 KV transfer currently supports only V2 eager MTP3 or the explicit graph/async profile."
            )
        target_eager = getattr(vllm_config.model_config, "enforce_eager", False)
        draft_eager = getattr(spec, "enforce_eager", False)
        if not (target_eager and draft_eager):
            compilation = getattr(vllm_config, "compilation_config", None)
            captures = getattr(compilation, "cudagraph_capture_sizes", None)
            graph_profile = (
                not target_eager
                and not draft_eager
                and getattr(compilation, "mode", None) == 3
                and getattr(getattr(compilation, "cudagraph_mode", None), "name", None) == "FULL_DECODE_ONLY"
                and isinstance(captures, list)
                and bool(captures)
                and max(captures) == 512
                and all(type(size) is int and size > 0 and size % 4 == 0 for size in captures)
                and getattr(parallel, "tensor_parallel_size", None) == 4
                and (
                    pp_producer
                    or (
                        (
                            getattr(parallel, "data_parallel_size", None) == 2
                            or (
                                transfer.kv_role == "kv_consumer" and getattr(parallel, "data_parallel_size", None) == 4
                            )
                        )
                        and getattr(parallel, "data_parallel_size_local", None) == 2
                    )
                )
                and getattr(parallel, "enable_expert_parallel", False)
                and getattr(vllm_config.scheduler_config, "async_scheduling", False)
                and getattr(vllm_config.scheduler_config, "max_num_seqs", None) == (32 if pp_producer else 256)
            )
            if not graph_profile:
                raise ValueError(
                    "Step5 MTP3 PD requires both eager runners or an explicit supported graph/async topology."
                )
        if getattr(spec, "num_speculative_tokens_per_batch_size", None) is not None:
            raise ValueError("Step5 MTP3 PD uses native static MTP3 without a batch-size schedule.")
        draft = getattr(getattr(spec, "draft_model_config", None), "hf_text_config", None)
        hidden_size = getattr(vllm_config.model_config.hf_text_config, "hidden_size", None)
        if (
            getattr(draft, "model_type", None) != "step5_mtp"
            or getattr(draft, "num_nextn_predict_layers", None) != 3
            or type(hidden_size) is not int
            or hidden_size <= 0
            or getattr(draft, "hidden_size", None) != hidden_size
        ):
            raise ValueError("Step5 MTP3 PD requires three matching Step5MTP draft modules.")
    if vllm_config.cache_config.enable_prefix_caching and not getattr(vllm_config, "use_v2_model_runner", False):
        raise ValueError("Step5 KV transfer prefix caching requires the V2 BLNHC model runner.")
    # A multimodal Step5 checkpoint is fine: the vision tower is never part of the
    # KV transfer path, and rejecting it here used to force a text-only target plus
    # a separate MTP draft view, which the vendor's single unified checkpoint does
    # not need (its MTP layers ship in the same shards as the text layers).


def step5_tailring_alignment_reserve(vllm_config, groups) -> int:
    """One alignment allocation per eligible PD slab, not per layer/group."""
    transfer = vllm_config.kv_transfer_config
    if transfer is None or not transfer.is_kv_transfer_instance:
        return 0
    if vllm_config.cache_config.get_resolved_kv_cache_layout() is KVCacheLayout.BLNHC:
        if not uses_step5_blnhc(vllm_config):
            return 0
        specs = _layer_specs(groups)
        if not specs or any(getattr(group, "host_resident", False) for group in groups):
            raise ValueError("Step5 PD BLNHC requires nonempty device groups")
        for spec in specs.values():
            _validate_step5_blnhc_spec(spec)
        return STEP5_PD_BUFFER_ALIGNMENT
    if supports_step5_tailring_shared_backing(vllm_config, groups):
        return STEP5_PD_BUFFER_ALIGNMENT
    return 0


def _layer_specs(groups):
    result = {}
    for group in groups:
        spec = group.kv_cache_spec
        for name in group.layer_names:
            if name in result:
                return None
            result[name] = spec.kv_cache_specs[name] if isinstance(spec, UniformTypeKVCacheSpecs) else spec
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
    if getattr(getattr(vllm_config, "attention_config", None), "hisparse_config", None) is not None:
        return False
    parallel = vllm_config.parallel_config
    if any(
        getattr(parallel, field, 1) != 1 for field in ("decode_context_parallel_size", "prefill_context_parallel_size")
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
        geometries.add((spec.block_size, spec.num_kv_heads, spec.head_size, spec.dtype, dense_page))
    return len(geometries) == 1


def supports_step5_tailring_shared_backing(vllm_config, groups):
    """Whether Step5 can consume descriptor-backed, page-interleaved caches."""
    if not getattr(vllm_config, "use_v2_model_runner", False):
        return False
    if getattr(vllm_config.model_config.hf_text_config, "model_type", None) != "step5":
        return False
    if vllm_config.kv_transfer_config is not None:
        if not vllm_config.kv_transfer_config.is_kv_transfer_instance:
            return False
        try:
            validate_step5_kv_transfer_config(vllm_config)
        except ValueError:
            return False
    additional = getattr(vllm_config, "additional_config", None) or {}
    if additional.get("enable_kvpp", False) or additional.get("sparse_kv_offload_config", {}).get("enabled", False):
        return False
    if getattr(getattr(vllm_config, "attention_config", None), "hisparse_config", None) is not None:
        return False
    parallel = vllm_config.parallel_config
    if any(
        getattr(parallel, name, 1) != 1 for name in ("decode_context_parallel_size", "prefill_context_parallel_size")
    ):
        return False
    layout = vllm_config.cache_config.get_resolved_kv_cache_layout()
    if not (layout.is_layer_compact and layout.is_block_compact):
        return False
    if any(getattr(group, "host_resident", False) for group in groups):
        return False
    specs = _layer_specs(groups)
    if not specs or not any(type(spec) is CircularBufferSpec for spec in specs.values()):
        return False
    pages = set()
    for spec in specs.values():
        if type(spec) not in (FullAttentionSpec, SlidingWindowSpec, CircularBufferSpec):
            return False
        if spec.block_size != 128 or getattr(spec, "store_on_host", False) or getattr(spec, "kv_quant_mode", 0) != 0:
            return False
        if getattr(spec, "num_head_slots", None) is not None or getattr(spec, "state_content_bytes", None) is not None:
            return False
        ratio = getattr(spec, "tokens_per_state", 1)
        if spec.head_size_v == 0:
            if spec.num_kv_heads != 1:
                return False
            if type(spec) is CircularBufferSpec:
                if spec.dtype != torch.bfloat16 or ratio != 1 or spec.page_size_padded is not None:
                    return False
            elif (
                type(spec) is not FullAttentionSpec or spec.dtype != torch.uint8 or ratio != 8 or spec.head_size != 256
            ):
                return False
        elif spec.dtype != torch.bfloat16 or spec.head_size != 192 or spec.head_size_v != 192 or ratio != 1:
            return False
        pages.add(spec.page_size_bytes)
    return len(pages) == 1


def allocate_step5_tailring_shared_backing(config, device, *, align_for_transfer=False):
    """Allocate one pool with descriptor-preserving main KV and sidecar views."""
    specs = _layer_specs(config.kv_cache_groups)
    sizes = {descriptor.size for descriptor in config.kv_cache_tensors}
    if not specs or len(sizes) != 1 or config.num_blocks <= 0:
        raise ValueError("Step5 TailRing requires one nonempty shared pool")
    size = sizes.pop()
    if size <= 0:
        raise ValueError("Step5 TailRing backing size must be positive")
    regions = {}
    pages = {spec.page_size_bytes for spec in specs.values()}
    if len(pages) != 1:
        raise ValueError("Step5 TailRing requires uniform physical pages")
    for descriptor in config.kv_cache_tensors:
        num_blocks_of = getattr(config, "num_blocks_of", None)
        if callable(num_blocks_of) and num_blocks_of(descriptor) != config.num_blocks:
            raise ValueError("Step5 TailRing descriptor block count differs from the device pool")
        if descriptor.layer_stride < 0:
            raise ValueError("Step5 TailRing layer stride must be nonnegative")
        for index, name in enumerate(descriptor.layers):
            if name not in specs or name in regions or getattr(descriptor, "host_resident", False):
                raise ValueError("Invalid Step5 TailRing descriptor layer")
            spec = specs[name]
            page = spec.page_size_bytes
            itemsize = spec.dtype.itemsize
            ratio = getattr(spec, "tokens_per_state", 1)
            heads, dim, dim_v = spec.num_kv_heads, spec.head_size, spec.head_size_v
            if (
                page <= 0
                or page % itemsize
                or size % itemsize
                or ratio <= 0
                or spec.block_size <= 0
                or spec.block_size % ratio
                or heads <= 0
                or dim <= 0
                or dim_v < 0
            ):
                raise ValueError("Invalid Step5 TailRing page or state geometry")
            payload = spec.block_size // ratio * heads * (dim + dim_v) * itemsize
            if payload > page or (type(spec) is CircularBufferSpec and payload != page):
                raise ValueError("Step5 TailRing payload exceeds or pads its physical page")
            if dim_v and (dim_v != dim or ratio != 1):
                raise ValueError("Step5 TailRing main K/V geometry must be symmetric")
            if descriptor.layer_stride % page:
                raise ValueError("Step5 TailRing layer stride must align to physical pages")
            offset = descriptor.offset + index * descriptor.layer_stride
            end = offset + (config.num_blocks - 1) * descriptor.block_stride + page
            if offset < 0 or descriptor.block_stride != page or offset % page or end > size:
                raise ValueError("Step5 TailRing descriptor exceeds or misaligns the shared pool")
            regions[name] = (offset, page)
    if set(regions) != set(specs):
        raise ValueError("Step5 TailRing descriptors must cover every layer")
    for group in config.kv_cache_groups:
        spans = sorted(
            (regions[name][0], regions[name][0] + config.num_blocks * regions[name][1]) for name in group.layer_names
        )
        if any(left[1] > right[0] for left, right in zip(spans, spans[1:])):
            raise ValueError("Step5 layers in one group cannot overlap")
    reserve = STEP5_PD_BUFFER_ALIGNMENT if align_for_transfer else 0
    raw = torch.zeros(size + reserve, dtype=torch.int8, device=device)
    if align_for_transfer:
        raw = raw.narrow(0, (-raw.data_ptr()) % STEP5_PD_BUFFER_ALIGNMENT, size)
    caches = {}
    for name, (offset, page) in regions.items():
        spec = specs[name]
        typed = raw.view(spec.dtype)
        itemsize = spec.dtype.itemsize
        storage_offset = typed.storage_offset() + offset // itemsize
        heads, dim = spec.num_kv_heads, spec.head_size
        ratio = getattr(spec, "tokens_per_state", 1)
        states = spec.block_size // ratio
        if spec.head_size_v == 0:
            caches[name] = typed.as_strided(
                (config.num_blocks, heads, states, dim),
                (page // itemsize, states * dim, dim, 1),
                storage_offset,
            )
        else:
            strides = (page // itemsize, heads * dim, dim, 1)
            shape = (config.num_blocks, spec.block_size, heads, dim)
            k = typed.as_strided(shape, strides, storage_offset)
            v = typed.as_strided(shape, strides, storage_offset + spec.block_size * heads * dim)
            caches[name] = (k, v)
    return caches


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
        raise ValueError("Step5 contiguous K/V backing does not support descriptor-based main-page zeroing.")
    sizes = {descriptor.size for descriptor in kv_cache_config.kv_cache_tensors}
    if len(sizes) != 1:
        raise ValueError("Step5 KV descriptors must reference one common backing size.")
    backing_size = sizes.pop()
    layer_size = kv_cache_config.num_blocks * next(iter(specs.values())).page_size_bytes
    if backing_size <= 0 or backing_size % layer_size:
        raise ValueError("Step5 shared KV backing must contain positive, complete layer slots.")
    regions = {}
    for descriptor in kv_cache_config.kv_cache_tensors:
        for index, name in enumerate(descriptor.layers):
            if name not in specs or name in regions:
                raise ValueError(f"Step5 shared KV descriptor has invalid layer {name!r}.")
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
                raise ValueError(f"Step5 shared KV descriptor is not a dense layer slot: {name!r}.")
            regions[name] = (start, layer_size)
    if set(regions) != set(specs):
        raise ValueError("Step5 shared KV descriptors must cover every group layer exactly once.")
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
