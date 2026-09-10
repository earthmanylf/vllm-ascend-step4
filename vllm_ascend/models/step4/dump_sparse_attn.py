"""Dump sparse_attention_prefill / sparse_attention_decode kernel inputs to .pt files.

Usage:
    # Set the env var before launching vLLM, then run inference.
    export VLLM_STEP4_DUMP_SPARSE_ATTN=1
    export VLLM_STEP4_DUMP_DIR=/tmp/step4_sparse_dump

    # After inference, inspect saved tensors:
    #   /tmp/step4_sparse_dump/prefill_0/
    #   /tmp/step4_sparse_dump/decode_0/
    #
    # Each directory contains:
    #   query.pt, key_cache.pt, value_cache.pt, packed_regions.pt,
    #   region_counts.pt, [kv_seqlens.pt], out.pt, lse.pt, meta.json
"""

import json
import os
import traceback
from pathlib import Path

_DUMP_ENABLED = os.environ.get("VLLM_STEP4_DUMP_SPARSE_ATTN", "0") == "1"
_DUMP_DIR = os.environ.get("VLLM_STEP4_DUMP_DIR", "/tmp/step4_sparse_dump")
_prefill_dumped = False
_decode_dumped = False


def dump_prefill(
    query,
    key_cache,
    value_cache,
    packed_regions,
    region_counts,
    out,
    lse,
    *,
    num_kv_groups,
    region_size,
    softmax_scale,
    block_regions,
    kernel_kwargs,
):
    if not _DUMP_ENABLED:
        return
    global _prefill_dumped
    if _prefill_dumped:
        return
    _prefill_dumped = True
    try:
        dump_dir = Path(_DUMP_DIR) / "prefill"
        dump_dir.mkdir(parents=True, exist_ok=True)

        def _save(name, tensor):
            if tensor is None:
                return
            if hasattr(tensor, "contiguous"):
                tensor = tensor.contiguous().cpu()
            torch.save(tensor, dump_dir / f"{name}.pt")

        _save("query", query)
        _save("key_cache", key_cache)
        _save("value_cache", value_cache)
        _save("packed_regions", packed_regions)
        _save("region_counts", region_counts)
        _save("out", out)
        _save("lse", lse)

        meta = {
            "kernel": "_sparse_attn_prefill_kernel",
            "num_kv_groups": num_kv_groups,
            "region_size": region_size,
            "softmax_scale": softmax_scale,
            "block_regions": block_regions,
            "kernel_kwargs": {k: str(v) for k, v in kernel_kwargs.items()},
            "tensor_shapes": {
                "query": list(query.shape) if query is not None else None,
                "key_cache": list(key_cache.shape) if key_cache is not None else None,
                "value_cache": list(value_cache.shape) if value_cache is not None else None,
                "packed_regions": list(packed_regions.shape) if packed_regions is not None else None,
                "region_counts": list(region_counts.shape) if region_counts is not None else None,
                "out": list(out.shape) if out is not None else None,
                "lse": list(lse.shape) if lse is not None else None,
            },
            "tensor_dtypes": {
                "query": str(query.dtype) if query is not None else None,
                "key_cache": str(key_cache.dtype) if key_cache is not None else None,
                "value_cache": str(value_cache.dtype) if value_cache is not None else None,
                "packed_regions": str(packed_regions.dtype) if packed_regions is not None else None,
                "region_counts": str(region_counts.dtype) if region_counts is not None else None,
                "out": str(out.dtype) if out is not None else None,
                "lse": str(lse.dtype) if lse is not None else None,
            },
            "tensor_strides": {
                "query": list(query.stride()) if query is not None else None,
                "key_cache": list(key_cache.stride()) if key_cache is not None else None,
                "value_cache": list(value_cache.stride()) if value_cache is not None else None,
                "packed_regions": list(packed_regions.stride()) if packed_regions is not None else None,
                "out": list(out.stride()) if out is not None else None,
                "lse": list(lse.stride()) if lse is not None else None,
            },
        }
        with open(dump_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    except Exception:
        traceback.print_exc()


def dump_decode(
    query,
    key_cache,
    value_cache,
    packed_regions,
    region_counts,
    kv_seqlens,
    out,
    lse,
    *,
    num_kv_groups,
    region_size,
    softmax_scale,
    num_splits,
    block_regions,
    kernel_kwargs,
):
    if not _DUMP_ENABLED:
        return
    global _decode_dumped
    if _decode_dumped:
        return
    _decode_dumped = True
    try:
        dump_dir = Path(_DUMP_DIR) / "decode"
        dump_dir.mkdir(parents=True, exist_ok=True)

        def _save(name, tensor):
            if tensor is None:
                return
            if hasattr(tensor, "contiguous"):
                tensor = tensor.contiguous().cpu()
            torch.save(tensor, dump_dir / f"{name}.pt")

        _save("query", query)
        _save("key_cache", key_cache)
        _save("value_cache", value_cache)
        _save("packed_regions", packed_regions)
        _save("region_counts", region_counts)
        _save("kv_seqlens", kv_seqlens)
        if out is not None and out.ndim == 4 and out.shape[0] == 1:
            _save("out", out[0])
            _save("lse", lse[0] if lse is not None else None)
        else:
            _save("out", out)
            _save("lse", lse)

        meta = {
            "kernel": "_sparse_attn_decode_kernel",
            "num_kv_groups": num_kv_groups,
            "region_size": region_size,
            "softmax_scale": softmax_scale,
            "num_splits": num_splits,
            "block_regions": block_regions,
            "kernel_kwargs": {k: str(v) for k, v in kernel_kwargs.items()},
            "tensor_shapes": {
                "query": list(query.shape) if query is not None else None,
                "key_cache": list(key_cache.shape) if key_cache is not None else None,
                "value_cache": list(value_cache.shape) if value_cache is not None else None,
                "packed_regions": list(packed_regions.shape) if packed_regions is not None else None,
                "region_counts": list(region_counts.shape) if region_counts is not None else None,
                "kv_seqlens": list(kv_seqlens.shape) if kv_seqlens is not None else None,
                "out": list(out.shape) if out is not None else None,
                "lse": list(lse.shape) if lse is not None else None,
            },
            "tensor_dtypes": {
                "query": str(query.dtype) if query is not None else None,
                "key_cache": str(key_cache.dtype) if key_cache is not None else None,
                "value_cache": str(value_cache.dtype) if value_cache is not None else None,
                "packed_regions": str(packed_regions.dtype) if packed_regions is not None else None,
                "region_counts": str(region_counts.dtype) if region_counts is not None else None,
                "kv_seqlens": str(kv_seqlens.dtype) if kv_seqlens is not None else None,
                "out": str(out.dtype) if out is not None else None,
                "lse": str(lse.dtype) if lse is not None else None,
            },
            "tensor_strides": {
                "query": list(query.stride()) if query is not None else None,
                "key_cache": list(key_cache.stride()) if key_cache is not None else None,
                "value_cache": list(value_cache.stride()) if value_cache is not None else None,
                "packed_regions": list(packed_regions.stride()) if packed_regions is not None else None,
                "out": list(out.stride()) if out is not None else None,
                "lse": list(lse.stride()) if lse is not None else None,
            },
        }
        with open(dump_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    except Exception:
        traceback.print_exc()
