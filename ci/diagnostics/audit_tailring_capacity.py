"""Host-only arithmetic replay of current TailRing cache spec/planner, no model or NPU allocation."""

import json
import math
from types import SimpleNamespace as NS

import torch
from vllm.v1.attention.backend import MultipleOf
from vllm.v1.core.kv_cache_planning import _get_kv_cache_groups_uniform_page_size
from vllm.v1.kv_cache_interface import CircularBufferSpec, FullAttentionSpec, SlidingWindowSpec

page = 128 * 1 * 192 * 2 * 2
specs = {}
for i in range(23):
    specs[f"dsa{i}"] = FullAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=192,
        head_size_v=192,
        dtype=torch.bfloat16,
        block_stride_alignment=MultipleOf(128),
    )
    specs[f"summary{i}"] = FullAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=256,
        head_size_v=0,
        dtype=torch.uint8,
        tokens_per_state=8,
        page_size_padded=page,
    )
    specs[f"ring{i}"] = CircularBufferSpec(
        block_size=128, num_kv_heads=1, head_size=384, head_size_v=0, dtype=torch.bfloat16, block_stride_alignment=1024
    )
for i in range(72):
    specs[f"swa{i}"] = SlidingWindowSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=192,
        head_size_v=192,
        dtype=torch.bfloat16,
        sliding_window=512,
        extra_retained_tokens=3,
        block_stride_alignment=MultipleOf(128),
    )
groups = _get_kv_cache_groups_uniform_page_size(specs)
cfg = NS(
    model_config=NS(max_model_len=1048576),
    max_in_flight_tokens=32768,
    parallel_config=NS(decode_context_parallel_size=1),
)
rows = []
for group in groups:
    s = group.kv_cache_spec
    required = math.ceil(s.max_memory_usage_bytes(cfg) / s.page_size_bytes)
    rows.append(
        dict(
            type=type(s).__name__,
            layers=len(group.layer_names),
            block_size=s.block_size,
            page_bytes=s.page_size_bytes,
            real_page_bytes=s.real_page_size_bytes,
            blocks_per_1M_request=required,
        )
    )
reported = 1421378
d = sum(x["blocks_per_1M_request"] for x in rows)
# The integer range of block counts consistent with the observed printed token capacity.
possible = [n for n in range(20000, 30000) if int(n / d * 1048576) == reported]
sequence = 131072 + 2048
full_per_req = math.ceil(sequence / 128)
summary_per_req = full_per_req
ring_per_req = 1
# Decode-time SWA resident bound conservatively includes an in-flight batch of 64 tokens (16 req *4).
swa_per_req = math.ceil((511 + 3 + 64) / 128) + 1
decode_per_req = full_per_req + summary_per_req + ring_per_req + 4 * swa_per_req
pool_block_bytes = 23 * page
print(
    json.dumps(
        dict(
            scope=(
                "CPU replay of installed planner with MTP3 retention and seven real groups; "
                "six cacheable groups appear in alignment log because private ring filtered. "
                "Per-DP pool23631 inferred exactly from live kv_cache_max_concurrency; "
                "metrics num_gpu_blocks189046 is DP sum, not per-DP."
            ),
            model_weight_gib_observed=27.66,
            kv_budget_gib_observed=49.76,
            groups=rows,
            blocks_per_1M_request=d,
            printed_capacity_tokens=reported,
            compatible_live_block_counts=possible,
            pool_bytes_per_block=pool_block_bytes,
            bs128_balanced_requests_per_dp=16,
            sequence_len=sequence,
            decode_blocks_per_request=decode_per_req,
            decode_pool_gib_for16=16 * decode_per_req * pool_block_bytes / 2**30,
            main_pool_gib_for16=16 * full_per_req * pool_block_bytes / 2**30,
            summary_pool_gib_for16=16 * summary_per_req * pool_block_bytes / 2**30,
            summary_real_to_padded_ratio=page / (128 // 8 * 256),
            note=(
                "Summary is independently addressed and padded to main page; "
                "startup maxlen-derived token capacity is not linear residency proof. "
                "Do not bypass guards without case-specific real group evidence."
            ),
        ),
        indent=2,
    )
)
