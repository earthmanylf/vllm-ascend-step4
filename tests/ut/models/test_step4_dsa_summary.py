#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
"""CPU UT: Step4 DSA paged summary sidecar plumbing (no NPU required).

Covers the host-side contract of the paged summary migration:
capacity/budget accounting on ``AscendStep4DSAImpl`` and the
extra-budget wiring on ``AscendStep4DSAAttention.get_kv_cache_spec``.
Kernel-level numerics live in
``tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_step4_dsa_summary_cache.py``.
"""

import torch

from vllm_ascend.models.step4.dsa_attention import AscendStep4DSAImpl


def _make_impl(**overrides) -> AscendStep4DSAImpl:
    args = dict(
        num_heads=8,
        head_size=192,
        scale=0.1,
        num_kv_heads=1,
        kv_cache_dtype="auto",
        topk_regions=16,
        region_size=8,
        proxy_dim=256,
        max_num_seqs=4,
        max_model_len=4096,
        block_size=128,
        max_num_batched_tokens=1024,
        decode_out_dtype=torch.bfloat16,
    )
    args.update(overrides)
    return AscendStep4DSAImpl(*[], **args)


def test_active_region_capacity_policy():
    impl = _make_impl(max_num_seqs=7)
    # 8x max_num_seqs, matching the upstream capacity policy.
    assert impl._csa_active_region_capacity() == 56


def test_block_size_must_align_with_region():
    try:
        _make_impl(block_size=20)
    except ValueError:
        return
    raise AssertionError("unaligned block_size must be rejected")


def test_fixed_runtime_state_budget_matches_layout():
    impl = _make_impl(max_num_seqs=4)
    capacity = 8 * 4
    proxy_dim = 256
    region_block_size = 8
    expected = (
        capacity * 4  # active_region_ids (int32)
        + 4  # allocation_success
        + 2 * capacity * proxy_dim * 4  # numerator + denominator (fp32)
        + capacity * 4  # max_logits (fp32)
        + 2 * capacity * region_block_size * proxy_dim * 2  # staging k/z (bf16)
        + capacity * region_block_size  # staging valid (uint8)
    )
    assert impl.csa_fixed_runtime_state_size_bytes() == expected


def test_selector_region_capacity():
    impl = _make_impl(max_model_len=4096, region_size=8)
    assert impl._max_regions == 512


def test_spec_carries_side_storage_budget():
    """The DSA layer's KV spec must carry the summary sidecar budget."""
    from unittest.mock import MagicMock

    from vllm_ascend.models.step4.model import AscendStep4DSAAttention

    layer = AscendStep4DSAAttention.__new__(AscendStep4DSAAttention)
    torch.nn.Module.__init__(layer)
    layer.num_kv_heads = 1
    layer.head_dim = 192
    layer.kv_cache_torch_dtype = torch.bfloat16
    layer.kv_cache_requires_zeroing = True
    layer.kv_cache_prefix_recompute_tokens = 128
    layer.kv_cache_extra_budget_page_size_bytes = lambda block_size: 2176
    layer.kv_cache_extra_budget_fixed_size_bytes = lambda block_size: 4096

    vllm_config = MagicMock()
    vllm_config.cache_config.block_size = 128
    spec = layer.get_kv_cache_spec(vllm_config)

    side = spec.side_storage_spec
    assert side.requires_zeroing
    assert side.extra_budget_page_size_bytes == 2176
    assert side.extra_budget_fixed_size_bytes == 4096
    assert side.prefix_cache_recompute_tokens == 128
