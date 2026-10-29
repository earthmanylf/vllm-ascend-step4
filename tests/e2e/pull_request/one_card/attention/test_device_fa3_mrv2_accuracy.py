# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device accuracy for the actual FA3 builder/forward boundary, not a model test."""

from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("window", [None, 128])
def test_mixed_query_lengths_high_physical_page_ids(window):
    pytest.importorskip("torch_npu")
    pytest.importorskip("flash_attn_npu_3")
    from vllm_ascend.attention.flash_attention_v3 import (
        AscendFlashAttentionImpl,
        AscendFlashAttentionMetadataBuilder,
    )

    torch.manual_seed(43)
    device = torch.device("npu:0")
    dtype = torch.bfloat16
    num_blocks, block_size, heads, kv_heads, dim = 12, 128, 8, 2, 128
    q_cpu = torch.randn(4, heads, dim).to(dtype)
    k_cpu = torch.randn(num_blocks, block_size, kv_heads, dim).to(dtype)
    v_cpu = torch.randn_like(k_cpu)
    # Physical block IDs deliberately exceed the per-sequence table width.
    tables_cpu = torch.tensor([[11, 4, 0], [7, 2, 10]], dtype=torch.int32)
    lengths = [256, 384]
    query_lengths = [1, 3]
    query_start = [0, 1, 4]
    cfg = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=2),
        compilation_config=SimpleNamespace(cudagraph_capture_sizes=[]),
        model_config=SimpleNamespace(runner_type="generate"),
        speculative_config=None,
    )
    # Isolate the builder/forward boundary from loading a full model/config.
    impl = AscendFlashAttentionImpl.__new__(AscendFlashAttentionImpl)
    impl.num_heads, impl.num_kv_heads, impl.head_size = heads, kv_heads, dim
    impl.scale = dim**-0.5
    impl.logits_soft_cap = 0.0
    impl.window_size = (-1, -1) if window is None else (window - 1, 0)
    cache = (k_cpu.to(device), v_cpu.to(device))
    builder = AscendFlashAttentionMetadataBuilder(SimpleNamespace(block_size=block_size), ["layer"], cfg, device)
    builder.bind_fa3_kv_cache({"layer": SimpleNamespace(impl=impl, kv_cache=cache)})
    common = SimpleNamespace(
        num_reqs=2,
        query_start_loc=torch.tensor(query_start, dtype=torch.int32, device=device),
        seq_lens=torch.tensor(lengths, dtype=torch.int32, device=device),
        block_table_tensor=tables_cpu.to(device),
        slot_mapping=torch.zeros(4, dtype=torch.int64, device=device),
        num_actual_tokens=4,
        num_input_tokens=4,
        max_query_len=3,
        causal=True,
        attn_state=None,
    )
    metadata = builder.build(0, common)
    query = q_cpu.to(device)
    output = torch.empty_like(query)
    impl.forward_impl(query, None, None, cache, metadata, output)
    reference = []
    for request, (kv_len, q_len) in enumerate(zip(lengths, query_lengths)):
        blocks = tables_cpu[request].long()
        keys = k_cpu[blocks].flatten(0, 1)[:kv_len].repeat_interleave(heads // kv_heads, dim=1).float()
        values = v_cpu[blocks].flatten(0, 1)[:kv_len].repeat_interleave(heads // kv_heads, dim=1).float()
        for offset in range(q_len):
            row = query_start[request] + offset
            position = kv_len - q_len + offset
            start = 0 if window is None else max(0, position - window + 1)
            scores = torch.einsum("hd,khd->hk", q_cpu[row].float(), keys[start : position + 1]) * impl.scale
            reference.append(torch.einsum("hk,khd->hd", scores.softmax(-1), values[start : position + 1]))
    torch.testing.assert_close(output.cpu().float(), torch.stack(reference), rtol=0.03, atol=0.03)
