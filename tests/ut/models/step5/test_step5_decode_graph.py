# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contracts for the actual BLNHC graph descriptor preparation methods."""

import ast
from dataclasses import dataclass
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest
import torch

SOURCE = Path(__file__).resolve().parents[4] / "vllm_ascend/models/step5/dsa_attention.py"


def builder():
    tree = ast.parse(SOURCE.read_text())
    names = {"AscendStep5DSADecodeMetadata"}
    methods = {"_prepare_decode_graph", "_prepare_gbsa_metadata", "_build_gbsa_metadata"}
    body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in names:
            body.append(node)
        elif isinstance(node, ast.ClassDef) and node.name == "AscendStep5DSAMetadataBuilder":
            body.extend(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name in methods)
    namespace = dict(
        torch=torch,
        dataclass=dataclass,
        gbsa_expected_counts=lambda positions, **kwargs: positions.unsqueeze(0),
        gbsa_sparse_attention_metadata=lambda counts, **kwargs: torch.cat((counts.flatten(), kwargs["seqused_kv"])),
    )
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), str(SOURCE), "exec"), namespace)  # noqa: S102
    result = SimpleNamespace(
        reorder_batch_threshold=4,
        gbsa_block_size=128,
        gbsa_num_heads=8,
        gbsa_num_kv_groups=1,
        gbsa_head_size=128,
        gbsa_region_size=8,
        gbsa_topk=512,
        supports_padded_decode_graph=True,
        decode_graph_buffers={},
        gbsa_meta_buffers={},
    )
    for name in methods:
        setattr(result, name, MethodType(namespace[name], result))
    return result


def common(real, *, padded=2, width=4, columns=8):
    return SimpleNamespace(
        num_reqs=padded,
        max_query_len=width,
        num_input_tokens=padded * width,
        num_actual_tokens=real * width,
        causal=True,
        query_start_loc_cpu=torch.arange(padded + 1, dtype=torch.int32) * width,
        query_start_loc=torch.arange(padded + 1, dtype=torch.int32) * width,
        seq_lens=torch.tensor([19] * real + [0] * (padded - real), dtype=torch.int32),
        _seq_lens_cpu=torch.tensor([19] * real + [0] * (padded - real), dtype=torch.int32),
        block_table_tensor=torch.arange(padded * columns, dtype=torch.int32).reshape(padded, columns),
        slot_mapping=torch.arange(padded * width, dtype=torch.int32),
    )


@pytest.mark.parametrize("width", [1, 4])
def test_replay_refreshes_the_captured_bucket_in_place(width):
    b = builder()
    captured = b._prepare_decode_graph(common(2, width=width), 2, for_capture=True)
    md = SimpleNamespace(
        graph_decode=captured, prefill=None, num_actual_tokens=2 * width, num_decodes=2, num_prefills=0
    )
    b._prepare_gbsa_metadata(md)
    fields = ("seq_lens", "block_table", "positions", "query_start_loc", "cu_seqlens_q", "gbsa_metadata")
    addresses = {field: getattr(captured, field).data_ptr() for field in fields}
    for real in (1, 0, 2):
        current = common(real, width=width)
        refreshed = b._prepare_decode_graph(current, real, for_capture=False)
        assert refreshed is captured
        b._prepare_gbsa_metadata(md)
        assert {field: getattr(captured, field).data_ptr() for field in fields} == addresses
        assert captured.seq_lens.tolist() == [19] * real + [0] * (2 - real)
        assert captured.positions.tolist() == list(range(19 - width, 19)) * real + [0] * ((2 - real) * width)
        assert torch.equal(captured.block_table[:real], current.block_table_tensor[:real])
        assert not captured.block_table[real:].any()
        assert captured.query_start_loc.tolist() == [0, width, 2 * width]
        assert captured.cu_seqlens_q.dtype == torch.int64
        assert torch.equal(captured.gbsa_metadata, torch.cat((captured.positions, captured.seq_lens)))


def test_graph_buckets_do_not_share_mutable_inputs():
    b = builder()
    first = b._prepare_decode_graph(common(2), 2, for_capture=True)
    second = b._prepare_decode_graph(common(2, width=1), 2, for_capture=True)
    before = first.positions.clone()
    b._prepare_decode_graph(common(0, width=1), 0, for_capture=False)
    assert first.seq_lens.data_ptr() != second.seq_lens.data_ptr()
    assert torch.equal(first.positions, before)


@pytest.mark.parametrize("invalid", ["query", "padding", "extent", "slots"])
def test_invalid_descriptor_is_rejected(invalid):
    b = builder()
    b._prepare_decode_graph(common(2), 2, for_capture=True)
    current = common(1)
    if invalid == "query":
        current.query_start_loc_cpu[1] = 3
    elif invalid == "padding":
        current._seq_lens_cpu[1] = 19
    elif invalid == "extent":
        current.num_actual_tokens = 3
    else:
        current.slot_mapping = current.slot_mapping[:1]
    with pytest.raises(ValueError):
        b._prepare_decode_graph(current, 1, for_capture=False)


def test_replay_cannot_create_an_uncaptured_bucket():
    with pytest.raises(RuntimeError, match="not captured"):
        builder()._prepare_decode_graph(common(1), 1, for_capture=False)
