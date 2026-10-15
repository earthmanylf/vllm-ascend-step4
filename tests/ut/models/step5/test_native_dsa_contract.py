# SPDX-License-Identifier: Apache-2.0
"""CPU checks of real native-op adapters and request metadata boundaries.

Custom kernels are replaced only at their API boundary. These tests do not
claim device arithmetic, native top-k tie behavior, or AICPU scheduling parity.
"""

import ast
import types
from pathlib import Path

import pytest
import torch

MODEL_DIR = Path(__file__).resolve().parents[4] / "vllm_ascend/models/step5"


def load_functions(filename, names, namespace):
    tree = ast.parse((MODEL_DIR / filename).read_text())
    body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            body.append(node)
        elif isinstance(node, ast.ClassDef):
            body.extend(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name in names)
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])),
            filename,
            "exec",
        ),
        namespace,
    )
    return types.SimpleNamespace(**{name: namespace[name] for name in names})


@pytest.mark.parametrize("region_len", [0, 2])
def test_indexer_uses_current_tuple_api_and_preserves_fp8_summary_storage(region_len):
    calls = []

    def native(query, weights, **kwargs):
        calls.append((query, weights, kwargs))
        ids = kwargs["selected_region_ids"]
        ids.fill_(-1)
        if kwargs["max_region_len"]:
            ids[:, :, 0] = 0
        return torch.zeros(1, 2, kwargs["max_region_len"]), ids

    adapter = load_functions(
        "indexer_region_score.py",
        {"indexer_score_topk"},
        {"torch": torch, "indexer_region_score": native},
    )
    query = torch.ones(2, 1, 4, 256, dtype=torch.bfloat16)
    weights = torch.full((2, 1, 4), -0.5)
    storage = torch.zeros(3, 16, 1, 256, dtype=torch.uint8)
    mean = storage.view(torch.float8_e4m3fn)
    qsl, seq = (
        torch.tensor([0, 2], dtype=torch.int32),
        torch.tensor([18], dtype=torch.int32),
    )
    scores, ids = adapter.indexer_score_topk(
        query,
        weights,
        mean,
        torch.tensor([[2]], dtype=torch.int32),
        qsl,
        seq,
        max_region_len=region_len,
        topk=512,
    )
    assert scores.shape == (2, region_len) and ids.shape == (2, 512)
    assert calls[0][0].dtype == torch.float8_e4m3fn
    kwargs = calls[0][2]
    assert kwargs["paged_key_summary"].data_ptr() == storage.data_ptr()
    assert kwargs["cu_seq_lens"] is qsl and kwargs["actual_kv_lengths"] is seq
    assert (
        kwargs["mask_mode"],
        kwargs["head_scale"],
        kwargs["cache_mode"],
        kwargs["sort_mode"],
    ) == (3, 1.0, 1, 1)
    assert ids.data_ptr() == kwargs["selected_region_ids"].data_ptr()
    with pytest.raises(ValueError, match="requires G"):
        adapter.indexer_score_topk(
            torch.ones(2, 2, 2, 256),
            weights,
            mean,
            None,
            qsl,
            seq,
            max_region_len=region_len,
            topk=512,
        )


def test_native_metadata_counts_cover_first_region_and_topk_boundary():
    calls = []

    def native(*args, **kwargs):
        calls.append((args, kwargs))
        return torch.tensor([17], dtype=torch.int32)

    ns = {
        "torch": torch,
        "generic_block_sparse_attention_metadata": native,
        "_SPARSE_PATTERN_PER_GROUP_PER_TOKEN": 4,
        "_MASK_CAUSAL": 1,
        "_QUANT_NONE": 0,
        "_SOFTMAX_PRECISION_FP32": 1,
    }
    adapter = load_functions(
        "generic_block_sparse_attention.py",
        {"gbsa_expected_counts", "gbsa_sparse_attention_metadata"},
        ns,
    )
    positions = torch.tensor([0, 7, 8, 4095, 4096, 4104])
    counts = adapter.gbsa_expected_counts(positions, groups=1, region_size=8, topk=512)
    assert counts.tolist() == [[1, 1, 2, 512, 513, 513]]
    qsl = torch.tensor([0, 6], dtype=torch.int64)
    seq = torch.tensor([4105], dtype=torch.int32)
    adapter.gbsa_sparse_attention_metadata(
        counts,
        num_heads=8,
        num_kv_groups=1,
        head_dim=128,
        region_size=8,
        topk=512,
        cu_seqlens_q=qsl,
        seqused_kv=seq,
        max_query_len=6,
        max_pages=33,
        page_size=128,
    )
    args, kwargs = calls[0]
    assert args[0].shape == (1, 6, 513) and args[1] is counts
    assert args[2:] == (8, 1, 128, [1, 8])
    assert kwargs["cu_seqlens_q"].data_ptr() == qsl.data_ptr()
    assert kwargs["seqused_kv"].data_ptr() == seq.data_ptr()
    assert (
        kwargs["layout_q"],
        kwargs["layout_kv"],
        kwargs["mask_mode"],
        kwargs["quant_mode"],
    ) == ("TND", "PA_BBND", 1, 0)
    assert kwargs["is_consistent_topk"] is False


def test_mixed_metadata_preserves_request_boundaries_and_journal_views():
    fn = load_functions("dsa_attention.py", {"_prepare_gbsa_metadata"}, {"torch": torch})._prepare_gbsa_metadata
    records = []
    builder = types.SimpleNamespace(
        query_start_loc_int64_buffer=torch.empty(9, dtype=torch.int64),
        capture_sizes=set(),
        _build_gbsa_metadata=lambda *args, **kwargs: records.append((args, kwargs)) or torch.tensor([7]),
    )
    states = torch.tensor([5, 2, 4, -1])
    d = types.SimpleNamespace(
        seq_lens=torch.tensor([18, 35], dtype=torch.int32),
        query_len=2,
        block_table=torch.tensor([[3, 8], [6, 1]], dtype=torch.int32),
        step5_request_state=states[:2],
    )
    p = types.SimpleNamespace(
        seq_lens=torch.tensor([12], dtype=torch.int32),
        context_lens=torch.tensor([9]),
        query_start_loc=torch.tensor([0, 3], dtype=torch.int32),
        num_query_tokens=3,
        max_query_len=3,
        block_table=torch.tensor([[5]], dtype=torch.int32),
        step5_request_state=states[2:3],
    )
    md = types.SimpleNamespace(
        query_start_loc=torch.tensor([0, 2, 4, 7, 7], dtype=torch.int32),
        num_actual_tokens=7,
        decode=d,
        prefill=p,
        num_decodes=2,
        num_prefills=1,
    )
    fn(builder, md)
    assert records[0][0][0].tolist() == [16, 17, 33, 34]
    assert records[0][0][1].tolist() == [0, 2, 4]
    assert records[0][0][2] is d.seq_lens and records[0][0][4] == 2
    assert records[1][0][0].tolist() == [9, 10, 11]
    assert records[1][0][1].tolist() == [0, 3]
    assert d.cu_seqlens_q.dtype == p.cu_seqlens_q.dtype == torch.int64
    assert d.cu_seqlens_q.data_ptr() == builder.query_start_loc_int64_buffer.data_ptr()
    assert d.step5_request_state.data_ptr() == states.data_ptr()
    assert d.step5_request_state.tolist() == [
        5,
        2,
    ] and p.step5_request_state.tolist() == [4]
    assert md.query_start_loc.tolist() == [0, 2, 4, 7, 7]


@pytest.mark.parametrize("path", ["decode", "verify", "prefill"])
def test_consumer_uses_paged_kv_and_request_varlen_without_flattening(path):
    records = []

    def indexer(*args, **kwargs):
        records.append(("indexer", args, kwargs))
        tokens, groups = args[0].shape[:2]
        return torch.zeros(groups * tokens, kwargs["max_region_len"]), torch.full(
            (groups * tokens, 512), -1, dtype=torch.int32
        )

    def attention(*args, **kwargs):
        records.append(("attention", args, kwargs))
        return torch.full_like(args[0], 2)

    name = "_forward_sparse_gqa_" + path
    fn = getattr(
        load_functions(
            "dsa_attention.py",
            {name},
            {
                "torch": torch,
                "indexer_score_topk": indexer,
                "gbsa_sparse_attention": attention,
            },
        ),
        name,
    )
    impl = types.SimpleNamespace(
        num_speculative_tokens=1,
        num_kv_groups=1,
        _max_regions=16,
        topk_regions=512,
        region_size=8,
        scale=0.125,
    )
    qlen = 2 if path == "verify" else 1
    d = types.SimpleNamespace(
        query_len=qlen,
        seq_lens=torch.tensor([18], dtype=torch.int32),
        block_table=torch.tensor([[2]], dtype=torch.int32),
        cu_seqlens_q=torch.tensor([0, qlen]),
        gbsa_metadata=torch.tensor([23]),
    )
    p = types.SimpleNamespace(
        num_query_tokens=3,
        seq_lens_host=[12],
        context_lens_host=[9],
        block_table=d.block_table,
        query_start_loc=torch.tensor([0, 3], dtype=torch.int32),
        cu_seqlens_q=torch.tensor([0, 3]),
        seq_lens=torch.tensor([12], dtype=torch.int32),
        gbsa_metadata=torch.tensor([29]),
    )
    md = types.SimpleNamespace(
        num_decodes=1,
        num_prefills=1,
        decode=d,
        prefill=p,
        query_start_loc=torch.tensor([0, qlen], dtype=torch.int32),
    )
    total = 3 if path == "prefill" else qlen
    cache = types.SimpleNamespace(mean_cache=torch.zeros(3, 16, 1, 256, dtype=torch.uint8))
    key = torch.empty(3, 128, 1, 128)
    value = torch.empty_like(key)
    output = torch.zeros(5, 8, 128)
    kwargs = dict(
        summary_cache=cache,
        md=md,
        query=torch.empty(total, 8, 128),
        key_cache=key,
        value_cache=value,
        index_q=torch.ones(total, 1, 4, 256),
        weights=torch.ones(total, 1, 4),
        output=output,
    )
    if path == "verify":
        kwargs["positions"] = torch.tensor([16, 17])
    elif path == "prefill":
        kwargs.update(positions=torch.tensor([7, 8, 9, 10, 11]), token_offset=2)
    fn(impl, **kwargs)
    _, inputs, _ = records[0]
    _, attn_inputs, _ = records[1]
    assert attn_inputs[1] is key and attn_inputs[2] is value
    assert inputs[3].shape[0] == 1 and inputs[4].tolist() == [0, total]
    assert inputs[5].tolist() == ([12] if path == "prefill" else [18])
    assert attn_inputs[6].tolist() == [0, total]
    assert attn_inputs[6].dtype == torch.int64
    assert attn_inputs[4].tolist() == ({"decode": [17], "verify": [16, 17], "prefill": [9, 10, 11]}[path])
    if path == "prefill":
        assert not output[:2].any() and (output[2:] == 2).all()
    else:
        assert (output[:total] == 2).all() and not output[total:].any()


def test_native_attention_passes_matching_metadata_and_paged_kv():
    calls = []
    idx = torch.tensor([[[0, 2]]], dtype=torch.int32)
    counts = torch.tensor([[2]], dtype=torch.int32)

    def native(*args, **kwargs):
        calls.append((args, kwargs))
        return torch.full_like(args[0], 3), None

    ns = {
        "torch": torch,
        "generic_block_sparse_attention": native,
        "_append_current_region": lambda *args, **kwargs: (idx, counts),
        "_SPARSE_PATTERN_PER_GROUP_PER_TOKEN": 4,
        "_MASK_CAUSAL": 1,
        "_QUANT_NONE": 0,
        "_SOFTMAX_PRECISION_FP32": 1,
    }
    fn = load_functions("generic_block_sparse_attention.py", {"gbsa_sparse_attention"}, ns).gbsa_sparse_attention
    q = torch.empty(1, 8, 128, dtype=torch.bfloat16)
    key = torch.empty(3, 128, 1, 128, dtype=torch.bfloat16)
    value = torch.empty_like(key)
    qsl = torch.tensor([0, 1], dtype=torch.int64)
    seq = torch.tensor([18], dtype=torch.int32)
    schedule = torch.tensor([7], dtype=torch.int32)
    table = torch.tensor([[2]], dtype=torch.int32)
    out = fn(
        q,
        key,
        value,
        torch.tensor([[0]], dtype=torch.int32),
        torch.tensor([17]),
        table,
        qsl,
        seq,
        schedule,
        num_kv_groups=1,
        region_size=8,
        softmax_scale=0.125,
    )
    args, kwargs = calls[0]
    assert args[1] is key and args[2] is value and args[3] is idx and args[4] is counts
    assert args[5] == [1, 8] and kwargs["metadata"] is schedule and kwargs["block_table"] is table
    assert kwargs["cu_seqlens_q"] is qsl and kwargs["seqused_kv"] is seq
    assert kwargs["layout_sparse_pattern"] == 4 and kwargs["mask_mode"] == 1
    assert kwargs["quant_mode"] == 0 and kwargs["softmax_precision"] == 1
    assert kwargs["return_softmax_lse"] is False and kwargs["is_consistent_topk"] is False
    assert out.dtype == q.dtype and (out == 3).all()
