# SPDX-License-Identifier: Apache-2.0
"""Snapshot handoffs must survive slot release and physical-page remapping."""

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector import (
    MTPPrefixSnapshotAdapter,
    SummaryCacheSnapshotAdapter,
)


def make_cache():
    device = "npu:0"
    return SimpleNamespace(
        num_pages=4,
        page_size=128,
        region_block_size=8,
        summaries_per_page=16,
        proxy_dim=4,
        mean_cache=torch.zeros((4, 16, 4), device=device, dtype=torch.uint8),
        _step5_csa_active_region_ids=torch.full((8,), -1, device=device, dtype=torch.int32),
        _step5_csa_active_slot_by_region=torch.full((64,), -1, device=device, dtype=torch.int32),
        _step5_csa_numerator_cache=torch.zeros((8, 1, 4), device=device),
        _step5_csa_denominator_cache=torch.zeros((8, 1, 4), device=device),
        _step5_csa_max_cache=torch.zeros((8, 1), device=device),
        _step5_csa_active_token_k=torch.zeros((8, 8, 1, 4), device=device, dtype=torch.bfloat16),
        _step5_csa_active_token_z=torch.zeros((8, 8, 1, 4), device=device, dtype=torch.bfloat16),
        _step5_csa_active_token_valid=torch.zeros((8, 8), device=device, dtype=torch.uint8),
    )


def test_snapshot_survives_release_and_restores_receiver_owned_slot():
    torch.npu.set_device(0)
    source, dest = make_cache(), make_cache()
    source._step5_csa_active_region_ids[2] = 16
    source._step5_csa_active_slot_by_region[16] = 2
    source._step5_csa_numerator_cache[2].fill_(0.75)
    source._step5_csa_denominator_cache[2].fill_(2.5)
    source._step5_csa_max_cache[2].fill_(-3)
    source._step5_csa_active_token_k[2].fill_(4)
    source._step5_csa_active_token_z[2].fill_(5)
    source._step5_csa_active_token_valid[2].fill_(1)
    exporter, importer = SummaryCacheSnapshotAdapter(source), SummaryCacheSnapshotAdapter(dest)
    exporter.export("request", [1], 7)
    payload = exporter.get("request")
    source._step5_csa_numerator_cache.zero_()
    source._step5_csa_active_region_ids.fill_(-1)
    importer.restore(payload, [3], 7)
    assert dest._step5_csa_active_region_ids[0].item() == 48
    assert dest._step5_csa_active_slot_by_region[48].item() == 0
    assert torch.all(dest._step5_csa_numerator_cache[0] == 0.75).item()
    assert torch.all(dest._step5_csa_denominator_cache[0] == 2.5).item()
    assert dest._step5_csa_max_cache[0].item() == -3
    assert torch.all(dest._step5_csa_active_token_k[0] == 4).item()
    assert torch.all(dest._step5_csa_active_token_z[0] == 5).item()
    assert dest._step5_csa_active_token_valid[0].tolist() == [1] * 7 + [0]
    with pytest.raises(RuntimeError, match="overwrite"):
        importer.restore(payload, [3], 7)
    with pytest.raises(RuntimeError, match="boundary mismatch"):
        importer.restore(payload, [2], 8)
    exporter.release("request")
    with pytest.raises(KeyError):
        exporter.get("request")


def test_empty_active_tail_is_a_valid_snapshot():
    torch.npu.set_device(0)
    exporter = SummaryCacheSnapshotAdapter(make_cache())
    exporter.export("aligned", [1], 8)
    importer = SummaryCacheSnapshotAdapter(make_cache())
    importer.restore(exporter.get("aligned"), [3], 8)
    assert torch.all(importer.cache._step5_csa_active_region_ids == -1).item()


def test_mtp_snapshot_keeps_exact_hidden_tail_across_prefill_chunks():
    pool = MTPPrefixSnapshotAdapter(4, 2)
    first = torch.arange(24, dtype=torch.float32).view(6, 4)
    pool.capture("p", 7, torch.arange(6), torch.arange(6), first)
    last = torch.arange(24, 28, dtype=torch.float32).view(1, 4)
    pool.capture("p", 7, torch.tensor([6]), torch.tensor([6]), last)
    pool.export("p", [], 7)
    first.zero_()
    last.zero_()
    pool.restore_for_request("d", pool.get("p"), 7)
    state = pool.take_received(["d"])["d"]
    assert state["positions"] == [5, 6]
    assert state["tokens"] == [5, 6]
    rows = torch.frombuffer(bytearray(state["hidden"]), dtype=torch.float32).view(2, 4)
    assert torch.equal(rows, torch.arange(20, 28, dtype=torch.float32).view(2, 4))
    assert pool.take_received(["d"]) == {}
    with pytest.raises(RuntimeError, match="boundary"):
        pool.export("p", [], 8)
    pool.release("p")
    with pytest.raises(KeyError):
        pool.get("p")


def test_mtp_snapshot_replays_only_draft_inputs_and_preserves_mixed_batch_slots():
    from vllm_ascend.spec_decode.step3p5 import AscendStep3p5MTPProposer

    class Table:
        def __init__(self):
            self.slot_mapping = SimpleNamespace(gpu=torch.empty(0, dtype=torch.long))

        def get_device_tensor(self):
            return torch.tensor([[3], [17]])

        def compute_slot_mapping(self, num_reqs, starts, positions):
            owners = torch.repeat_interleave(torch.tensor([3, 17]), starts[1:] - starts[:-1])
            self.slot_mapping.gpu = owners * 128 + positions

    proposer = AscendStep3p5MTPProposer.__new__(AscendStep3p5MTPProposer)
    table = Table()
    proposer.runner = SimpleNamespace(input_batch=SimpleNamespace(req_ids=["d", "other"], block_table=[table]))
    proposer.draft_attn_groups = [SimpleNamespace(kv_cache_group_id=0)]
    proposer._per_group_slot_mappings = {}
    proposer._per_group_block_tables = {}
    prefix_rows = torch.arange(8, dtype=torch.float32).view(2, 4)
    prefix = {
        "boundary": 7,
        "dtype": "torch.float32",
        "width": 4,
        "positions": [5, 6],
        "tokens": [264, 455],
        "hidden": prefix_rows.view(torch.uint8).numpy().tobytes(),
    }
    cad = SimpleNamespace(
        query_start_loc_cpu=torch.tensor([0, 1, 5]),
        query_start_loc=torch.tensor([0, 1, 5]),
        seq_lens_cpu=None,
        seq_lens=torch.tensor([8, 14]),
        num_reqs=2,
        num_actual_tokens=5,
    )
    ids = torch.tensor([1234, 20, 21, 22, 23])
    hidden = torch.arange(20, dtype=torch.float32).view(5, 4)
    result = proposer._prepend_mtp_prefix_replay(
        {"d": prefix},
        ids,
        torch.tensor([7, 10, 11, 12, 13]),
        hidden,
        torch.tensor([0, 4]),
        cad,
    )
    assert result[0].tolist() == [264, 455, 1234, 20, 21, 22, 23]
    assert result[3].tolist() == [2, 6]
    assert result[4].query_start_loc_cpu.tolist() == [0, 3, 7]
    assert table.slot_mapping.gpu.tolist() == [389, 390, 391, 2186, 2187, 2188, 2189]
    assert hidden.shape == (5, 4)
    assert cad.query_start_loc_cpu.tolist() == [0, 1, 5]
