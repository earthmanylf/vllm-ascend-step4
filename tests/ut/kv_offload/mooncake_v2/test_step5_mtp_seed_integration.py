# SPDX-License-Identifier: Apache-2.0
"""Source-isolated MTP3 producer registration/publication/response checks."""

from dataclasses import dataclass
from types import SimpleNamespace as NS

import pytest
import torch

from .test_step5_mtp_seed_state import candidate
from .test_step5_transfer_contract import load_definitions


def setup_scheduler():
    module = candidate()
    ns = dict(
        dataclass=dataclass,
        KVConnectorMetadata=object,
        MooncakeBaseConnectorScheduler=object,
        RequestStatus=NS(FINISHED_LENGTH_CAPPED="length"),
        is_step5_mtp3_pd=module.is_step5_mtp3_pd,
        redact_mtp_seed_params=module.redact_mtp_seed_params,
        logger=NS(debug=lambda *args: None, info=lambda *args: None),
        time=NS(time=lambda: 100),
    )
    load_definitions("metadata.py", {"ReqMeta", "MooncakeConnectorMetadata"}, ns)
    load_definitions("pull_scheduler.py", {"MooncakePullConnectorScheduler"}, ns)
    cls = ns["MooncakePullConnectorScheduler"]
    obj = cls.__new__(cls)
    obj.vllm_config = NS(
        use_v2_model_runner=True,
        model_config=NS(hf_text_config=NS(model_type="step5", hidden_size=4)),
        speculative_config=NS(method="mtp", num_speculative_tokens=3),
        kv_transfer_config=NS(is_kv_transfer_instance=True, is_kv_producer=True, is_kv_consumer=False),
    )
    obj.kv_transfer_config = obj.vllm_config.kv_transfer_config
    obj.engine_id = "engine-P"
    obj.side_channel_host, obj.side_channel_port = "127.0.0.1", 19214
    obj._reqs_need_recv, obj._reqs_need_send = {}, {}
    obj._reqs_in_batch = set()
    obj._mtp_producer_prompt_lens, obj._mtp_restore_requests = {}, {}
    obj.tp_size = 2
    registry = module.MTPSeedRegistry("engine-P", tp_size=2, hidden_size=4)
    delayed = []
    obj._sending_thread = NS(
        mtp_seed_registry=registry,
        register_mtp_request=registry.register,
        take_mtp_seeds=registry.take,
        add_delayed_request=lambda *args: delayed.append(args),
    )
    obj._get_transfer_block_ids = lambda ids, count: ids
    req = NS(
        request_id="P1",
        prompt_token_ids=list(range(128)),
        output_token_ids=[42],
        status="length",
        sampling_params=NS(max_tokens=1),
        kv_transfer_params={"do_remote_decode": True, "do_remote_prefill": False},
    )
    return obj, req, registry, module, delayed


@pytest.mark.parametrize("has_update_stream", [False, True])
@pytest.mark.parametrize("pp_rank", range(4))
def test_worker_captures_after_drafter_and_before_return(monkeypatch, has_update_stream, pp_rank):
    import ast
    import sys
    from pathlib import Path

    source = Path(__file__).resolve().parents[4] / "vllm_ascend/worker/worker.py"
    cls = next(n for n in ast.parse(source.read_text()).body if getattr(n, "name", "") == "NPUWorker")
    fn = next(n for n in cls.body if getattr(n, "name", "") == "sample_tokens")
    fn.decorator_list = []
    calls = []

    class Event:
        def record(self, stream):
            calls.append("fence")

    fake = NS(
        capture_mtp_seeds=lambda *args, **kwargs: calls.append("capture") or {"P1": "seed"},
        publish_mtp_seed=lambda *args, **kwargs: calls.append("publish"),
    )
    monkeypatch.setitem(sys.modules, "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.mtp_state", fake)

    def context():
        calls.append("context")
        return {"P1": 128}, ("127.0.0.1", 19214, "engine-P"), 1

    output = NS()

    def sample(grammar):
        calls.append("drafter")
        return output

    spec = NS(input_batch=NS(), update_stream=NS() if has_update_stream else None)

    def wait(stream):
        assert stream is not None, "Eager mode has no update stream"

    runner = NS(
        sample_tokens=sample,
        speculator=spec if pp_rank == 3 else None,
        kv_connector=NS(kv_connector=NS(get_mtp_capture_context=context)),
        main_stream=NS(wait_stream=wait),
        ascend_config=NS(scheduler_config=NS(profiling_chunk_config=NS())),
    )
    worker = NS(
        model_runner=runner,
        use_v2_model_runner=True,
        vllm_config=NS(
            model_config=NS(hf_text_config=NS(model_type="step5")),
            kv_transfer_config=NS(is_kv_producer=True),
            speculative_config=NS(method="mtp", num_speculative_tokens=3),
        ),
    )
    ns = dict(
        torch=NS(npu=NS(Event=Event)),
        get_pp_group=lambda: NS(is_last_rank=pp_rank == 3),
        _attach_profiling_chunk_execution_time=lambda *args: calls.append("profile"),
    )
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, fn], type_ignores=[])), str(source), "exec"), ns)
    assert ns["sample_tokens"](worker, None) is output
    expected = (
        ["context", "drafter", "fence", "capture", "publish", "profile"] if pp_rank == 3 else ["drafter", "profile"]
    )
    assert calls == expected


def test_d_worker_receives_seed_even_without_remote_token_read():
    module = candidate()
    ns = dict(
        MooncakeBaseConnectorWorker=object,
        MTPSeedState=module.MTPSeedState,
        decode_mtp_seed_params=module.decode_mtp_seed_params,
    )
    load_definitions("pull_worker.py", {"MooncakePullConnectorWorker"}, ns)
    worker = ns["MooncakePullConnectorWorker"].__new__(ns["MooncakePullConnectorWorker"])
    worker.tp_size, worker.tp_rank = 2, 1
    worker.vllm_config = NS(model_config=NS(hf_text_config=NS(hidden_size=4)))
    worker.mtp_seed_states, worker._seen_mtp_seeds = {}, {}
    states = {}
    for rank in range(2):
        state = module.MTPSeedState.snapshot(
            torch.ones(4, dtype=torch.bfloat16), producer_request_id="P1", position=127, token_id=42, tp_rank=rank
        )
        states[str(rank)] = state.to_wire()
    params = dict(mtp3_seeds=states, remote_request_id="P1", remote_num_prompt_tokens=128, last_token_id=42)
    metadata = NS(requests={}, mtp_restore_requests={"D1": params})
    worker.start_load_kv(metadata)
    assert worker.mtp_seed_states["D1"].tp_rank == 1
    del worker.mtp_seed_states["D1"]  # once applied, duplicate metadata must not reinitialize
    worker.start_load_kv(metadata)
    assert not worker.mtp_seed_states
    worker.clear_mtp_seed_requests({"D1"})
    assert not worker._seen_mtp_seeds


def test_d_speculator_restores_before_upstream_prepare(monkeypatch):
    import ast
    import sys
    from pathlib import Path

    import numpy as np

    module = candidate()
    state = module.MTPSeedState.snapshot(
        torch.ones(4, dtype=torch.bfloat16), producer_request_id="P1", position=127, token_id=42, tp_rank=1
    )
    store = {"D1": state}
    monkeypatch.setitem(
        sys.modules,
        "vllm.distributed.kv_transfer",
        NS(get_kv_transfer_group=lambda: NS(get_mtp_seed_store=lambda: store)),
    )
    monkeypatch.setitem(sys.modules, "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.mtp_state", module)
    source = Path(__file__).resolve().parents[4] / "vllm_ascend/worker/v2/spec_decode/multi_module_mtp/speculator.py"
    cls = next(
        n for n in ast.parse(source.read_text()).body if getattr(n, "name", "") == "AscendMultiModuleMTPSpeculator"
    )
    fn = next(n for n in cls.body if getattr(n, "name", "") == "_prepare_inputs")

    class Base:
        def _prepare_inputs(self, *args, **kwargs):
            assert self.cached_draft_input_ids[0].tolist() == [42, 0]
            assert not store

    cls.body = [fn]
    cls.bases = [ast.Name(id="Base", ctx=ast.Load())]
    ns = {"Base": Base}
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, cls], type_ignores=[])), str(source), "exec"), ns)
    obj = ns[cls.name].__new__(ns[cls.name])
    obj._step5_mtp_seed_consumer = True
    obj._exact_draft_seq_lens = True
    obj._mtp_seed_restore_active = True
    obj.cached_draft_input_ids = torch.full((1, 2), 99, dtype=torch.int64)
    obj.cached_target_hidden_states = torch.full((1, 2, 4), 99, dtype=torch.bfloat16)
    obj.input_buffers = NS(seq_lens=torch.tensor([129]), seq_lens_cpu=torch.tensor([0]))
    batch = NS(
        req_ids=["D1"],
        num_reqs=1,
        idx_mapping_np=np.array([0]),
        query_start_loc_np=np.array([0, 1]),
        input_ids=torch.tensor([42]),
        positions=torch.tensor([128]),
    )
    obj._prepare_inputs(None, batch, 1, 1, torch.tensor([1]), torch.tensor([0]), None, None)
    assert obj.input_buffers.seq_lens_cpu.item() == 129


def test_profiling_prepare_does_not_access_uninitialized_kv_group(monkeypatch):
    import ast
    import sys
    from pathlib import Path

    source = Path(__file__).resolve().parents[4] / "vllm_ascend/worker/v2/spec_decode/multi_module_mtp/speculator.py"
    cls = next(
        n for n in ast.parse(source.read_text()).body if getattr(n, "name", "") == "AscendMultiModuleMTPSpeculator"
    )
    fn = next(n for n in cls.body if getattr(n, "name", "") == "_prepare_inputs")

    class Base:
        def _prepare_inputs(self, *args, **kwargs):
            pass

    def forbidden():
        raise AssertionError("Profiling must not access the KV group")

    monkeypatch.setitem(sys.modules, "vllm.distributed.kv_transfer", NS(get_kv_transfer_group=forbidden))
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.mtp_state",
        NS(restore_mtp_seed_batch=lambda *args: None),
    )
    cls.body = [fn]
    cls.bases = [ast.Name(id="Base", ctx=ast.Load())]
    ns = {"Base": Base}
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, cls], type_ignores=[])), str(source), "exec"), ns)
    obj = ns[cls.name].__new__(ns[cls.name])
    obj._step5_mtp_seed_consumer = True
    obj._exact_draft_seq_lens = True
    obj._mtp_seed_restore_active = False
    obj.input_buffers = NS(seq_lens=torch.tensor([1]), seq_lens_cpu=torch.tensor([0]))
    obj._prepare_inputs(None)
    assert obj.input_buffers.seq_lens_cpu.item() == 1


def test_producer_metadata_persists_until_final_prompt_step():
    obj, req, _, _, _ = setup_scheduler()
    obj.update_state_after_alloc(req, NS(), 0)
    first = obj.build_connector_meta(NS())
    second = obj.build_connector_meta(NS())
    assert first.mtp_producer_prompt_lens == second.mtp_producer_prompt_lens == {"P1": 128}
    assert second.mtp_seed_endpoint == ("127.0.0.1", 19214, "engine-P")


def test_response_requires_all_seeds_before_delayed_free_is_exposed():
    obj, req, _, _, delayed = setup_scheduler()
    obj.update_state_after_alloc(req, NS(), 0)
    with pytest.raises(ValueError, match="every TP"):
        obj.request_finished(req, ([1], [2]))
    assert not delayed and not obj._reqs_need_send


def test_producer_response_contains_immutable_all_rank_seed():
    obj, req, registry, module, delayed = setup_scheduler()
    obj.update_state_after_alloc(req, NS(), 0)
    for rank in range(2):
        state = module.MTPSeedState.snapshot(
            torch.ones(4, dtype=torch.bfloat16), producer_request_id="P1", position=127, token_id=42, tp_rank=rank
        )
        assert registry.publish("engine-P", state.to_wire())
    keep, params = obj.request_finished(req, ([1], [2]))
    assert keep and delayed
    assert params["remote_num_prompt_tokens"] == 128
    assert set(params["mtp3_seeds"]) == {"0", "1"}
    assert "P1" not in obj._mtp_producer_prompt_lens
