# SPDX-License-Identifier: Apache-2.0
"""CPU checks for immutable MTP3 handoff seeds and request-slot initialization."""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake/mtp_state.py"


def candidate():
    assert SOURCE.is_file(), "MTP3 seed-state implementation is missing"
    spec = importlib.util.spec_from_file_location("step5_seed_state_cpu", SOURCE)
    value = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = value
    spec.loader.exec_module(value)
    return value


def seed(module):
    tensor = torch.tensor([1, 2, 3, 4], dtype=torch.bfloat16)
    state = module.MTPSeedState.snapshot(tensor, producer_request_id="P1", position=127, token_id=42, tp_rank=1)
    return tensor, state


def decode(module, wire):
    return module.MTPSeedState.from_wire(
        wire,
        expected_request_id="P1",
        expected_position=127,
        expected_token_id=42,
        expected_tp_rank=1,
        expected_hidden_size=4,
    )


def test_seed_snapshot_owns_bytes_and_roundtrips():
    module = candidate()
    tensor, state = seed(module)
    before = state.hidden_bytes
    tensor.zero_()
    assert state.hidden_bytes == before
    assert decode(module, state.to_wire()) == state
    assert "hidden_bytes=" not in repr(state)


def test_restore_initializes_only_current_request_slot():
    module = candidate()
    _, state = seed(module)
    tokens = torch.full((3, 2), 999, dtype=torch.int64)
    hidden = torch.full((3, 2, 4), 999, dtype=torch.bfloat16)
    old_tokens, old_hidden = tokens.clone(), hidden.clone()
    module.restore_mtp_seed(state, tokens, hidden, 1)
    assert tokens[1].tolist() == [42, 0]
    torch.testing.assert_close(hidden[1, 0], torch.tensor([1, 2, 3, 4], dtype=torch.bfloat16))
    assert not hidden[1, 1].count_nonzero()
    assert torch.equal(tokens[[0, 2]], old_tokens[[0, 2]])
    assert torch.equal(hidden[[0, 2]], old_hidden[[0, 2]])


@pytest.mark.parametrize(
    "field,value",
    [
        ("version", 2),
        ("position", 128),
        ("tp_rank", 0),
        ("producer_request_id", "P2"),
        ("hidden_size", 8),
        ("sha256", "bad"),
        ("hidden", "bad"),
        ("token_id", True),
    ],
)
def test_invalid_seed_context_or_payload_rejected(field, value):
    module = candidate()
    _, state = seed(module)
    wire = state.to_wire()
    wire[field] = value
    with pytest.raises(ValueError):
        decode(module, wire)


def test_capture_uses_real_request_slot_and_final_prompt_position():
    from types import SimpleNamespace as NS

    import numpy as np

    module = candidate()
    hidden = torch.zeros((3, 2, 4), dtype=torch.bfloat16)
    hidden[0, 1] = torch.tensor([1, 2, 3, 4], dtype=torch.bfloat16)
    speculator = NS(
        num_speculative_steps=3,
        cached_draft_input_ids=torch.tensor([[9, 42], [7, 8], [5, 6]]),
        cached_target_hidden_states=hidden,
        input_buffers=NS(positions=torch.tensor([99, 127, 31])),
        last_token_indices=torch.tensor([0, 1, 2]),
    )
    batch = NS(req_ids=["mid", "P1", "local"], num_reqs=3, idx_mapping_np=np.array([2, 0, 1]))
    states = module.capture_mtp_seeds(speculator, batch, {"mid": 256, "P1": 128}, tp_rank=1)
    assert set(states) == {"P1"}
    assert states["P1"].token_id == 42 and states["P1"].position == 127
    assert decode(module, states["P1"].to_wire()) == states["P1"]


def test_capture_does_not_publish_uncomputed_or_future_prompt_state():
    from types import SimpleNamespace as NS

    import numpy as np

    module = candidate()
    speculator = NS(
        num_speculative_steps=3,
        cached_draft_input_ids=torch.tensor([[9, 42]]),
        cached_target_hidden_states=torch.ones((1, 2, 4), dtype=torch.bfloat16),
        input_buffers=NS(positions=torch.tensor([128])),
        last_token_indices=torch.tensor([0]),
    )
    batch = NS(req_ids=["P1"], num_reqs=1, idx_mapping_np=np.array([0]))
    assert module.capture_mtp_seeds(speculator, batch, {"P1": 128}, tp_rank=1) == {}


def test_registry_requires_every_tp_rank_before_handoff():
    module = candidate()
    registry = module.MTPSeedRegistry("engine-P", tp_size=2, hidden_size=4)
    registry.register("P1", 128)
    _, state = seed(module)
    wire = state.to_wire()
    wire["tp_rank"] = 0
    assert registry.publish("engine-P", wire)
    with pytest.raises(ValueError, match="every TP"):
        registry.take("P1", 42)
    assert registry.publish("engine-P", state.to_wire())
    params = registry.take("P1", 42)
    assert set(params) == {"0", "1"}
    assert not registry.publish("engine-P", wire)  # already consumed; never republish stale state


def test_registry_duplicate_conflict_and_cancellation():
    module = candidate()
    registry = module.MTPSeedRegistry("engine-P", tp_size=2, hidden_size=4)
    registry.register("P1", 128)
    _, state = seed(module)
    assert registry.publish("engine-P", state.to_wire())
    assert registry.publish("engine-P", state.to_wire())
    changed = state.to_wire()
    changed["token_id"] = 43
    with pytest.raises(ValueError, match="Conflicting"):
        registry.publish("engine-P", changed)
    registry.discard("P1")
    assert not registry.publish("engine-P", state.to_wire())


def test_registry_never_accepts_other_engine_or_wrong_prompt_boundary():
    module = candidate()
    registry = module.MTPSeedRegistry("engine-P", tp_size=2, hidden_size=4)
    registry.register("P1", 128)
    _, state = seed(module)
    assert not registry.publish("engine-other", state.to_wire())
    bad = state.to_wire()
    bad["position"] = 126
    with pytest.raises(ValueError):
        registry.publish("engine-P", bad)


def test_real_control_ack_stores_seed_without_finishing_kv():
    import queue
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from types import SimpleNamespace as NS

    import msgspec
    import zmq

    from .test_step5_transfer_contract import load_definitions

    module = candidate()
    ns = dict(
        threading=threading,
        msgspec=msgspec,
        zmq=zmq,
        Any=object,
        GET_META_MSG=b"get_meta_msg",
        DONE_RECVING_MSG=b"done_recving_msg",
        ACK_MSG=b"ACK",
        MTP_SEED_MESSAGE=module.MTP_SEED_MESSAGE,
        logger=NS(warning=lambda *args: None, exception=lambda *args: None),
    )
    load_definitions("pull_scheduler.py", {"MooncakeSchedulerSendingThread"}, ns)
    sender = ns["MooncakeSchedulerSendingThread"].__new__(ns["MooncakeSchedulerSendingThread"])
    sender.mtp_seed_registry = module.MTPSeedRegistry("engine-P", tp_size=2, hidden_size=4)
    sender.stop_event = threading.Event()
    sender.finished_requests = queue.SimpleQueue()
    sender.delayed_free_requests = {"P1": 123}
    sender.register_mtp_request("P1", 128)
    endpoint = queue.Queue()
    ctx = zmq.Context()

    def serve():
        with ctx.socket(zmq.ROUTER) as sock:
            port = sock.bind_to_random_port("tcp://127.0.0.1")
            sock.setsockopt(zmq.RCVTIMEO, 50)
            endpoint.put(("127.0.0.1", port, "engine-P"))
            sender._run_busy_loop(sock)

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(serve)
        try:
            address = endpoint.get(timeout=2)
            _, state = seed(module)
            module.publish_mtp_seed(address, state, timeout_ms=1000)
            wire = state.to_wire()
            wire["tp_rank"] = 0
            rank0 = module.MTPSeedState.from_wire(
                wire,
                expected_request_id="P1",
                expected_position=127,
                expected_token_id=42,
                expected_tp_rank=0,
                expected_hidden_size=4,
            )
            module.publish_mtp_seed(address, rank0, timeout_ms=1000)
            assert set(sender.take_mtp_seeds("P1", 42)) == {"0", "1"}
            assert module.publish_mtp_seed(address, state, timeout_ms=1000) is False
            with pytest.raises(RuntimeError, match="rejected"):
                module.publish_mtp_seed((address[0], address[1], "engine-other"), state, timeout_ms=1000)
            assert sender.finished_requests.empty()
            assert sender.delayed_free_requests == {"P1": 123}
        finally:
            sender.stop_event.set()
            future.result(timeout=2)
            ctx.term()


def test_batch_restore_requires_exact_first_continuation_position():
    from types import SimpleNamespace as NS

    import numpy as np

    module = candidate()
    _, state = seed(module)
    spec = NS(
        cached_draft_input_ids=torch.full((2, 2), 99, dtype=torch.int64),
        cached_target_hidden_states=torch.full((2, 2, 4), 99, dtype=torch.bfloat16),
    )
    batch = NS(
        req_ids=["D1"],
        num_reqs=1,
        idx_mapping_np=np.array([1]),
        query_start_loc_np=np.array([0, 1]),
        positions=torch.tensor([128]),
        input_ids=torch.tensor([42]),
    )
    store = {"D1": state}
    assert module.restore_mtp_seed_batch(spec, batch, torch.tensor([0]), store) == {"D1"}
    assert not store
    assert spec.cached_draft_input_ids[1].tolist() == [42, 0]
    assert spec.cached_draft_input_ids[0].tolist() == [99, 99]


def test_batch_restore_seeds_embedding_window_for_reused_vl_slot():
    from types import SimpleNamespace as NS

    import numpy as np

    module = candidate()
    _, state = seed(module)
    embeds = torch.full((2, 2, 4), 99, dtype=torch.bfloat16)
    spec = NS(
        cached_draft_input_ids=torch.full((2, 2), 99, dtype=torch.int64),
        cached_target_hidden_states=torch.full((2, 2, 4), 99, dtype=torch.bfloat16),
        cached_draft_input_embeds=embeds,
        model=NS(embed_input_ids=lambda ids: ids[:, None].expand(-1, 4).to(torch.bfloat16)),
    )
    batch = NS(
        req_ids=["D1"],
        num_reqs=1,
        idx_mapping_np=np.array([1]),
        query_start_loc_np=np.array([0, 1]),
        positions=torch.tensor([128]),
        input_ids=torch.tensor([42]),
    )
    store = {"D1": state}
    module.restore_mtp_seed_batch(spec, batch, torch.tensor([0]), store)
    assert torch.equal(embeds[1, 0], torch.full((4,), 42, dtype=torch.bfloat16))
    assert torch.count_nonzero(embeds[1, 1]) == 0
    assert torch.equal(embeds[0], torch.full((2, 4), 99, dtype=torch.bfloat16))


def test_bad_first_step_never_mutates_cache_or_consumes_seed():
    from types import SimpleNamespace as NS

    import numpy as np

    module = candidate()
    _, state = seed(module)
    spec = NS(
        cached_draft_input_ids=torch.full((1, 2), 99, dtype=torch.int64),
        cached_target_hidden_states=torch.full((1, 2, 4), 99, dtype=torch.bfloat16),
    )
    batch = NS(
        req_ids=["D1"],
        num_reqs=1,
        idx_mapping_np=np.array([0]),
        query_start_loc_np=np.array([0, 1]),
        positions=torch.tensor([129]),
        input_ids=torch.tensor([42]),
    )
    store = {"D1": state}
    with pytest.raises(ValueError):
        module.restore_mtp_seed_batch(spec, batch, torch.tensor([0]), store)
    assert store and spec.cached_draft_input_ids[0].tolist() == [99, 99]


def test_seed_logs_redact_hidden_payload_without_mutation():
    module = candidate()
    _, state = seed(module)
    params = {"remote_host": "P", "mtp3_seeds": {1: state.to_wire()}}
    redacted = module.redact_mtp_seed_params(params)
    assert redacted["remote_host"] == "P"
    assert state.to_wire()["hidden"] not in repr(redacted)
    assert isinstance(params["mtp3_seeds"], dict)
