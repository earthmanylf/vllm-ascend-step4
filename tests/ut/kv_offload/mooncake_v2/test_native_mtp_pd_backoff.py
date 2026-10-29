# SPDX-License-Identifier: Apache-2.0
"""CPU checks compiling production scheduler methods without runtime imports."""

import ast
import math
import time
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

ROOT = Path(__file__).resolve().parents[4]
MOONCAKE = ROOT / "vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake"


def load_class(path, name, namespace):
    tree = ast.parse(path.read_text())
    node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[future, node], type_ignores=[])), str(path), "exec"),
        namespace,
    )
    return namespace[name]


@pytest.fixture
def scheduler():
    namespace = dict(
        MambaSpec=type("MambaSpec", (), {}),
        SlidingWindowSpec=type("SlidingWindowSpec", (), {}),
        UniformTypeKVCacheSpecs=type("UniformTypeKVCacheSpecs", (), {}),
        is_circular_kv_cache_spec=lambda spec: False,
        cdiv=lambda n, d: math.ceil(n / d),
        logger=NS(debug=lambda *a: None, info=lambda *a: None, warning=lambda *a: None),
        redact_mtp_seed_params=lambda params: params,
        is_step5_mtp3_pd=lambda cfg: True,
        RequestStatus=NS(FINISHED_LENGTH_CAPPED="finished"),
        MooncakeConnectorMetadata=lambda: NS(add_new_req=lambda **kw: None),
        time=time,
    )
    load_class(MOONCAKE / "base_scheduler.py", "MooncakeBaseConnectorScheduler", namespace)
    cls = load_class(MOONCAKE / "pull_scheduler.py", "MooncakePullConnectorScheduler", namespace)
    obj = cls.__new__(cls)
    obj.vllm_config = NS(
        num_prefill_lookahead_tokens=3,
        model_config=NS(hf_text_config=NS(model_type="step5", hidden_size=8), hf_config=NS()),
        cache_config=NS(enable_prefix_caching=False),
    )
    obj.need_truncate = False
    obj.block_size = 128
    obj.tp_size = obj.dcp_size = 1
    obj.num_speculative_tokens = 3
    obj.group_block_size = [128]
    obj.group_unique_specs = [[object()]]
    obj.kv_transfer_config = NS(is_kv_producer=True)
    obj._reqs_need_recv, obj._reqs_need_send, obj._reqs_recv_info = {}, {}, {}
    obj._reqs_in_batch = set()
    obj._sending_thread = NS(add_delayed_request=lambda *a: None)
    obj.engine_id, obj.side_channel_host, obj.side_channel_port = "producer", "localhost", 9000
    return obj


def request(length, *, embeds=False, skip=False):
    return NS(
        request_id="req",
        prompt_token_ids=None if embeds else list(range(length)),
        prompt_embeds=list(range(length)) if embeds else None,
        _all_token_ids=list(range(length)),
        num_prompt_tokens=length,
        max_tokens=8,
        kv_transfer_params={"do_remote_decode": True},
        get_skip_reading_prefix_cache=lambda: skip,
        status="finished",
        output_token_ids=[],
    )


@pytest.mark.parametrize("length,expected", [(129, 127), (513, 511), (1, 1), (2, 2)])
@pytest.mark.parametrize("embeds", [False, True])
def test_prefill_truncates_once_before_prefix_lookup(scheduler, length, expected, embeds):
    req = request(length, embeds=embeds)
    scheduler.on_new_request(req)
    scheduler.on_new_request(req)
    assert req.num_prompt_tokens == expected
    assert req._all_token_ids == list(range(expected))
    assert (req.prompt_embeds if embeds else req.prompt_token_ids) == list(range(expected))
    assert req.max_tokens == (1 if expected != length else 8)
    assert bool(req.kv_transfer_params.get("_p_side_truncated")) == (expected != length)


@pytest.mark.parametrize(
    "length,computed,expected", [(129, 0, 127), (513, 0, 511), (513, 128, 383), (1, 0, 1), (2, 0, 2)]
)
@pytest.mark.parametrize("remote_count", [False, True])
@pytest.mark.parametrize("embeds", [False, True])
def test_decode_original_prompt_credit_without_seeds(scheduler, length, computed, expected, remote_count, embeds):
    req = request(length, embeds=embeds)
    req.kv_transfer_params = {"do_remote_prefill": True}
    if remote_count:
        req.kv_transfer_params["remote_num_prompt_tokens"] = length if length <= 2 else length - 2
    assert scheduler.get_num_new_matched_tokens(req, computed) == (expected, True)
    assert req.num_prompt_tokens == length


def test_prompt_logprobs_skip_truncation(scheduler):
    req = request(129, skip=True)
    scheduler._truncate_request_for_prefill(req)
    assert req.num_prompt_tokens == 129


@pytest.mark.parametrize("stateful,lookahead,expected", [(False, 0, 0), (True, 0, 1), (True, 3, 2), (False, 3, 2)])
def test_backoff_combines_state_and_config(scheduler, stateful, lookahead, expected):
    scheduler.need_truncate = stateful
    scheduler.vllm_config.num_prefill_lookahead_tokens = lookahead
    assert scheduler._prefill_backoff() == expected


def test_decode_transfer_allocation_without_seed_fields(scheduler):
    req = request(513)
    req.kv_transfer_params = {
        "do_remote_prefill": True,
        "remote_num_prompt_tokens": 511,
        "remote_block_ids": ([5, 6, 7, 8],),
        "remote_engine_id": "producer",
        "remote_host": "localhost",
        "remote_port": 9000,
        "remote_request_id": "producer-request",
    }
    blocks = NS(get_unhashed_block_ids_all_groups=lambda: ([1, 2, 3, 4],), get_block_ids=lambda: ([1, 2, 3, 4],))
    credit, asynchronous = scheduler.get_num_new_matched_tokens(req, 0)
    assert (credit, asynchronous) == (511, True)
    scheduler.update_state_after_alloc(req, blocks, credit)
    assert scheduler._reqs_need_recv["req"] == (req, ([1, 2, 3, 4],), ([1, 2, 3, 4],), 511)
    assert not req.kv_transfer_params["do_remote_prefill"]
    assert scheduler.get_num_new_matched_tokens(req, 0) == (0, False)
    scheduler.build_connector_meta(NS())
    assert not scheduler._reqs_need_recv


def test_alloc_and_build_no_seed_metadata(scheduler):
    req = request(129)
    # P allocation must not require sampling params or a seed registry.
    scheduler.update_state_after_alloc(req, None, 0)
    req.kv_transfer_params = {"do_remote_prefill": True}
    scheduler.update_state_after_alloc(req, None, 0)
    meta = scheduler.build_connector_meta(NS())
    assert meta.reqs_in_batch == {"req"}
    assert not getattr(meta, "mtp_restore_requests", {})
    assert not getattr(meta, "mtp_producer_prompt_lens", {})
    assert getattr(meta, "mtp_seed_endpoint", None) is None


@pytest.mark.parametrize("length", [129, 513, 1, 2])
def test_finish_without_sampled_token_or_seed(scheduler, length):
    req = request(length)
    scheduler.on_new_request(req)
    delayed, params = scheduler.request_finished(req, ([1, 2, 3, 4, 5],))
    expected = length - 2 if length > 2 else length
    assert delayed
    assert params["remote_num_prompt_tokens"] == expected
    assert len(params["remote_block_ids"][0]) == math.ceil(expected / 128)
    assert "mtp3_seeds" not in params
    assert params.get("last_token_id") is None


def test_no_active_seed_capture_or_restore():
    targets = [
        (ROOT / "vllm_ascend/worker/worker.py", "NPUWorker", {"sample_tokens"}),
        (
            ROOT / "vllm_ascend/worker/v2/spec_decode/multi_module_mtp/speculator.py",
            "AscendMultiModuleMTPSpeculator",
            {"__init__", "propose", "_prepare_inputs"},
        ),
        (
            MOONCAKE / "pull_scheduler.py",
            "MooncakePullConnectorScheduler",
            {
                "__init__",
                "get_num_new_matched_tokens",
                "update_state_after_alloc",
                "build_connector_meta",
                "request_finished",
                "set_xfer_handshake_metadata_from_workers",
            },
        ),
        (MOONCAKE / "pull_scheduler.py", "MooncakeSchedulerSendingThread", {"__init__", "_run_busy_loop"}),
    ]
    forbidden = {
        "capture_mtp_seeds",
        "publish_mtp_seed",
        "restore_mtp_seed_batch",
        "decode_mtp_seed_params",
        "register_mtp_request",
        "take_mtp_seeds",
        "get_mtp_capture_context",
        "get_mtp_seed_store",
        "MTPSeedRegistry",
    }
    for path, name, methods in targets:
        tree = ast.parse(path.read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)
        for method in cls.body:
            if not isinstance(method, ast.FunctionDef) or method.name not in methods:
                continue
            for node in ast.walk(method):
                if isinstance(node, ast.Call):
                    called = getattr(node.func, "id", getattr(node.func, "attr", None))
                    assert called not in forbidden, (path, method.name, called)
                if isinstance(node, ast.keyword):
                    assert node.arg != "mtp_seed_hidden_size", (path, method.name)
