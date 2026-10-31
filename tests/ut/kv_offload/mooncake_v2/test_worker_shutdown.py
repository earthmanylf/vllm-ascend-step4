# SPDX-License-Identifier: Apache-2.0
"""Normal shutdown drains Python work; it is not a native failure DMA fence."""

import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace as NS

import pytest

from .test_step5_transfer_contract import load_definitions


@pytest.fixture
def receiver_type():
    namespace = dict(
        threading=threading,
        queue=queue,
        torch=NS(npu=NS(set_device=lambda device: None)),
        logger=NS(exception=lambda *args: None, error=lambda *args: None),
        KVConnectorTransferResults=lambda: NS(finished_recving=set(), failed_recving=set()),
    )
    load_definitions("pull_worker.py", {"MooncakePullRecvingThread"}, namespace)
    return namespace["MooncakePullRecvingThread"]


def make_receiver(cls):
    thread = cls.__new__(cls)
    threading.Thread.__init__(thread, daemon=True)
    thread.device = 0
    thread.is_step5 = True
    thread.ready_event = threading.Event()
    thread.request_queue = queue.Queue()
    thread.finished_requests = queue.SimpleQueue()
    thread.invalid_block_ids = set()
    thread.invalid_block_ids_lock = threading.Lock()
    thread.can_report_invalid_block_ids = False
    thread.quarantined_requests = {}
    thread.executor = ThreadPoolExecutor(max_workers=1)
    thread._closing = False
    thread._closed_before_start = False
    thread._close_lock = threading.Lock()
    thread._handle_requests = lambda *args: set()
    return thread


def test_idle_shutdown_wakes_joins_and_is_idempotent(receiver_type):
    thread = make_receiver(receiver_type)
    thread.start()
    assert thread.ready_event.wait(2)
    thread.shutdown()
    thread.shutdown()
    assert not thread.is_alive()
    assert thread.request_queue.unfinished_tasks == 0
    with pytest.raises(RuntimeError):
        thread.add_requests("p", {"r": NS(remote_host="p", remote_port=1)})
    with pytest.raises(RuntimeError):
        thread.executor.submit(lambda: None)


def test_shutdown_processes_accepted_work_fifo(receiver_type):
    thread = make_receiver(receiver_type)
    calls = []
    thread._handle_requests = lambda engine, host, port, requests: calls.append(next(iter(requests))) or set()
    for name in ("a", "b"):
        thread.add_requests("p", {name: NS(remote_host="p", remote_port=1)})
    thread.start()
    thread.shutdown()
    assert calls == ["a", "b"]
    assert thread.get_and_clear_finished_requests().finished_recving == {"a", "b"}
    assert not thread.is_alive()
    assert thread.request_queue.unfinished_tasks == 0


def test_shutdown_does_not_certify_unknown_native_completion(receiver_type):
    thread = make_receiver(receiver_type)
    retained = NS(remote_host="p", remote_port=1)
    thread._handle_requests = lambda *args: {"unsafe"}
    thread.add_requests("p", {"unsafe": retained})
    thread.start()
    with pytest.raises(RuntimeError, match="quarantin"):
        thread.shutdown()
    assert not thread.is_alive()
    assert thread.quarantined_requests == {"unsafe": retained}
    assert not thread.get_and_clear_finished_requests().finished_recving


def test_facade_shutdown_closes_worker_and_scheduler():
    namespace = dict(
        KVConnectorBase_V1=type("Base", (), {}), SupportsHMA=type("HMA", (), {}), vllm_version_is=lambda *args: False
    )
    load_definitions("connector.py", {"MooncakeBaseConnector"}, namespace)
    connector = namespace["MooncakeBaseConnector"].__new__(namespace["MooncakeBaseConnector"])
    calls = []
    connector.connector_worker = NS(shutdown=lambda: calls.append("worker"))
    connector.connector_scheduler = NS(shutdown=lambda: calls.append("scheduler"))
    connector.shutdown()
    assert calls == ["worker", "scheduler"]


def test_producer_worker_shutdown_without_receive_thread():
    namespace = dict(MooncakeBaseConnectorWorker=type("Base", (), {}))
    load_definitions("pull_worker.py", {"MooncakePullConnectorWorker"}, namespace)
    worker = namespace["MooncakePullConnectorWorker"].__new__(namespace["MooncakePullConnectorWorker"])
    worker._recving_thread = None
    worker.shutdown()


def test_unstarted_receiver_keeps_accepted_work(receiver_type):
    thread = make_receiver(receiver_type)
    thread.add_requests("p", {"a": NS(remote_host="p", remote_port=1)})
    with pytest.raises(RuntimeError, match="unstarted"):
        thread.shutdown()
    assert not thread._closing
    assert thread.request_queue.qsize() == 1
    thread.start()
    thread.shutdown()
    assert thread.get_and_clear_finished_requests().finished_recving == {"a"}


def test_empty_unstarted_shutdown_then_start_exits(receiver_type):
    thread = make_receiver(receiver_type)
    thread.shutdown()
    thread.shutdown()
    assert thread.request_queue.unfinished_tasks == 0
    thread.start()
    thread.join(timeout=2)
    assert not thread.is_alive()


def test_two_concurrent_shutdowns_are_bounded(receiver_type):
    thread = make_receiver(receiver_type)
    thread.start()
    assert thread.ready_event.wait(2)
    errors = []

    def close():
        try:
            thread.shutdown()
        except Exception as error:
            errors.append(error)

    closers = [threading.Thread(target=close, daemon=True) for _ in range(2)]
    for closer in closers:
        closer.start()
    for closer in closers:
        closer.join(timeout=2)
        assert not closer.is_alive()
    assert not errors and not thread.is_alive()


def test_receiver_rejects_self_join(receiver_type, monkeypatch):
    thread = make_receiver(receiver_type)
    with monkeypatch.context() as context:
        context.setattr(threading, "current_thread", lambda: thread)
        with pytest.raises(RuntimeError, match="join itself"):
            thread.shutdown()
    assert not thread._closing
    thread.shutdown()


def test_facade_scheduler_cleanup_after_worker_failure():
    namespace = dict(
        KVConnectorBase_V1=type("Base", (), {}), SupportsHMA=type("HMA", (), {}), vllm_version_is=lambda *args: False
    )
    load_definitions("connector.py", {"MooncakeBaseConnector"}, namespace)
    connector = namespace["MooncakeBaseConnector"].__new__(namespace["MooncakeBaseConnector"])
    calls = []

    def fail():
        calls.append("worker")
        raise RuntimeError("retained unknown READ")

    connector.connector_worker = NS(shutdown=fail)
    connector.connector_scheduler = NS(shutdown=lambda: calls.append("scheduler"))
    retained_worker = connector.connector_worker
    with pytest.raises(RuntimeError, match="retained"):
        connector.shutdown()
    assert calls == ["worker", "scheduler"]
    assert connector.connector_worker is retained_worker
