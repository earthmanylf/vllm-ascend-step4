# SPDX-License-Identifier: Apache-2.0
"""Unknown Step5 READ completion must not enter the normal free/recompute path."""

import queue
import threading
from types import SimpleNamespace as NS

import pytest

from .test_step5_transfer_contract import load_definitions


class StopReceiver(BaseException):
    pass


class BoundedQueue(queue.Queue):
    def get(self, *args, **kwargs):
        if self.empty():
            raise StopReceiver
        return super().get(*args, **kwargs)


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


@pytest.mark.parametrize("is_step5", [True, False])
@pytest.mark.parametrize("raises", [True, False])
@pytest.mark.parametrize("can_report", [True, False])
def test_unknown_completion_never_published_for_step5(receiver_type, is_step5, raises, can_report):
    thread = receiver_type.__new__(receiver_type)
    thread.device = 0
    thread._closed_before_start = False
    thread.is_step5 = is_step5
    thread.ready_event = threading.Event()
    thread.request_queue = BoundedQueue()
    thread.finished_requests = queue.SimpleQueue()
    thread.invalid_block_ids = set()
    thread.invalid_block_ids_lock = threading.Lock()
    thread.can_report_invalid_block_ids = can_report
    thread.quarantined_requests = {}
    unsafe = NS(remote_host="p", remote_port=1, local_block_ids=([4], [7], [9]))
    healthy = NS(remote_host="p", remote_port=1, local_block_ids=([5], [8], [10]))
    calls = []

    def handle(engine, host, port, requests):
        calls.append(set(requests))
        if "unsafe" in requests:
            if raises:
                raise RuntimeError("READ completion is unknown; late native DMA may remain")
            return {"unsafe"}
        return set()

    thread._handle_requests = handle
    thread.request_queue.put(("p", {"unsafe": unsafe}))
    thread.request_queue.put(("p", {"healthy": healthy}))
    with pytest.raises(StopReceiver):
        thread.run()
    result = thread.get_and_clear_finished_requests()
    assert result.finished_recving == ({"healthy"} if is_step5 else {"unsafe", "healthy"})
    assert result.failed_recving == (set() if is_step5 else {"unsafe"})
    assert thread.quarantined_requests == ({"unsafe": unsafe} if is_step5 else {})
    assert thread.get_and_clear_invalid_block_ids() == ({4, 7, 9} if can_report and not is_step5 else set())
    assert calls == [{"unsafe"}, {"healthy"}]
    assert thread.request_queue.unfinished_tasks == 0

    # A duplicate already queued cannot dispatch another READ into retained blocks.
    thread.request_queue.put(("p", {"unsafe": unsafe}))
    if is_step5:
        with pytest.raises(StopReceiver):
            thread.run()
        assert calls == [{"unsafe"}, {"healthy"}]
        assert not thread.get_and_clear_finished_requests().finished_recving
        assert thread.quarantined_requests["unsafe"] is unsafe
