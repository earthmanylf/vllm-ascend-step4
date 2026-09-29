# SPDX-License-Identifier: Apache-2.0
"""Runner-owned reservations for ordinary prompt-prefill CSA state.

The device allocator owns a disjoint prefix of the active cache. Each request
managed here reserves a pair in the suffix until finish/preemption. This lets
decode consume a prefill tail without letting its allocator reuse the reserved
address behind the CPU manager's back.
"""

from collections import deque
from dataclasses import dataclass

import torch

CSA_DEVICE_SLOTS_PER_REQUEST = 8
CSA_PREFILL_SLOTS_PER_REQUEST = 2


@dataclass
class CSAPrefillSlotPlan:
    task_prefix: torch.Tensor
    target_slots: torch.Tensor
    source_slots: torch.Tensor
    num_tasks: int
    # Keep the source buffers alive while their asynchronous copies are pending.
    host_buffers: tuple[torch.Tensor, ...]


@dataclass
class _RequestSlots:
    pair: int
    region: int = -1
    slot: int = -1


class CSASlotManager:
    """Stable request IDs own reservations; batch row numbers never own state."""

    def __init__(self, max_num_seqs: int, max_num_tokens: int, device: torch.device, pin_memory: bool):
        self.max_num_seqs = max_num_seqs
        self.device = device
        self.pin_memory = pin_memory
        self.slot_base = CSA_DEVICE_SLOTS_PER_REQUEST * max_num_seqs
        self.free_pairs = deque(range(max_num_seqs))
        self.requests: dict[str, _RequestSlots] = {}
        self.device_buffers = (
            torch.empty(max_num_seqs + 1, dtype=torch.int32, device=device),
            torch.empty(max_num_tokens, dtype=torch.int32, device=device),
            torch.empty(max_num_tokens, dtype=torch.int32, device=device),
        )

    def release(self, req_id: str) -> None:
        # The runner schedules device-side KV/CSA cleanup before calling this.
        reservation = self.requests.pop(req_id, None)
        if reservation is not None:
            self.free_pairs.append(reservation.pair)

    def plan(
        self,
        req_ids: list[str],
        contexts: list[int],
        seq_lens: list[int],
        region_size: int,
    ) -> CSAPrefillSlotPlan | None:
        # A paused request retains its pair. If too many such requests exist,
        # leave this batch on the device-managed path without reclaiming state.
        if len(set(req_ids)) != len(req_ids):
            raise ValueError("CSA prefill requires unique request IDs")
        missing = sum(req_id not in self.requests for req_id in req_ids)
        if missing > len(self.free_pairs):
            return None
        if len(req_ids) != len(contexts) or len(req_ids) != len(seq_lens):
            raise ValueError("CSA prefill request lengths do not match")
        if region_size <= 0 or any(c < 0 or s <= c for c, s in zip(contexts, seq_lens)):
            return None

        prefix = [0]
        targets: list[int] = []
        sources: list[int] = []
        pending: list[tuple[_RequestSlots, int, int]] = []
        for req_id, context, seq_len in zip(req_ids, contexts, seq_lens):
            reservation = self.requests.get(req_id)
            if reservation is None:
                reservation = _RequestSlots(self.free_pairs.popleft())
                self.requests[req_id] = reservation
            first_region = context // region_size
            last_region = (seq_len - 1) // region_size
            count = last_region - first_region + 1
            source = reservation.slot if reservation.region == first_region else -1
            sources.extend([source] + [-1] * (count - 1))
            targets.extend([-1] * count)
            if seq_len % region_size:
                first_slot = self.slot_base + CSA_PREFILL_SLOTS_PER_REQUEST * reservation.pair
                # Keep the same tail in place; otherwise choose the other address
                # so a completing first task and a new tail can execute in parallel.
                target = reservation.slot
                if reservation.region != last_region or target < 0:
                    target = first_slot + (target == first_slot)
                targets[-1] = target
                pending.append((reservation, last_region, target))
            else:
                pending.append((reservation, -1, -1))
            prefix.append(prefix[-1] + count)

        host = tuple(torch.tensor(values, dtype=torch.int32, pin_memory=self.pin_memory)
                     for values in (prefix, targets, sources))
        device = tuple(buffer[:value.numel()] for buffer, value in zip(self.device_buffers, host))
        for dst, src in zip(device, host):
            dst.copy_(src, non_blocking=True)
        for reservation, region, slot in pending:
            reservation.region = region
            reservation.slot = slot
        return CSAPrefillSlotPlan(device[0], device[1], device[2], prefix[-1], host)
