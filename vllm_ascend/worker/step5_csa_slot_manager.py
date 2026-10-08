# SPDX-License-Identifier: Apache-2.0
"""Runner-owned slots in a separate prefill CSA state pool."""

from collections import deque
from dataclasses import dataclass

import torch

CSA_SLOTS_PER_REQUEST = 2


@dataclass
class CSAMetadata:
    # Prefill 使用每请求的 region 数及源、目标槽位。
    region_count: torch.Tensor
    target_slot: torch.Tensor
    source_slot: torch.Tensor
    max_regions_per_request: int
    # 保留 CPU 源数组直到异步复制完成。
    host_buffers: tuple[torch.Tensor, ...]
    # 提示词结束或只剩一个 token（走原单 token 路径）时交接尾部。
    handoff: torch.Tensor | None = None
    needs_handoff: bool = False


@dataclass
class _RequestSlots:
    pair: int
    region: int = -1
    slot: int = -1


class CSASlotManager:
    """Stable request IDs own reservations; batch row numbers never own state."""

    def __init__(self, max_num_seqs: int, device: torch.device, pin_memory: bool):
        self.max_num_seqs = max_num_seqs
        self.device = device
        self.pin_memory = pin_memory
        self.free_pairs = deque(range(max_num_seqs))
        self.requests: dict[str, _RequestSlots] = {}
        self.handoff_buffer = torch.empty(max_num_seqs + 1, dtype=torch.int32, device=device)
        self.device_buffers = (
            torch.empty(max_num_seqs + 1, dtype=torch.int32, device=device),
            torch.empty(max_num_seqs + 1, dtype=torch.int32, device=device),
            torch.empty(max_num_seqs + 1, dtype=torch.int32, device=device),
        )

    def release(self, req_id: str) -> None:
        # The runner schedules device-side KV/CSA cleanup before calling this.
        reservation = self.requests.pop(req_id, None)
        if reservation is not None:
            self.free_pairs.append(reservation.pair)

    def check_prefill_handoff(self, req_ids: list[str]) -> None:
        # 单 token 中间分块也可能被归到原路径；尚未交接的私有状态不能静默跳过。
        for req_id in req_ids:
            reservation = self.requests.get(req_id)
            if reservation is not None and reservation.slot >= 0:
                raise NotImplementedError(
                    "CSA single-token intermediate prefill chunks require an explicit state handoff"
                )

    def dummy_prefill(self, num_requests: int) -> CSAMetadata:
        # Profiling has no live requests. Launch the same update entrypoint with
        # zero tasks, without allocating request reservations or modifying state.
        host = (
            torch.zeros(num_requests, dtype=torch.int32, pin_memory=self.pin_memory),
            torch.full((num_requests,), -1, dtype=torch.int32, pin_memory=self.pin_memory),
            torch.full((num_requests,), -1, dtype=torch.int32, pin_memory=self.pin_memory),
        )
        device = tuple(buffer[:num_requests] for buffer in self.device_buffers)
        for dst, src in zip(device, host):
            dst.copy_(src, non_blocking=True)
        return CSAMetadata(device[0], device[1], device[2], 0, host)

    def plan(
        self,
        req_ids: list[str],
        contexts: list[int],
        seq_lens: list[int],
        region_size: int,
        prompt_lens: list[int],
    ) -> CSAMetadata:
        # Paused requests retain their pairs; never reclaim them to hide exhaustion.
        if len(set(req_ids)) != len(req_ids):
            raise ValueError("CSA prefill requires unique request IDs")
        missing = sum(req_id not in self.requests for req_id in req_ids)
        if missing > len(self.free_pairs):
            raise RuntimeError("CSA request slot capacity exhausted before model execution")
        if len(req_ids) != len(contexts) or len(req_ids) != len(seq_lens):
            raise ValueError("CSA prefill request lengths do not match")
        if region_size <= 0 or any(c < 0 or s <= c for c, s in zip(contexts, seq_lens)):
            raise ValueError("CSA prefill requires positive query lengths and region size")

        if len(prompt_lens) != len(req_ids) or any(s > p for s, p in zip(seq_lens, prompt_lens)):
            raise ValueError("CSA prefill requires prompt lengths covering this chunk")

        region_counts: list[int] = []
        max_regions_per_request = 0
        targets: list[int] = []
        sources: list[int] = []
        pending: list[tuple[_RequestSlots, int, int]] = []
        handoffs: list[int] = []
        for req_id, context, seq_len, prompt_len in zip(req_ids, contexts, seq_lens, prompt_lens):
            reservation = self.requests.get(req_id)
            if reservation is None:
                reservation = _RequestSlots(self.free_pairs.popleft())
                self.requests[req_id] = reservation
            first_region = context // region_size
            last_region = (seq_len - 1) // region_size
            count = last_region - first_region + 1
            max_regions_per_request = max(max_regions_per_request, count)
            source = reservation.slot if reservation.region == first_region else -1
            if context % region_size and source < 0:
                raise NotImplementedError("CSA prefill cannot resume a partial region without its runner-owned state")
            sources.append(source)
            targets.append(-1)
            if seq_len % region_size:
                first_slot = CSA_SLOTS_PER_REQUEST * reservation.pair
                # Keep the same tail in place; otherwise choose the other address
                # so a completing first task and a new tail can execute in parallel.
                target = reservation.slot
                if reservation.region != last_region or target < 0:
                    target = first_slot + (target == first_slot)
                targets[-1] = target
                pending.append((reservation, last_region, target))
            else:
                pending.append((reservation, -1, -1))
            handoff = int(seq_len + 1 >= prompt_len and seq_len % region_size != 0)
            handoffs.append(handoff)
            if handoff:
                pending[-1] = (reservation, -1, -1)
            region_counts.append(count)

        host = tuple(torch.tensor(values, dtype=torch.int32, pin_memory=self.pin_memory)
                     for values in (region_counts, targets, sources))
        device = tuple(buffer[:value.numel()] for buffer, value in zip(self.device_buffers, host))
        for dst, src in zip(device, host):
            dst.copy_(src, non_blocking=True)
        for reservation, region, slot in pending:
            reservation.region = region
            reservation.slot = slot
        handoff_host = torch.tensor(handoffs, dtype=torch.int32, pin_memory=self.pin_memory)
        handoff_device = self.handoff_buffer[:len(req_ids)]
        handoff_device.copy_(handoff_host, non_blocking=True)
        return CSAMetadata(device[0], device[1], device[2], max_regions_per_request,
                           (*host, handoff_host), handoff_device, any(handoffs))
