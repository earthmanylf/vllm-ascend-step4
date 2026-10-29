# SPDX-License-Identifier: Apache-2.0
"""Immutable request-private seed for the first Step5 MTP3 decode proposal."""

import base64
import binascii
import hashlib
import sys
import threading
from dataclasses import dataclass, field
from typing import Any

import torch

MTP_SEED_VERSION = 1
MTP_SEED_MESSAGE = b"publish_mtp_seed_v1"
MAX_SEED_HIDDEN_SIZE = 16384


def is_step5_mtp3_pd(vllm_config: Any) -> bool:
    transfer = getattr(vllm_config, "kv_transfer_config", None)
    spec = getattr(vllm_config, "speculative_config", None)
    return bool(
        transfer is not None
        and transfer.is_kv_transfer_instance
        and getattr(vllm_config, "use_v2_model_runner", False)
        and getattr(getattr(vllm_config.model_config, "hf_text_config", None), "model_type", None) == "step5"
        and spec is not None
        and spec.method == "mtp"
        and spec.num_speculative_tokens == 3
    )


@dataclass(frozen=True)
class MTPSeedState:
    producer_request_id: str
    position: int
    token_id: int
    tp_rank: int
    hidden_size: int
    hidden_bytes: bytes = field(repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.producer_request_id, str) or not 0 < len(self.producer_request_id) <= 1024:
            raise ValueError("Invalid MTP seed request identity")
        for value in (self.position, self.token_id, self.tp_rank):
            if type(value) is not int or value < 0:
                raise ValueError("Invalid MTP seed position, token or rank")
        if type(self.hidden_size) is not int or not 0 < self.hidden_size <= MAX_SEED_HIDDEN_SIZE:
            raise ValueError("Invalid MTP seed hidden dimension")
        if type(self.hidden_bytes) is not bytes or len(self.hidden_bytes) != self.hidden_size * 2:
            raise ValueError("MTP seed byte length does not match BF16 hidden dimension")
        hidden = torch.frombuffer(bytearray(self.hidden_bytes), dtype=torch.bfloat16)
        if not torch.isfinite(hidden).all():
            raise ValueError("Nonfinite MTP seed hidden state")

    @classmethod
    def snapshot(cls, hidden: torch.Tensor, **context: Any) -> "MTPSeedState":
        if hidden.ndim != 1 or hidden.dtype != torch.bfloat16:
            raise ValueError("MTP seed requires one BF16 target hidden row")
        if not 0 < hidden.numel() <= MAX_SEED_HIDDEN_SIZE:
            raise ValueError("MTP seed hidden row is too large or empty")
        # An owning CPU copy, not a view into a producer request slot that can
        # later be overwritten. This copy does not certify native READ drain.
        data = hidden.detach().to("cpu").contiguous().view(torch.uint8).numpy().tobytes()
        return cls(hidden_size=hidden.numel(), hidden_bytes=data, **context)

    def to_wire(self) -> dict[str, Any]:
        return {
            "version": MTP_SEED_VERSION,
            "dtype": "torch.bfloat16",
            "byteorder": sys.byteorder,
            "producer_request_id": self.producer_request_id,
            "position": self.position,
            "token_id": self.token_id,
            "tp_rank": self.tp_rank,
            "hidden_size": self.hidden_size,
            "sha256": hashlib.sha256(self.hidden_bytes).hexdigest(),
            "hidden": base64.b64encode(self.hidden_bytes).decode("ascii"),
        }

    @classmethod
    def from_wire(
        cls,
        wire: dict[str, Any],
        *,
        expected_request_id: str,
        expected_position: int,
        expected_token_id: int,
        expected_tp_rank: int,
        expected_hidden_size: int,
    ) -> "MTPSeedState":
        if type(wire) is not dict:
            raise ValueError("Invalid MTP seed envelope")
        expected = {
            "version": MTP_SEED_VERSION,
            "position": expected_position,
            "token_id": expected_token_id,
            "tp_rank": expected_tp_rank,
            "hidden_size": expected_hidden_size,
        }
        for name, value in expected.items():
            if type(wire.get(name)) is not int or wire[name] != value:
                raise ValueError(f"MTP seed context mismatch: {name}")
        if wire.get("producer_request_id") != expected_request_id or wire.get("dtype") != "torch.bfloat16":
            raise ValueError("MTP seed request or dtype mismatch")
        if wire.get("byteorder") != sys.byteorder or not 0 < expected_hidden_size <= MAX_SEED_HIDDEN_SIZE:
            raise ValueError("Unsupported MTP seed byte order or dimension")
        encoded = wire.get("hidden")
        if type(encoded) is not str or len(encoded) != ((expected_hidden_size * 2 + 2) // 3) * 4:
            raise ValueError("Invalid MTP seed encoded length")
        try:
            data = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("Invalid MTP seed payload") from exc
        if wire.get("sha256") != hashlib.sha256(data).hexdigest():
            raise ValueError("MTP seed payload checksum mismatch")
        return cls(
            expected_request_id, expected_position, expected_token_id, expected_tp_rank, expected_hidden_size, data
        )


def publish_mtp_seed(endpoint: tuple[str, int, str], seed: MTPSeedState, *, timeout_ms: int = 10000) -> bool:
    """Wait for scheduler storage ACK before returning a producer model output."""
    import msgspec
    import zmq

    host, port, engine_id = endpoint
    if not isinstance(host, str) or type(port) is not int or not 0 < port < 65536:
        raise ValueError("Invalid MTP seed scheduler endpoint")
    if type(timeout_ms) is not int or not 0 < timeout_ms <= 30000:
        raise ValueError("Invalid MTP seed publication timeout")
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    payload = msgspec.msgpack.encode((MTP_SEED_MESSAGE, engine_id, seed.to_wire()))
    with zmq.Context() as context, context.socket(zmq.REQ) as sock:
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.SNDTIMEO, timeout_ms)
        sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
        sock.connect(f"tcp://{host}:{port}")
        sock.send(payload)
        response = sock.recv()
        if response == b"ACK":
            return True
        if response == b"MTP_SEED_IGNORED":
            return False
        raise RuntimeError("MTP seed scheduler rejected publication")


class MTPSeedRegistry:
    """Scheduler-side CPU ownership; publication ACK never means KV READ completion."""

    def __init__(self, engine_id: str, *, tp_size: int, hidden_size: int) -> None:
        if type(tp_size) is not int or not 0 < tp_size <= 256:
            raise ValueError("Invalid MTP seed TP size")
        if type(hidden_size) is not int or not 0 < hidden_size <= MAX_SEED_HIDDEN_SIZE:
            raise ValueError("Invalid MTP seed hidden size")
        self.engine_id = engine_id
        self.tp_size = tp_size
        self.hidden_size = hidden_size
        self._prompts: dict[str, int] = {}
        self._states: dict[str, dict[int, MTPSeedState]] = {}
        self._lock = threading.RLock()

    def register(self, request_id: str, prompt_len: int) -> None:
        if not isinstance(request_id, str) or not 0 < len(request_id) <= 1024:
            raise ValueError("Invalid MTP seed request identity")
        if type(prompt_len) is not int or prompt_len <= 0:
            raise ValueError("Invalid MTP seed prompt length")
        with self._lock:
            if request_id in self._prompts and self._prompts[request_id] != prompt_len:
                raise ValueError("Conflicting MTP seed request registration")
            self._prompts[request_id] = prompt_len
            self._states.setdefault(request_id, {})

    def publish(self, engine_id: str, wire: dict[str, Any]) -> bool:
        if engine_id != self.engine_id:
            return False
        if type(wire) is not dict or not isinstance(wire.get("producer_request_id"), str):
            raise ValueError("Invalid MTP seed publication envelope")
        request_id = wire["producer_request_id"]
        with self._lock:
            prompt_len = self._prompts.get(request_id)
            if prompt_len is None:
                return False  # Late/cancelled requests cannot recreate retained state.
            rank = wire.get("tp_rank")
            if type(rank) is not int or not 0 <= rank < self.tp_size:
                raise ValueError("Invalid MTP seed publication TP rank")
            state = MTPSeedState.from_wire(
                wire,
                expected_request_id=request_id,
                expected_position=prompt_len - 1,
                expected_token_id=wire.get("token_id"),
                expected_tp_rank=rank,
                expected_hidden_size=self.hidden_size,
            )
            existing = self._states[request_id].get(rank)
            if existing is not None and existing != state:
                raise ValueError("Conflicting MTP seed publication")
            self._states[request_id][rank] = state
            return True

    def take(self, request_id: str, token_id: int) -> dict[str, dict[str, Any]]:
        with self._lock:
            states = self._states.get(request_id, {})
            if set(states) != set(range(self.tp_size)):
                raise ValueError("MTP3 handoff requires a seed from every TP worker")
            if any(state.token_id != token_id for state in states.values()):
                raise ValueError("MTP seed token differs from producer's sampled token")
            result = {str(rank): state.to_wire() for rank, state in sorted(states.items())}
            self.discard(request_id)
            return result

    def clear(self) -> None:
        with self._lock:
            self._prompts.clear()
            self._states.clear()

    def discard(self, request_id: str) -> None:
        with self._lock:
            self._prompts.pop(request_id, None)
            self._states.pop(request_id, None)


def restore_mtp_seed(
    seed: MTPSeedState,
    cached_input_ids: torch.Tensor,
    cached_hidden_states: torch.Tensor,
    request_slot: int,
) -> None:
    """Seed only the preceding row; the first local query must fill the next row."""
    if (
        cached_input_ids.ndim != 2
        or cached_input_ids.shape[1] != 2
        or cached_hidden_states.ndim != 3
        or cached_hidden_states.shape[1:] != (2, seed.hidden_size)
        or cached_input_ids.shape[0] != cached_hidden_states.shape[0]
        or cached_input_ids.dtype != torch.int64
        or cached_hidden_states.dtype != torch.bfloat16
        or cached_input_ids.device != cached_hidden_states.device
    ):
        raise ValueError("MTP3 cached input/hidden buffer geometry mismatch")
    if type(request_slot) is not int or not 0 <= request_slot < cached_input_ids.shape[0]:
        raise ValueError("Invalid MTP seed destination request slot")
    # Clone owns CPU storage while the synchronous device copy is submitted.
    hidden = torch.frombuffer(bytearray(seed.hidden_bytes), dtype=torch.bfloat16).clone()
    hidden = hidden.to(cached_hidden_states.device)
    cached_input_ids[request_slot].zero_()
    cached_hidden_states[request_slot].zero_()
    cached_input_ids[request_slot, 0] = seed.token_id
    cached_hidden_states[request_slot, 0].copy_(hidden)


def decode_mtp_seed_params(params: dict[str, Any], *, tp_size: int, hidden_size: int) -> dict[int, MTPSeedState]:
    seeds = params.get("mtp3_seeds")
    if type(seeds) is not dict or set(seeds) != {str(rank) for rank in range(tp_size)}:
        raise ValueError("MTP3 remote prefill requires seeds from every TP worker")
    prompt_len = params.get("remote_num_prompt_tokens")
    token_id = params.get("last_token_id")
    request_id = params.get("remote_request_id")
    if type(prompt_len) is not int or prompt_len <= 0 or type(token_id) is not int or token_id < 0:
        raise ValueError("Invalid MTP3 producer handoff token context")
    return {
        rank: MTPSeedState.from_wire(
            seeds[str(rank)],
            expected_request_id=request_id,
            expected_position=prompt_len - 1,
            expected_token_id=token_id,
            expected_tp_rank=rank,
            expected_hidden_size=hidden_size,
        )
        for rank in range(tp_size)
    }


def restore_mtp_seed_batch(
    speculator: Any,
    input_batch: Any,
    num_rejected: torch.Tensor,
    states: dict[str, MTPSeedState],
) -> set[str]:
    pending = []
    slots = set()
    for index, request_id in enumerate(input_batch.req_ids[: input_batch.num_reqs]):
        state = states.get(request_id)
        if state is None:
            continue
        start = int(input_batch.query_start_loc_np[index])
        slot = int(input_batch.idx_mapping_np[index])
        if not 0 <= start < input_batch.positions.numel() or not 0 <= slot < speculator.cached_draft_input_ids.shape[0]:
            raise ValueError("Invalid MTP3 first proposal request position or slot")
        position = int(input_batch.positions[start].detach().to("cpu").item())
        token_id = int(input_batch.input_ids[start].detach().to("cpu").item())
        rejected = int(num_rejected[index].detach().to("cpu").item())
        if position != state.position + 1 or token_id != state.token_id or rejected != 0:
            raise ValueError("MTP3 first proposal does not match the producer handoff context")
        if slot in slots or state.hidden_size != speculator.cached_target_hidden_states.shape[-1]:
            raise ValueError("MTP3 seed destination collision or hidden dimension mismatch")
        slots.add(slot)
        pending.append((request_id, state, slot))
    cached_embeds = getattr(speculator, "cached_draft_input_embeds", None)
    seed_embeds = None
    if pending and cached_embeds is not None:
        hidden_cache = speculator.cached_target_hidden_states
        if (
            cached_embeds.shape != hidden_cache.shape
            or cached_embeds.device != hidden_cache.device
            or cached_embeds.dtype != hidden_cache.dtype
        ):
            raise ValueError("MTP3 cached embedding buffer geometry mismatch")
        # These are P's sampled vocabulary tokens, not image placeholders.
        # Reconstruct with the same draft embedding path used by upstream MTP.
        token_ids = torch.tensor(
            [state.token_id for _, state, _ in pending], dtype=torch.int64, device=cached_embeds.device
        )
        seed_embeds = speculator.model.embed_input_ids(token_ids)
        if seed_embeds.shape != (len(pending), cached_embeds.shape[-1]):
            raise ValueError("MTP3 seed embedding shape mismatch")
    for index, (request_id, state, slot) in enumerate(pending):
        restore_mtp_seed(state, speculator.cached_draft_input_ids, speculator.cached_target_hidden_states, slot)
        if seed_embeds is not None:
            cached_embeds[slot].zero_()
            cached_embeds[slot, 0].copy_(seed_embeds[index])
        del states[request_id]
    return {request_id for request_id, _, _ in pending}


def capture_mtp_seeds(
    speculator: Any,
    input_batch: Any,
    producer_prompt_lens: dict[str, int],
    *,
    tp_rank: int,
    ready_event: Any = None,
) -> dict[str, MTPSeedState]:
    """Capture after the drafter writes its cache, only at the original prompt end."""
    if not producer_prompt_lens:
        return {}
    if speculator.num_speculative_steps != 3:
        raise ValueError("MTP seed capture requires the three-module profile")
    hidden = speculator.cached_target_hidden_states
    tokens = speculator.cached_draft_input_ids
    if (
        hidden.ndim != 3
        or hidden.shape[1] != 2
        or tokens.shape != hidden.shape[:2]
        or hidden.dtype != torch.bfloat16
        or tokens.dtype != torch.int64
        or tokens.device != hidden.device
    ):
        raise ValueError("Invalid producer MTP cache window")
    if hidden.device.type != "cpu":
        if ready_event is None:
            raise ValueError("NPU MTP seed capture requires the owning stream readiness event")
        ready_event.synchronize()
    num_reqs = input_batch.num_reqs
    if (
        type(num_reqs) is not int
        or not 0 <= num_reqs <= len(input_batch.req_ids)
        or num_reqs > len(input_batch.idx_mapping_np)
        or num_reqs > speculator.last_token_indices.numel()
    ):
        raise ValueError("Invalid MTP capture request count")
    indices = speculator.last_token_indices[:num_reqs].detach().to("cpu").tolist()
    if any(type(index) is not int or not 0 <= index < speculator.input_buffers.positions.numel() for index in indices):
        raise ValueError("Invalid MTP capture last-token index")
    positions = speculator.input_buffers.positions[indices].detach().to("cpu").tolist()
    result = {}
    for index, request_id in enumerate(input_batch.req_ids[:num_reqs]):
        prompt_len = producer_prompt_lens.get(request_id)
        if prompt_len is None:
            continue
        if type(prompt_len) is not int or prompt_len <= 0:
            raise ValueError("Invalid producer prompt length for MTP seed")
        position = positions[index]
        if position != prompt_len - 1:
            continue
        slot = int(input_batch.idx_mapping_np[index])
        if not 0 <= slot < hidden.shape[0]:
            raise ValueError("Invalid producer MTP request slot")
        token_id = int(tokens[slot, -1].detach().to("cpu").item())
        result[request_id] = MTPSeedState.snapshot(
            hidden[slot, -1],
            producer_request_id=request_id,
            position=position,
            token_id=token_id,
            tp_rank=tp_rank,
        )
    return result


def redact_mtp_seed_params(params: dict[str, Any] | None) -> dict[str, Any] | None:
    """Do not log prompt-derived hidden-state bytes in transfer parameter logs."""
    if params is None or "mtp3_seeds" not in params:
        return params
    redacted = dict(params)
    redacted["mtp3_seeds"] = "<request-private MTP3 seed omitted>"
    return redacted
