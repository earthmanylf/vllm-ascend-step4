# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Benchmark Prefill/Decode TPS for different batch sizes and input lengths.

This script benchmarks a running vLLM service to measure:
- Prefill TPS (tokens per second during prefill phase)
- Decode TPS (tokens per second during decode phase)
- Theoretical FLOPS (floating point operations per second)
- MFU (Model FLOPS Utilization) - efficiency metric

The script automatically adjusts max_tokens based on the formula:
max_tokens = (batch_size * input_len / max_num_batched_tokens) * 2

This accounts for chunked_prefill with decode priority scheduling.

Usage:
    # Start your vLLM service first:
    vllm serve <your_model> --port 8000

    # Basic benchmark (TPS only):
    python benchmark_pd_tps.py \
        --host 127.0.0.1 \
        --port 8000 \
        --model <your_model> \
        --batch-sizes 1,2,4,8 \
        --input-lens 128,512,1024,2048 \
        --max-num-batched-tokens 8192 \
        --num-iters 5

    # Full benchmark with FLOPS and MFU calculation:
    python benchmark_pd_tps.py \
        --host 127.0.0.1 \
        --port 8000 \
        --model <your_model> \
        --tp-size 2 \
        --dp-size 1 \
        --precision bf16 \
        --batch-sizes 1,8,32 \
        --input-lens 1024,4096 \
        --max-num-batched-tokens 8192

    # For very high batch sizes, increase max connections to avoid bottlenecks:
    python benchmark_pd_tps.py \
        --host 127.0.0.1 \
        --port 8000 \
        --model <your_model> \
        --batch-sizes 64,128,256 \
        --input-lens 1024,2048 \
        --max-num-batched-tokens 16384 \
        --max-connections 2000

    # Stream-based measurement provides accurate prefill/decode separation:
    python benchmark_pd_tps.py \
        --host 127.0.0.1 \
        --port 8000 \
        --model <your_model> \
        --batch-sizes 1,8,32 \
        --input-lens 1024,4096 \
        --max-num-batched-tokens 8192 \
        --num-iters 3

Note: To enable FLOPS and MFU calculations, you need:
1. The model's config.json file
2. Correct parallelism settings (--tp-size, --dp-size)
3. The flops_calculator module must be available
"""

import argparse
import asyncio
import codecs
import hashlib
import json
import math
import random
import statistics
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

try:
    import aiohttp
except ImportError:
    raise ImportError("aiohttp is required for this benchmark. Install it with: pip install aiohttp") from None

from flops_calculator import sequence_flops

KV_CACHE_GUARD_FACTOR = 0.9


def load_capacity_metadata(path: str | None, dp_size: int) -> dict[str, Any]:
    """Only explicit externally verified logical DP capacities; never infer replicas."""
    if path is None:
        raise ValueError(
            "KV capacity unknown: supply --capacity-metadata from verified engine logs; no capacity is guessed"
        )
    with open(path, encoding="utf-8") as handle:
        metadata = json.load(handle)
    capacities = metadata.get("per_dp_capacity_tokens")
    if not isinstance(capacities, dict) or set(capacities) != {str(i) for i in range(dp_size)}:
        raise ValueError("per_dp_capacity_tokens must contain exactly ranks 0..dp_size-1")
    for value in [*capacities.values(), metadata.get("max_model_len")]:
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError("capacity tokens and max_model_len must be positive integers")
    if not isinstance(metadata.get("source"), str) or not metadata["source"].strip():
        raise ValueError("capacity metadata requires a nonempty verified source")
    return metadata


def require_completed_results(results: dict[str, Any]) -> None:
    cases = [value for key, value in results.items() if key.startswith("bs")]
    if not cases or any(value.get("status") != "completed" for value in cases):
        raise RuntimeError("Suite rejected: every selected case must complete; skipped/empty results are not success")
    if any(
        not value.get("decode_window_quality", {}).get("validated") or not value["decode_window_quality"].get("valid")
        for value in cases
    ):
        raise RuntimeError("Suite rejected: decode quality must be validated and valid")


def _extract_weight_version(value: Any) -> int | None:
    """Best-effort extraction for server weight_version response shapes."""
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    if isinstance(value, dict):
        for key in ("weight_version", "current_weight_version", "version"):
            if key in value:
                version = _extract_weight_version(value[key])
                if version is not None:
                    return version
        for nested_value in value.values():
            version = _extract_weight_version(nested_value)
            if version is not None:
                return version
    if isinstance(value, list):
        for item in value:
            version = _extract_weight_version(item)
            if version is not None:
                return version
    return None


def _as_positive_int(value: Any) -> int:
    if value is None or isinstance(value, bool):
        return 0
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return parsed if parsed > 0 else 0


def _iter_dicts(value: Any):
    if isinstance(value, dict):
        yield value
        for nested_value in value.values():
            yield from _iter_dicts(nested_value)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_dicts(item)


def _select_transformer_config_for_flops(config: dict[str, Any]) -> dict[str, Any]:
    text_config = config.get("text_config")
    if isinstance(text_config, dict):
        return text_config
    return config


def _load_transformer_config_for_flops(model_name: str) -> dict[str, Any] | None:
    config_path = Path(model_name) / "config.json"
    if config_path.exists():
        try:
            with open(config_path, encoding="utf-8") as f:
                raw_config = json.load(f)
            transformer_config = _select_transformer_config_for_flops(raw_config)
            print(f"Loaded transformer config from: {config_path}")
            return transformer_config
        except Exception as e:
            print(f"Warning: Failed to load config from {config_path}: {e}")
            return None

    print(f"Warning: Config file not found at {config_path}")
    return None


def _kv_cache_tokens_from_blocks(config: dict[str, Any]) -> int:
    num_gpu_blocks = _as_positive_int(config.get("num_gpu_blocks"))
    if num_gpu_blocks == 0:
        num_gpu_blocks = _as_positive_int(config.get("num_gpu_blocks_override"))
    if num_gpu_blocks == 0:
        num_gpu_blocks = _as_positive_int(config.get("num_blocks"))
    block_size = _as_positive_int(config.get("block_size"))
    if num_gpu_blocks == 0 or block_size == 0:
        return 0
    return num_gpu_blocks * block_size


def _kv_cache_tokens_from_cache_config(cache_config: Any) -> int:
    if isinstance(cache_config, list):
        return sum(_kv_cache_tokens_from_cache_config(config) for config in cache_config if isinstance(config, dict))
    if isinstance(cache_config, dict):
        return _kv_cache_tokens_from_blocks(cache_config)
    return 0


def _kv_cache_tokens_from_nested_info(value: Any) -> int:
    candidates = []
    for config in _iter_dicts(value):
        candidates.append(_kv_cache_tokens_from_blocks(config))
    return max(candidates, default=0)


def _gpu_memory_utilization_from_info(value: Any, default: float = 0.9) -> float:
    for config in _iter_dicts(value):
        if "gpu_memory_utilization" not in config:
            continue
        try:
            return float(config["gpu_memory_utilization"])
        except (TypeError, ValueError):
            return default
    return default


def _json_safe(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value))
    except (TypeError, ValueError):
        return json.loads(json.dumps(value, default=str))


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * percentile / 100.0
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return sorted_values[lower]
    lower_weight = upper - rank
    upper_weight = rank - lower
    return sorted_values[lower] * lower_weight + sorted_values[upper] * upper_weight


def _numeric_summary(values: list[int | float]) -> dict[str, int | float]:
    if not values:
        return {"count": 0, "avg": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0, "max": 0}
    return {
        "count": len(values),
        "avg": statistics.mean(values),
        "p10": _percentile(values, 10),
        "p50": statistics.median(values),
        "p90": _percentile(values, 90),
        "max": max(values),
    }


class BenchmarkQualityError(RuntimeError):
    """Raised when a benchmark run produced a low-confidence decode window."""

    def __init__(self, message: str, result: dict[str, Any]):
        super().__init__(message)
        self.result = result


def _count_chunk_tokens_from_token_ids(choice: dict[str, Any], chunk_index: int) -> int | None:
    """Return emitted token count from token_ids.

    Empty finish chunks may omit token_ids and are ignored. Non-empty chunks must
    carry token_ids so speculative multi-token chunks are counted exactly.
    """
    if "token_ids" not in choice:
        if not choice.get("text"):
            return None
        raise RuntimeError(
            "Streaming benchmark requires token_ids for every non-empty "
            f"response chunk; missing token_ids at chunk_index={chunk_index}. "
            "Set return_token_ids=true on the completions request."
        )

    token_ids = choice["token_ids"]
    if not isinstance(token_ids, list):
        raise RuntimeError(
            "Streaming benchmark requires token_ids to be a list; "
            f"got {type(token_ids).__name__} at chunk_index={chunk_index}."
        )

    if not token_ids:
        if not choice.get("text"):
            return None
        raise RuntimeError(
            "Streaming benchmark received a non-empty response chunk with empty "
            f"token_ids at chunk_index={chunk_index}."
        )
    return len(token_ids)


@dataclass
class TpsMetric:
    """Metrics for recording TPS test results"""

    batch_size: int
    input_len: int
    output_len: int
    prefill_tps: float = 0.0  # TPS during prefill phase
    decode_tps_e2e_window: float = 0.0
    # ^ E2E serve decode throughput inside the stable decode window.
    prefill_latency: float = 0.0  # Prefill latency in seconds
    decode_latency: float = 0.0
    # ^ Mean client-observed ITL in the decode window, in seconds.
    decode_window_start: float = 0.0
    decode_window_end: float = 0.0
    decode_window_duration: float = 0.0
    decode_window_tokens: int = 0
    decode_window_chunks: int = 0
    decode_window_token_coverage: float = 0.0
    decode_window_itl_count: int = 0
    itl_avg: float = 0.0
    itl_p50: float = 0.0
    itl_p90: float = 0.0
    itl_p99: float = 0.0
    response_chunk_tokens_avg: float = 0.0
    response_chunk_tokens_p10: float = 0.0
    response_chunk_tokens_p50: float = 0.0
    response_chunk_tokens_p90: float = 0.0
    response_chunk_tokens_max: int = 0
    total_latency: float = 0.0  # Total latency in seconds
    prefill_flops: float = 0.0  # Prefill FLOPs in TFLOPs
    decode_flops: float = 0.0  # Decode FLOPs in TFLOPs
    prefill_mfu: float = 0.0  # Prefill Model FLOPs Utilization
    decode_mfu: float = 0.0
    max_connections: int = 1024  # HTTP connection pool size used
    complete_requests: int = 0  # Number of requests that reached max_tokens
    incomplete_requests: int = 0  # Number of requests that didn't reach max_tokens
    total_completion_tokens: int = 0
    decode_window_quality: dict[str, Any] = field(default_factory=dict)
    request_logprobs: int | None = None  # Request payload logprobs value
    return_token_ids: bool = False  # Whether return_token_ids is requested
    weight_version: int | None = None  # Requested server-side weight version
    status: str = "completed"
    skip_reason: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class VLLMBenchmarkClient:
    """HTTP client for testing vLLM service performance"""

    def __init__(
        self,
        host: str,
        port: int,
        model_name: str,
        max_connections: int = 1024,
        request_logprobs: int | None = None,
        return_token_ids: bool = False,
        weight_version: int | None = None,
        request_total_timeout: int = 900,
        request_sock_connect_timeout: int = 30,
        request_sock_read_timeout: int = 300,
        raw_artifacts_dir: str | None = None,
    ):
        self.raw_artifacts_dir = Path(raw_artifacts_dir) if raw_artifacts_dir else None
        self.batch_sequence = 0
        self.artifact_context = "unassigned"
        self.base_url = f"http://{host}:{port}"
        self.model_name = model_name
        self.max_connections = max_connections
        self.request_logprobs = request_logprobs
        # Token counting is exact only with token_ids. Keep accepting the
        # constructor flag for CLI/config compatibility, but always request IDs.
        if not return_token_ids:
            return_token_ids = True
        self.return_token_ids = return_token_ids
        self.weight_version = weight_version
        self.request_total_timeout = request_total_timeout
        self.request_sock_connect_timeout = request_sock_connect_timeout
        self.request_sock_read_timeout = request_sock_read_timeout
        self.session: aiohttp.ClientSession | None = None
        self.weight_control_session: aiohttp.ClientSession | None = None

    async def __aenter__(self):
        # 创建连接器，设置更高的连接池限制
        benchmark_connector = aiohttp.TCPConnector(
            limit=self.max_connections,  # 总连接数限制
            limit_per_host=self.max_connections,  # 每个主机的连接数限制
            enable_cleanup_closed=True,  # 启用清理已关闭的连接
            # 不复用连接：保证 asyncio.gather 真并发，避免连接池排队污染 max(ttft)
            force_close=True,
            ttl_dns_cache=300,  # DNS缓存5分钟
        )

        # Keep weight-version control requests on one connection so that
        # SO_REUSEPORT routes GET -> POST -> GET to the same API process.
        weight_control_connector = aiohttp.TCPConnector(
            limit=1,
            limit_per_host=1,
            enable_cleanup_closed=True,
            force_close=False,
            ttl_dns_cache=300,
        )

        # 设置超时配置
        timeout = aiohttp.ClientTimeout(
            total=self.request_total_timeout,
            sock_connect=self.request_sock_connect_timeout,
            sock_read=self.request_sock_read_timeout,
        )

        self.session = aiohttp.ClientSession(
            connector=benchmark_connector,
            timeout=timeout,
        )
        self.weight_control_session = aiohttp.ClientSession(
            connector=weight_control_connector,
            timeout=timeout,
        )
        return self

    async def __aexit__(self, _exc_type, _exc_val, _exc_tb):
        if self.weight_control_session:
            await self.weight_control_session.close()
        if self.session:
            await self.session.close()

    async def health_check(self) -> bool:
        """Check if the service is running properly"""
        for _ in range(3):
            try:
                async with self.session.get(f"{self.base_url}/health") as response:
                    print(f"Health check response: {response.status}")
                    if response.status == 200:
                        return True
                    else:
                        await asyncio.sleep(10)
            except Exception as e:
                print(f"Health check failed: {e}")
                await asyncio.sleep(10)
        return False

    async def get_server_info(self) -> dict:
        """Save read-only probe evidence; unknown schema is not a capacity source."""
        async with self.session.get(f"{self.base_url}/server_info", params={"config_format": "json"}) as response:
            body = await response.text()
            record = {"status": response.status, "body": body, "headers": dict(response.headers)}
            if self.raw_artifacts_dir:
                (self.raw_artifacts_dir / "server_info.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
            if response.status != 200:
                return {}
            try:
                value = json.loads(body)
            except json.JSONDecodeError:
                return {}
            return value if isinstance(value, dict) else {}

    async def set_weight_version(self, weight_version: int) -> dict:
        """Set server-side weight_version through the dev-mode endpoint."""
        payload = {"weight_version": weight_version}
        async with self.weight_control_session.post(
            f"{self.base_url}/set_model_weight",
            json=payload,
            headers={"Content-Type": "application/json"},
        ) as response:
            response_text = await response.text()
            if response.status != 200:
                raise Exception(f"Failed to set weight_version: {response.status} - {response_text}")
            try:
                return json.loads(response_text)
            except json.JSONDecodeError:
                return {"raw_response": response_text}

    async def get_weight_version(self) -> dict:
        """Get server-side weight_version through the dev-mode endpoint."""
        try:
            async with self.weight_control_session.get(f"{self.base_url}/weight_version") as response:
                return await response.json()
        except Exception:
            return {}

    async def get_active_requests(self, timeout_s: float = 10.0) -> int | None:
        """Get the current server-side unfinished request count."""
        try:
            timeout = aiohttp.ClientTimeout(total=timeout_s)
            async with self.session.get(f"{self.base_url}/active_requests", timeout=timeout) as response:
                response_text = await response.text()
                if response.status != 200:
                    print(f"Warning: failed to query active requests: {response.status} - {response_text[:200]}")
                    return None
                payload = json.loads(response_text)
                active_requests = payload.get("active_requests")
                if active_requests is None:
                    print(f"Warning: /active_requests response missing active_requests: {payload}")
                    return None
                return int(active_requests)
        except Exception as e:
            print(f"Warning: failed to query active requests: {e}")
            return None

    async def abort_inflight_requests(self, timeout_s: float = 30.0, resume_retries: int = 3) -> bool:
        """Abort in-flight generation requests through the dev pause endpoint."""
        timeout = aiohttp.ClientTimeout(total=timeout_s)
        try:
            async with self.session.post(
                f"{self.base_url}/pause",
                params={"mode": "abort", "clear_cache": "true"},
                timeout=timeout,
            ) as response:
                response_text = await response.text()
                if response.status != 200:
                    print(
                        "Warning: failed to abort in-flight requests via /pause: "
                        f"{response.status} - {response_text[:200]}"
                    )
                    return False
                print("  Aborted in-flight requests via /pause?mode=abort")
        except Exception as e:
            print(f"Warning: failed to abort in-flight requests via /pause: {e}")
            return False

        last_resume_error = "unknown error"
        for attempt_idx in range(1, resume_retries + 1):
            try:
                async with self.session.post(f"{self.base_url}/resume", timeout=timeout) as response:
                    response_text = await response.text()
                    if response.status == 200:
                        print("  Resumed generation after cleanup")
                        return True
                    last_resume_error = f"{response.status} - {response_text[:200]}"
            except Exception as e:
                last_resume_error = str(e)

            if attempt_idx < resume_retries:
                print(
                    "Warning: failed to resume generation after cleanup "
                    f"(attempt {attempt_idx}/{resume_retries}): {last_resume_error}"
                )
                await asyncio.sleep(min(5, attempt_idx))

        raise RuntimeError(f"Server cleanup paused generation but failed to resume it: {last_resume_error}")

    async def wait_for_no_active_requests(
        self,
        timeout_s: float,
        poll_interval_s: float = 5.0,
    ) -> bool:
        """Best-effort wait until the server reports no unfinished requests."""
        if timeout_s <= 0:
            active_requests = await self.get_active_requests()
            if active_requests is None:
                return False
            if active_requests <= 0:
                print("  Server cleanup complete: active_requests=0")
                return True
            print(f"Warning: server cleanup timed out with active_requests={active_requests}")
            return False

        deadline = time.monotonic() + timeout_s
        last_report = 0.0
        last_active_requests = None
        while True:
            active_requests = await self.get_active_requests()
            now = time.monotonic()
            if active_requests is None:
                return False
            last_active_requests = active_requests
            if active_requests <= 0:
                print("  Server cleanup complete: active_requests=0")
                return True
            if now >= deadline:
                print(f"Warning: server cleanup timed out with active_requests={last_active_requests}")
                return False
            if last_report == 0.0 or now - last_report >= 30.0:
                print(f"  Waiting for server cleanup: active_requests={active_requests}")
                last_report = now
            await asyncio.sleep(min(poll_interval_s, max(0.1, deadline - now)))

    async def generate_completion_stream(self, prompt_tokens, max_tokens, temperature=0.0, request_id=None):
        artifact_id = f"{self.artifact_context}-{request_id}-{uuid.uuid4().hex}"
        record = {
            "artifact_id": artifact_id,
            "context": self.artifact_context,
            "wall_time_unix": time.time(),
            "reads": [],
            "raw_parts": [],
            "status": "started",
        }
        try:
            result = await self._generate_completion_stream(prompt_tokens, max_tokens, temperature, request_id, record)
            record.update(status="completed", result=result)
            return result
        except BaseException as error:
            record.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            raw = b"".join(record.pop("raw_parts"))
            if self.raw_artifacts_dir:
                (self.raw_artifacts_dir / f"{artifact_id}.sse").write_bytes(raw)
                (self.raw_artifacts_dir / f"{artifact_id}.json").write_text(
                    json.dumps(record, indent=2), encoding="utf-8"
                )

    async def _generate_completion_stream(
        self,
        prompt_tokens: list[int],
        max_tokens: int,
        temperature: float = 0.0,
        request_id: str = None,
        record: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Send streaming completion request and return detailed timing data"""
        payload = {
            "model": self.model_name,
            "prompt": prompt_tokens,
            "max_tokens": max_tokens,
            "min_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
            "ignore_eos": True,
        }
        if self.request_logprobs is not None:
            payload["logprobs"] = self.request_logprobs
        payload["return_token_ids"] = True

        record["payload"] = payload
        record["prompt_sha256"] = hashlib.sha256(json.dumps(prompt_tokens, separators=(",", ":")).encode()).hexdigest()
        chunks = []
        finishes = []
        usages = []
        weight_versions = set()
        request_start_time = time.perf_counter()
        record["request_start_timestamp"] = request_start_time

        async with self.session.post(
            f"{self.base_url}/v1/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
        ) as response:
            record.update(http_status=response.status, http_headers=dict(response.headers))
            if response.status != 200:
                body = await response.read()
                record["raw_parts"].append(body)
                raise RuntimeError(f"Request failed: {response.status} - {body.decode(errors='replace')}")
            buffer = ""
            decoder = codecs.getincrementaldecoder("utf-8")()
            done = False
            byte_offset = 0
            async for data in response.content.iter_chunked(4096):
                record["reads"].append(
                    {
                        "read_index": len(record["reads"]),
                        "timestamp": time.perf_counter(),
                        "offset": byte_offset,
                        "length": len(data),
                    }
                )
                byte_offset += len(data)
                record["raw_parts"].append(data)
                buffer += decoder.decode(data)
                buffer = buffer.replace("\r\n", "\n")
                while "\n\n" in buffer:
                    event, buffer = buffer.split("\n\n", 1)
                    event_data_lines = []
                    for line in event.splitlines():
                        line = line.strip()
                        if not line.startswith("data:"):
                            if line.startswith("event:") and line[6:].strip() == "error":
                                raise RuntimeError("SSE error event")
                            continue
                        data_str = line[5:].lstrip(" ")
                        if data_str == "[DONE]":
                            done = True
                            break
                        event_data_lines.append(data_str)
                    if done:
                        break
                    if not event_data_lines:
                        continue
                    try:
                        chunk = json.loads("\n".join(event_data_lines))
                    except json.JSONDecodeError as error:
                        raise RuntimeError("Malformed SSE JSON") from error
                    if not isinstance(chunk, dict) or chunk.get("error"):
                        raise RuntimeError(f"SSE error payload: {chunk}")
                    if chunk.get("usage") is not None:
                        usages.append(chunk["usage"])
                    for choice in chunk.get("choices", []):
                        if choice.get("finish_reason") is not None:
                            finishes.append({"reason": choice["finish_reason"], "timestamp": time.perf_counter()})
                    record.update(chunks=chunks, finishes=finishes, usages=usages)
                    chunk_weight_version = chunk.get("weight_version")
                    if isinstance(chunk_weight_version, list):
                        weight_versions.update(v for v in chunk_weight_version if v is not None)
                    elif chunk_weight_version is not None:
                        weight_versions.add(chunk_weight_version)
                    if chunk.get("choices"):
                        absolute_timestamp = time.perf_counter()
                        chunk["absolute_timestamp"] = absolute_timestamp
                        chunk["relative_timestamp"] = absolute_timestamp - request_start_time
                        chunks.append(chunk)
                if done:
                    break

        record.update(done=done, trailing_buffer=buffer)
        if not done:
            raise RuntimeError("SSE stream ended without [DONE]")
        if not finishes or any(item["reason"] != "length" for item in finishes):
            raise RuntimeError(f"Expected finish_reason=length, got {finishes}")

        # Process chunks to extract timing data
        token_data = []
        ttft = None
        prev_absolute_time = request_start_time
        first_token_timestamp = 0.0
        completion_tokens = 0

        for i, chunk in enumerate(chunks):
            if "choices" in chunk and chunk["choices"]:
                choice = chunk["choices"][0]
                chunk_tokens = _count_chunk_tokens_from_token_ids(choice, i)
                if chunk_tokens is None:
                    continue

                absolute_time = chunk["absolute_timestamp"]

                if ttft is None:
                    # First token - calculate TTFT
                    ttft = absolute_time - request_start_time
                    first_token_timestamp = absolute_time
                    itl = ttft  # First token ITL is same as TTFT
                else:
                    # Subsequent tokens - calculate ITL
                    itl = absolute_time - prev_absolute_time

                completion_tokens += chunk_tokens

                token_data.append(
                    {
                        "token_index": i,
                        "absolute_timestamp": absolute_time,
                        "itl": itl,
                        "is_first_token": ttft is not None and len(token_data) == 0,
                        "chunk_tokens": chunk_tokens,
                    }
                )

                prev_absolute_time = absolute_time

        record["token_data"] = token_data
        if completion_tokens != max_tokens:
            raise RuntimeError(f"Exact output length required: {completion_tokens} != {max_tokens}")
        for usage in usages:
            if not isinstance(usage, dict):
                raise RuntimeError("Malformed usage payload")
            if usage.get("completion_tokens") is not None and usage["completion_tokens"] != completion_tokens:
                raise RuntimeError("usage.completion_tokens disagrees with token_ids count")
        return {
            "request_id": request_id,
            "input_tokens": len(prompt_tokens),
            "output_tokens": len(token_data),
            "completion_tokens": completion_tokens,
            "expected_tokens": max_tokens,
            "is_complete": completion_tokens == max_tokens,
            "ttft": ttft if ttft is not None else 0.0,
            "request_start_timestamp": request_start_time,
            "first_token_timestamp": first_token_timestamp,
            "last_token_timestamp": token_data[-1]["absolute_timestamp"] if token_data else 0.0,
            "finish_events": finishes,
            "usage": usages,
            "done": done,
            "token_data": token_data,
            "total_latency": chunks[-1]["absolute_timestamp"] - request_start_time if chunks else 0.0,
            "weight_versions": sorted(weight_versions),
        }

    async def test_streaming_benchmark(
        self,
        batch_prompts: list[list[int]],
        output_len: int,
        validate_decode_quality: bool = True,
    ) -> dict[str, Any]:
        """Test streaming benchmark to get prefill and decode metrics.

        Uses precise timing analysis for accurate separation.

        Args:
            batch_prompts: List of token sequences for each request
            output_len: Maximum output length
            validate_decode_quality: Whether to reject low-confidence
                decode windows. Warmup calls should disable this.
        """
        self.batch_sequence += 1
        self.artifact_context = (
            f"batch{self.batch_sequence:04d}-bs{len(batch_prompts)}"
            f"-input{len(batch_prompts[0])}-output{output_len}"
            f"-{'measured' if validate_decode_quality else 'warmup'}"
        )
        tasks = []
        batch_size = len(batch_prompts)

        # Start all requests concurrently
        for i, prompt_tokens in enumerate(batch_prompts):
            task = asyncio.create_task(
                self.generate_completion_stream(prompt_tokens, max_tokens=output_len, request_id=f"req_{i}")
            )
            tasks.append(task)

        # Wait for all responses
        try:
            all_results = await asyncio.gather(*tasks)
        except Exception:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

        # Extract timing data and check completeness
        ttft_times = []  # Time to First Token for each request
        all_token_data = []  # All token timing data across requests
        first_token_timestamps = []
        complete_last_token_timestamps = []
        total_input_tokens = 0
        total_output_tokens = 0
        complete_requests = 0
        incomplete_requests = []
        observed_weight_versions = set()

        for result in all_results:
            if result["ttft"] > 0:
                ttft_times.append(result["ttft"])
            request_token_data = result.get("token_data", [])
            first_token_timestamp = result.get("first_token_timestamp") or (
                request_token_data[0]["absolute_timestamp"] if request_token_data else 0.0
            )
            last_token_timestamp = result.get("last_token_timestamp") or (
                request_token_data[-1]["absolute_timestamp"] if request_token_data else 0.0
            )
            if first_token_timestamp > 0:
                first_token_timestamps.append(first_token_timestamp)
            total_input_tokens += result["input_tokens"]
            total_output_tokens += result["output_tokens"]
            observed_weight_versions.update(result.get("weight_versions", []))

            # Track request completion
            if result["is_complete"]:
                complete_requests += 1
                if last_token_timestamp > 0:
                    complete_last_token_timestamps.append(last_token_timestamp)
            else:
                incomplete_requests.append(
                    {
                        "request_id": result["request_id"],
                        "expected": result["expected_tokens"],
                        "actual": result["completion_tokens"],
                    }
                )

            # Collect all token data with request info
            for token_info in result["token_data"]:
                token_info["request_id"] = result["request_id"]
                all_token_data.append(token_info)

        # Report incomplete requests
        if incomplete_requests:
            print(f"  {len(incomplete_requests)}/{batch_size} requests incomplete")

        # Calculate prefill metrics
        last_ttft = max(ttft_times) if ttft_times else 0
        prefill_tps = total_input_tokens / last_ttft if last_ttft > 0 else 0

        total_wall_time = max(r["total_latency"] for r in all_results) if all_results else 0

        total_completion_tokens = sum(r.get("completion_tokens", 0) for r in all_results)
        non_first_tokens = [t for t in all_token_data if not t["is_first_token"]]
        total_decode_candidate_tokens = sum(max(0, int(t.get("chunk_tokens", 1))) for t in non_first_tokens)
        response_chunk_token_counts = [
            int(token_info.get("chunk_tokens", 0))
            for result in all_results
            for token_info in result.get("token_data", [])
            if int(token_info.get("chunk_tokens", 0)) > 0
        ]
        response_chunk_tokens_avg = statistics.mean(response_chunk_token_counts) if response_chunk_token_counts else 0.0
        response_chunk_tokens_p10 = _percentile(response_chunk_token_counts, 10)
        response_chunk_tokens_p50 = (
            statistics.median(response_chunk_token_counts) if response_chunk_token_counts else 0.0
        )
        response_chunk_tokens_p90 = _percentile(response_chunk_token_counts, 90)
        response_chunk_tokens_max = max(response_chunk_token_counts) if response_chunk_token_counts else 0

        decode_window_start = max(first_token_timestamps) if first_token_timestamps else 0.0
        decode_window_end = min(complete_last_token_timestamps) if complete_last_token_timestamps else 0.0
        decode_window_duration = max(0.0, decode_window_end - decode_window_start)

        tokens_by_request: dict[str, list[dict[str, Any]]] = {}
        for token in all_token_data:
            tokens_by_request.setdefault(token["request_id"], []).append(token)

        decode_window_tokens = 0
        decode_window_chunks = 0
        decode_window_itls = []
        for request_tokens in tokens_by_request.values():
            request_tokens.sort(key=lambda x: x["absolute_timestamp"])
            prev_token = None
            for token in request_tokens:
                ts = token["absolute_timestamp"]
                chunk_tokens = max(0, int(token.get("chunk_tokens", 1)))
                in_decode_window = (
                    not token["is_first_token"] and chunk_tokens > 0 and decode_window_start < ts <= decode_window_end
                )
                if in_decode_window:
                    decode_window_tokens += chunk_tokens
                    decode_window_chunks += 1

                if (
                    prev_token is not None
                    and in_decode_window
                    and prev_token["absolute_timestamp"] >= decode_window_start
                    and ts > prev_token["absolute_timestamp"]
                ):
                    decode_window_itls.append(ts - prev_token["absolute_timestamp"])
                prev_token = token

        decode_tps_e2e_window = decode_window_tokens / decode_window_duration if decode_window_duration > 0 else 0.0
        itl_avg = statistics.mean(decode_window_itls) if decode_window_itls else 0.0
        itl_p50 = statistics.median(decode_window_itls) if decode_window_itls else 0.0
        itl_p90 = _percentile(decode_window_itls, 90)
        itl_p99 = _percentile(decode_window_itls, 99)
        decode_latency = itl_avg
        decode_window_token_coverage = (
            decode_window_tokens / total_decode_candidate_tokens if total_decode_candidate_tokens > 0 else 0.0
        )
        min_decode_window_token_coverage = 0.25
        decode_quality_validated = validate_decode_quality and output_len >= 64
        weight_version_validated = self.weight_version is not None
        decode_quality_issues = []
        if decode_quality_validated:
            if incomplete_requests:
                decode_quality_issues.append(f"{len(incomplete_requests)}/{batch_size} requests incomplete")
            if len(first_token_timestamps) < batch_size:
                decode_quality_issues.append(
                    f"first-token timestamps missing: {len(first_token_timestamps)}/{batch_size}"
                )
            if not complete_last_token_timestamps:
                decode_quality_issues.append("no complete request end timestamp")
            if decode_window_duration <= 0:
                decode_quality_issues.append(f"decode window duration is non-positive: {decode_window_duration:.6f}s")
            if decode_window_tokens <= 0:
                decode_quality_issues.append("no decode tokens inside E2E window")
            if not decode_window_itls:
                decode_quality_issues.append("no ITL samples inside E2E window")
            if decode_window_token_coverage < min_decode_window_token_coverage:
                decode_quality_issues.append(
                    "decode window token coverage too low: "
                    f"{decode_window_token_coverage:.3f} "
                    f"< {min_decode_window_token_coverage:.3f}"
                )
            if decode_tps_e2e_window <= 0:
                decode_quality_issues.append(f"decode_tps_e2e_window is non-positive: {decode_tps_e2e_window:.2f}")
        if weight_version_validated:
            expected_weight_version = self.weight_version
            if observed_weight_versions != {expected_weight_version}:
                decode_quality_issues.append(
                    "observed weight_versions do not match requested "
                    f"weight_version={expected_weight_version}: "
                    f"{sorted(observed_weight_versions)}"
                )
        decode_window_quality = {
            "validated": decode_quality_validated or weight_version_validated,
            "valid": not decode_quality_issues,
            "issues": decode_quality_issues,
            "min_decode_window_token_coverage": min_decode_window_token_coverage,
            "decode_window_start": decode_window_start,
            "decode_window_end": decode_window_end,
            "decode_window_duration": decode_window_duration,
            "decode_window_tokens": decode_window_tokens,
            "decode_window_chunks": decode_window_chunks,
            "decode_window_itl_count": len(decode_window_itls),
            "decode_window_token_coverage": decode_window_token_coverage,
            "total_decode_candidate_tokens": total_decode_candidate_tokens,
        }

        print(
            f"  === Diagnostics (batch_size={batch_size}, "
            f"input_len={len(batch_prompts[0]) if batch_prompts else 0}) ==="
        )
        print(
            f"  decode_tps_e2e_window={decode_tps_e2e_window:.2f} tok/s "
            f"(window={decode_window_duration:.4f}s, "
            f"tokens={decode_window_tokens}, "
            f"coverage={decode_window_token_coverage:.3f})"
        )
        print(
            f"  ITL: avg={itl_avg * 1000:.2f}ms, "
            f"p50={itl_p50 * 1000:.2f}ms, "
            f"p90={itl_p90 * 1000:.2f}ms, "
            f"p99={itl_p99 * 1000:.2f}ms, "
            f"samples={len(decode_window_itls)}"
        )
        print(f"  Token count: total={total_completion_tokens}, decode_candidates={total_decode_candidate_tokens}")
        print(
            f"  Response chunk tokens: avg={response_chunk_tokens_avg:.2f}, "
            f"p50={response_chunk_tokens_p50:.2f}, "
            f"p90={response_chunk_tokens_p90:.2f}, "
            f"max={response_chunk_tokens_max}"
        )
        print(
            "  Decode window quality: "
            f"valid={decode_window_quality['valid']}, "
            f"token_coverage={decode_window_token_coverage:.3f}, "
            f"duration={decode_window_duration:.4f}s, "
            f"issues={decode_quality_issues}"
        )

        result_payload = {
            "prefill_tps": prefill_tps,
            "decode_tps_e2e_window": decode_tps_e2e_window,
            "prefill_latency": last_ttft,
            "decode_latency": decode_latency,
            "decode_window_start": decode_window_start,
            "decode_window_end": decode_window_end,
            "decode_window_duration": decode_window_duration,
            "decode_window_tokens": decode_window_tokens,
            "decode_window_chunks": decode_window_chunks,
            "decode_window_token_coverage": decode_window_token_coverage,
            "decode_window_itl_count": len(decode_window_itls),
            "itl_avg": itl_avg,
            "itl_p50": itl_p50,
            "itl_p90": itl_p90,
            "itl_p99": itl_p99,
            "response_chunk_tokens_avg": response_chunk_tokens_avg,
            "response_chunk_tokens_p10": response_chunk_tokens_p10,
            "response_chunk_tokens_p50": response_chunk_tokens_p50,
            "response_chunk_tokens_p90": response_chunk_tokens_p90,
            "response_chunk_tokens_max": response_chunk_tokens_max,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_latency": total_wall_time,
            "complete_requests": complete_requests,
            "incomplete_requests": len(incomplete_requests),
            "total_completion_tokens": total_completion_tokens,
            "decode_window_quality": decode_window_quality,
        }
        if self.raw_artifacts_dir:
            (self.raw_artifacts_dir / f"{self.artifact_context}-iteration.json").write_text(
                json.dumps(
                    {
                        "metrics": result_payload,
                        "requests": all_results,
                        "semantics": "client E2E overlap; ITL is SSE event interval, not token ITL",
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        if decode_quality_issues:
            raise BenchmarkQualityError(
                "benchmark quality check failed: " + "; ".join(decode_quality_issues),
                result_payload,
            )
        return result_payload


def generate_random_tokens(batch_size: int, input_len: int, seed: int = 42, vocab_size: int = 32000) -> list[list[int]]:
    """Generate test prompts with random token IDs

    Args:
        batch_size: Batch size
        input_len: Input length
        seed: Random seed
        vocab_size: Vocabulary size for token generation

    Returns:
        List of prompt token lists
    """
    batch_prompts = []

    for i in range(batch_size):
        # Use different seed for each request
        random.seed(seed + i)
        tokens = [random.randint(1, vocab_size - 1) for _ in range(input_len)]
        batch_prompts.append(tokens)

    # Removed verbose logging

    return batch_prompts


async def run_single_benchmark(
    client: VLLMBenchmarkClient,
    batch_size: int,
    input_len: int,
    output_len: int,
    num_iters: int = 3,
    seed: int = 42,
    vocab_size: int = 32000,
    max_connections: int = 1024,
    transformer_config: dict[str, Any] | None = None,
    prefill_total_gpu_flops: float = 0.0,
    decode_total_gpu_flops: float = 0.0,
    stats_last_iters: int | None = None,
) -> TpsMetric:
    """Run benchmark for a single configuration using streaming API"""

    if num_iters <= 0:
        raise ValueError("num_iters must be positive")
    if stats_last_iters is None:
        effective_stats_iters = num_iters
    elif stats_last_iters <= 0 or stats_last_iters > num_iters:
        raise ValueError(
            "stats_last_iters must be positive and no greater than num_iters; "
            f"got stats_last_iters={stats_last_iters}, num_iters={num_iters}"
        )
    else:
        effective_stats_iters = stats_last_iters

    print(
        f"Test configuration: batch_size={batch_size}, input_len={input_len}, max_tokens={output_len} (auto-calculated)"
    )

    prefill_tps_results = []
    decode_tps_e2e_window_results = []
    total_latency_results = []
    prefill_latency_results = []
    decode_latency_results = []
    itl_p50_results = []
    itl_p90_results = []
    itl_p99_results = []

    # Run multiple iterations
    last_iteration_results = None
    for iter_idx in range(num_iters):
        print(f"  Iteration {iter_idx + 1}/{num_iters}")

        # Generate test prompts (regenerate for each iteration to ensure correctness)
        batch_prompts = generate_random_tokens(batch_size, input_len, seed + iter_idx, vocab_size)

        # Run streaming benchmark
        results = await client.test_streaming_benchmark(batch_prompts, output_len)
        last_iteration_results = results  # Keep track of last iteration

        print(
            f"  Prefill TPS: {results['prefill_tps']:.2f}, "
            f"Decode TPS E2E Window: {results['decode_tps_e2e_window']:.2f}"
        )

        prefill_tps_results.append(results["prefill_tps"])
        decode_tps_e2e_window_results.append(results["decode_tps_e2e_window"])
        total_latency_results.append(results["total_latency"])
        prefill_latency_results.append(results["prefill_latency"])
        decode_latency_results.append(results["decode_latency"])
        itl_p50_results.append(results["itl_p50"])
        itl_p90_results.append(results["itl_p90"])
        itl_p99_results.append(results["itl_p99"])
        # Wait between iterations
        await asyncio.sleep(3.0)

    if stats_last_iters is not None:
        stats_start_iter = num_iters - effective_stats_iters + 1
        print(f"  Aggregating iterations {stats_start_iter}-{num_iters} ({effective_stats_iters}/{num_iters})")
    prefill_tps_results = prefill_tps_results[-effective_stats_iters:]
    decode_tps_e2e_window_results = decode_tps_e2e_window_results[-effective_stats_iters:]
    total_latency_results = total_latency_results[-effective_stats_iters:]
    prefill_latency_results = prefill_latency_results[-effective_stats_iters:]
    decode_latency_results = decode_latency_results[-effective_stats_iters:]
    itl_p50_results = itl_p50_results[-effective_stats_iters:]
    itl_p90_results = itl_p90_results[-effective_stats_iters:]
    itl_p99_results = itl_p99_results[-effective_stats_iters:]

    # Calculate averages from the selected trailing iterations.
    avg_prefill_tps = statistics.mean(prefill_tps_results) if prefill_tps_results else 0
    avg_decode_tps_e2e_window = statistics.mean(decode_tps_e2e_window_results) if decode_tps_e2e_window_results else 0
    avg_total_latency = statistics.mean(total_latency_results) if total_latency_results else 0
    if stats_last_iters is None:
        # Preserve the original benchmark behavior unless trailing aggregation
        # is explicitly requested.
        avg_prefill_latency = last_iteration_results["prefill_latency"] if last_iteration_results else 0
    else:
        avg_prefill_latency = statistics.mean(prefill_latency_results) if prefill_latency_results else 0
    avg_decode_latency = statistics.mean(decode_latency_results) if decode_latency_results else 0
    avg_itl_p50 = statistics.mean(itl_p50_results) if itl_p50_results else 0
    avg_itl_p90 = statistics.mean(itl_p90_results) if itl_p90_results else 0
    avg_itl_p99 = statistics.mean(itl_p99_results) if itl_p99_results else 0
    # Calculate FLOPS
    prefill_flops = 0.0
    decode_flops = 0.0
    if transformer_config and sequence_flops:
        try:
            # Calculate prefill FLOPS (TFLOPs)
            prefill_flops = batch_size * sequence_flops(transformer_config, input_len, None, "prefill")
            # Calculate decode FLOPS (TFLOPs)
            decode_flops = batch_size * sequence_flops(transformer_config, input_len, 1, "decode")
        except Exception as e:
            print(f"Warning: Failed to calculate FLOPs: {e}")

    # Calculate MFU
    prefill_mfu = 0.0
    decode_mfu = 0.0
    if prefill_total_gpu_flops > 0 and avg_prefill_latency > 0:
        # Prefill MFU uses the prefill-side GPU denominator.
        prefill_mfu = (prefill_flops / avg_prefill_latency) / prefill_total_gpu_flops

    if decode_total_gpu_flops > 0 and avg_decode_latency > 0:
        # Decode MFU uses the decode-side GPU denominator.
        decode_mfu = (decode_flops / avg_decode_latency) / decode_total_gpu_flops

    return TpsMetric(
        batch_size=batch_size,
        input_len=input_len,
        output_len=output_len,
        prefill_tps=avg_prefill_tps,
        decode_tps_e2e_window=avg_decode_tps_e2e_window,
        prefill_latency=avg_prefill_latency,
        decode_latency=avg_decode_latency,
        decode_window_start=last_iteration_results.get("decode_window_start", 0.0) if last_iteration_results else 0.0,
        decode_window_end=last_iteration_results.get("decode_window_end", 0.0) if last_iteration_results else 0.0,
        decode_window_duration=last_iteration_results.get("decode_window_duration", 0.0)
        if last_iteration_results
        else 0.0,
        decode_window_tokens=last_iteration_results.get("decode_window_tokens", 0) if last_iteration_results else 0,
        decode_window_chunks=last_iteration_results.get("decode_window_chunks", 0) if last_iteration_results else 0,
        decode_window_token_coverage=last_iteration_results.get("decode_window_token_coverage", 0.0)
        if last_iteration_results
        else 0.0,
        decode_window_itl_count=last_iteration_results.get("decode_window_itl_count", 0)
        if last_iteration_results
        else 0,
        itl_avg=avg_decode_latency,
        itl_p50=avg_itl_p50,
        itl_p90=avg_itl_p90,
        itl_p99=avg_itl_p99,
        response_chunk_tokens_avg=last_iteration_results.get("response_chunk_tokens_avg", 0.0)
        if last_iteration_results
        else 0.0,
        response_chunk_tokens_p10=last_iteration_results.get("response_chunk_tokens_p10", 0.0)
        if last_iteration_results
        else 0.0,
        response_chunk_tokens_p50=last_iteration_results.get("response_chunk_tokens_p50", 0.0)
        if last_iteration_results
        else 0.0,
        response_chunk_tokens_p90=last_iteration_results.get("response_chunk_tokens_p90", 0.0)
        if last_iteration_results
        else 0.0,
        response_chunk_tokens_max=last_iteration_results.get("response_chunk_tokens_max", 0)
        if last_iteration_results
        else 0,
        total_latency=avg_total_latency,
        prefill_flops=prefill_flops,
        decode_flops=decode_flops,
        prefill_mfu=prefill_mfu,
        decode_mfu=decode_mfu,
        max_connections=max_connections,
        complete_requests=last_iteration_results.get("complete_requests", 0) if last_iteration_results else 0,
        incomplete_requests=last_iteration_results.get("incomplete_requests", 0) if last_iteration_results else 0,
        total_completion_tokens=last_iteration_results.get("total_completion_tokens", 0)
        if last_iteration_results
        else 0,
        decode_window_quality=last_iteration_results.get("decode_window_quality", {}) if last_iteration_results else {},
        request_logprobs=client.request_logprobs,
        return_token_ids=client.return_token_ids,
        weight_version=client.weight_version,
    )


async def cleanup_failed_attempt(
    client: VLLMBenchmarkClient,
    cleanup_mode: str,
    cleanup_timeout: int,
    cleanup_poll_interval: int,
) -> None:
    """Clean server-side requests left behind by a failed benchmark attempt."""
    if cleanup_mode == "off":
        return

    print(f"  Cleaning up server after failed attempt (mode={cleanup_mode})")
    # Give a multi-process API server one more cleanup pass in case the first
    # pass observes a briefly stale shared load slot.
    for pass_idx in range(2):
        if pass_idx:
            print("  Retrying server cleanup after a short delay")
            await asyncio.sleep(min(10, max(1, cleanup_poll_interval)))

        if cleanup_mode == "abort":
            aborted = await client.abort_inflight_requests(timeout_s=max(30.0, float(cleanup_timeout)))
            if not aborted:
                raise RuntimeError(
                    "Server abort cleanup is unavailable. The benchmark requires "
                    "the dev /pause endpoint for --failure-cleanup-mode=abort; "
                    "use --failure-cleanup-mode=drain or off explicitly if abort "
                    "cleanup is not available."
                )

        if await client.wait_for_no_active_requests(
            cleanup_timeout,
            cleanup_poll_interval,
        ):
            return

    raise RuntimeError(
        "Server still has active requests after cleanup; refusing to run "
        "the next benchmark attempt because it may be contaminated"
    )


async def run_benchmark_suite(
    host: str,
    port: int,
    model_name: str,
    batch_sizes: list[int],
    input_lens: list[int],
    output_lens: list[int] | None,
    max_num_batched_tokens: int,
    case_filter: set[tuple[int, int]] | None = None,
    num_iters: int = 3,
    seed: int = 42,
    vocab_size: int = 32000,
    max_connections: int = 1024,
    tp_size: int = 1,
    dp_size: int = 1,
    prefill_gpu_count: int | None = None,
    decode_gpu_count: int | None = None,
    precision: str = "bf16",
    request_logprobs: int | None = None,
    return_token_ids: bool = False,
    weight_version: int | None = None,
    spec_factor: float = 1.0,
    request_total_timeout: int = 900,
    request_sock_connect_timeout: int = 30,
    request_sock_read_timeout: int = 300,
    config_retries: int = 0,
    failure_cleanup_mode: str = "off",
    failure_cleanup_timeout: int = 300,
    failure_cleanup_poll_interval: int = 5,
    stats_last_iters: int | None = None,
    disable_mfu: bool = False,
    capacity_metadata: str | None = None,
    raw_artifacts_dir: str | None = None,
) -> dict[str, Any]:
    """Run complete benchmark suite.

    Returns results in benchmark_static_performance.py format.
    """

    if config_retries != 0 or failure_cleanup_mode != "off" or weight_version is not None:
        raise ValueError(
            "Safe copy requires --config-retries 0 --failure-cleanup-mode off and no weight-version mutation"
        )
    if not raw_artifacts_dir:
        raise ValueError("--raw-artifacts-dir is required for safe independent copy")
    artifacts = Path(raw_artifacts_dir)
    artifacts.mkdir(parents=True, exist_ok=False)
    capacity = load_capacity_metadata(capacity_metadata, dp_size)
    (artifacts / "capacity-metadata.json").write_text(json.dumps(capacity, indent=2), encoding="utf-8")
    import sys

    (artifacts / "run.json").write_text(
        json.dumps(
            {
                "argv": sys.argv,
                "wall_time_unix": time.time(),
                "seed": seed,
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "flops_script_sha256": hashlib.sha256(
                    Path(__file__).with_name("flops_calculator.py").read_bytes()
                ).hexdigest(),
                "measurement_semantics": "client E2E overlap only; no proof of pure decode or DP routing",
                "itl_semantics": "SSE event intervals; token counts from token_ids",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    failure_cleanup_mode = failure_cleanup_mode.lower()
    if failure_cleanup_mode not in {"off", "drain", "abort"}:
        raise ValueError(f"failure_cleanup_mode must be one of: off, drain, abort; got {failure_cleanup_mode!r}")
    if failure_cleanup_timeout < 0:
        raise ValueError("failure_cleanup_timeout must be non-negative")
    if failure_cleanup_poll_interval <= 0:
        raise ValueError("failure_cleanup_poll_interval must be positive")
    if num_iters <= 0:
        raise ValueError("num_iters must be positive")
    if stats_last_iters is not None and (stats_last_iters <= 0 or stats_last_iters > num_iters):
        raise ValueError(
            "stats_last_iters must be positive and no greater than num_iters; "
            f"got stats_last_iters={stats_last_iters}, num_iters={num_iters}"
        )
    if not return_token_ids:
        print("Forcing return_token_ids=true because TPS token counting requires token_ids in streaming chunks.")
        return_token_ids = True

    results = []

    # Load transformer config for FLOPS calculation
    transformer_config = None if disable_mfu else _load_transformer_config_for_flops(model_name)

    # Get GPU FLOPS for MFU calculation
    single_gpu_flops = 0.0
    total_gpu_flops = 0.0
    prefill_total_gpu_flops = 0.0
    decode_total_gpu_flops = 0.0
    try:
        if disable_mfu:
            raise RuntimeError("MFU explicitly disabled: N/A; no NVIDIA probe or peak assumption")
        from flops_calculator import get_gpu_flops

        single_gpu_flops = get_gpu_flops()
        default_gpu_count = tp_size * dp_size
        effective_prefill_gpu_count = prefill_gpu_count or default_gpu_count
        effective_decode_gpu_count = decode_gpu_count or default_gpu_count
        prefill_total_gpu_flops = effective_prefill_gpu_count * single_gpu_flops
        decode_total_gpu_flops = effective_decode_gpu_count * single_gpu_flops
        if prefill_gpu_count is None and decode_gpu_count is None:
            total_gpu_flops = prefill_total_gpu_flops
        else:
            total_gpu_flops = prefill_total_gpu_flops + decode_total_gpu_flops
        print(f"GPU FLOPS: {single_gpu_flops} TFLOPS per GPU")
        print(
            f"Prefill GPU FLOPS: {prefill_total_gpu_flops} TFLOPS total, "
            f"Decode GPU FLOPS: {decode_total_gpu_flops} TFLOPS total"
        )
    except Exception as e:
        print(f"Warning: Failed to get GPU FLOPS: {e}")

    async with VLLMBenchmarkClient(
        host,
        port,
        model_name,
        max_connections=max_connections,
        request_logprobs=request_logprobs,
        return_token_ids=return_token_ids,
        weight_version=weight_version,
        request_total_timeout=request_total_timeout,
        request_sock_connect_timeout=request_sock_connect_timeout,
        request_sock_read_timeout=request_sock_read_timeout,
        raw_artifacts_dir=raw_artifacts_dir,
    ) as client:
        # Health check
        if not await client.health_check():
            raise Exception(f"Cannot connect to service {host}:{port}")
        print(f"Successfully connected to service {host}:{port}")
        print("Using streaming API with random prompts for each request")
        if request_logprobs is not None:
            print(f"Request logprobs: {request_logprobs}")
        print(f"Return token IDs: {client.return_token_ids}")
        current_weight_info = {}
        if weight_version is not None:
            current_weight_info = await client.get_weight_version()
            current_weight_version = _extract_weight_version(current_weight_info)
            if current_weight_version == weight_version:
                print(f"Server weight_version already {weight_version}; skip /set_model_weight")
            else:
                print(f"Setting server weight_version: {weight_version} (current={current_weight_version})")
                weight_info = await client.set_weight_version(weight_version)
                print(f"Weight version endpoint response: {weight_info}")
                current_weight_info = await client.get_weight_version()
                current_weight_version = _extract_weight_version(current_weight_info)
                if current_weight_version != weight_version:
                    raise RuntimeError(
                        "Server weight_version did not update after "
                        "/set_model_weight: "
                        f"expected {weight_version}, got {current_weight_version}; "
                        f"set_response={weight_info}, current_response="
                        f"{current_weight_info}. Ensure the server enables "
                        "VLLM_ENABLE_WEIGHT_VERSION=1 for weight-version "
                        "benchmarks."
                    )

        await client.get_server_info()
        # Unsupported /server_info schema is evidence, never a guessed capacity.
        capacities = capacity["per_dp_capacity_tokens"]
        num_kv_cache = sum(capacities.values())
        kv_cache_capacity_source = capacity["source"]
        server_info_cache_config = capacity
        gpu_memory_utilization = None
        kv_capacity_unavailable_reason = None

        all_case_pairs = [(batch_size, input_len) for batch_size in batch_sizes for input_len in input_lens]
        if case_filter is not None:
            if not case_filter:
                raise ValueError("case_filter must select at least one case")
            invalid_cases = case_filter - set(all_case_pairs)
            if invalid_cases:
                raise ValueError(f"case_filter contains cases outside batch_sizes/input_lens: {sorted(invalid_cases)}")
            selected_case_pairs = [case_pair for case_pair in all_case_pairs if case_pair in case_filter]
            if not selected_case_pairs:
                raise ValueError("case_filter selected zero benchmark cases")
            print(
                "Static case filter active: "
                f"{len(selected_case_pairs)}/{len(all_case_pairs)} "
                "batch_size/input_len pairs selected"
            )
        else:
            selected_case_pairs = all_case_pairs

        # Run tests for all selected configurations
        output_len_count = len(output_lens) if output_lens else 1
        total_configs = len(selected_case_pairs) * output_len_count
        config_idx = 0

        # Preflight all explicit cases before sending any generation request.
        if output_lens:
            for batch_size, input_len in selected_case_pairs:
                for output_len in output_lens:
                    usable = [math.floor(v * KV_CACHE_GUARD_FACTOR) for v in capacities.values()]
                    sequence_len = input_len + output_len
                    if batch_size <= 0 or input_len <= 0 or output_len <= 0:
                        raise ValueError("All benchmark sizes must be positive")
                    if sequence_len > capacity["max_model_len"]:
                        raise RuntimeError("Selected case exceeds explicit max_model_len")
                    if sequence_len > max(usable) or batch_size * sequence_len > sum(usable):
                        raise RuntimeError("Selected case exceeds explicit KV capacity guard; no warmup sent")

        # Warmup - fixed prefill with input_len=1024, bs=16, max_tokens=10
        if kv_capacity_unavailable_reason is None:
            warmup_prompts = [[random.randint(1, vocab_size - 1) for _ in range(1024)] for _ in range(16)]
            warmup_attempts = config_retries + 1
            for warmup_attempt in range(1, warmup_attempts + 1):
                try:
                    await client.test_streaming_benchmark(warmup_prompts, 10, validate_decode_quality=False)
                    break
                except Exception as e:
                    print(f"Warmup attempt {warmup_attempt}/{warmup_attempts} failed: {e}")
                    if warmup_attempt >= warmup_attempts:
                        raise
                    await cleanup_failed_attempt(
                        client,
                        failure_cleanup_mode,
                        failure_cleanup_timeout,
                        failure_cleanup_poll_interval,
                    )
                    await asyncio.sleep(min(15, 5 * warmup_attempt))
        for batch_size, input_len in selected_case_pairs:
            if output_lens:
                config_output_lens = output_lens
            else:
                # Calculate output_len for this configuration when not provided.
                # Goal: dilute chunked-prefill mixed-step ITL pollution so
                # median_itl reflects pure-decode steady state.
                #
                # prefill_steps ≈ ceil(bs × input / mbt): how many engine
                # iterations are needed to drain all prompt tokens through
                # chunked prefill at the configured token budget.
                #
                # decode_steps = factor × prefill_steps: median stays clean
                # when pure-decode samples > polluted ones (O ≥ 2C+2 strict;
                # 2.5x gives ~21% margin against scheduler/SSE jitter).
                #
                # Bounds: 4096 (keeps the E2E decode window non-empty under
                # high-concurrency first-token jitter at large batch sizes, e.g.
                # bs=256, where a short output collapses the decode window to a
                # non-positive duration) ~ 6144 tokens (scaled by spec_factor for
                # speculative decoding so MTP keeps the same decode-step cap
                # instead of being clipped by token count).
                if max_num_batched_tokens <= 0:
                    raise ValueError(f"max_num_batched_tokens must be positive, got {max_num_batched_tokens}")
                MIN_OUTPUT_LEN = 4096
                DECODE_STEPS_PER_PREFILL_STEP = 3
                BASE_MAX_OUTPUT_LEN = 6144
                prefill_steps = max(
                    1,
                    math.ceil(batch_size * input_len / max_num_batched_tokens),
                )
                # spec_factor = 1 + N*accept_rate scales token count up
                # because each decode step emits >1 token under spec
                # decoding; without scaling, decode_steps would be diluted
                # below prefill_steps and median_itl gets polluted by
                # prefill chunks.
                max_output_len = math.ceil(BASE_MAX_OUTPUT_LEN * max(1.0, spec_factor))
                auto_output_len = min(
                    max_output_len,
                    max(
                        MIN_OUTPUT_LEN,
                        math.ceil(DECODE_STEPS_PER_PREFILL_STEP * prefill_steps * spec_factor),
                    ),
                )
                config_output_lens = [auto_output_len]

            for output_len in config_output_lens:
                if output_len <= 0:
                    raise ValueError(f"output_len must be positive, got {output_len}")

                if kv_capacity_unavailable_reason is not None:
                    print(
                        f"  Skipping configuration: {kv_capacity_unavailable_reason}",
                        flush=True,
                    )
                    results.append(
                        TpsMetric(
                            batch_size=batch_size,
                            input_len=input_len,
                            output_len=output_len,
                            request_logprobs=client.request_logprobs,
                            return_token_ids=client.return_token_ids,
                            weight_version=client.weight_version,
                            status="skipped",
                            skip_reason=kv_capacity_unavailable_reason,
                        )
                    )
                    continue

                required_kv_tokens = (input_len + output_len) * batch_size
                if input_len + output_len > max(math.floor(v * KV_CACHE_GUARD_FACTOR) for v in capacities.values()):
                    raise RuntimeError("No DP can hold even one request under capacity guard")
                usable_kv_tokens = math.floor(num_kv_cache * KV_CACHE_GUARD_FACTOR)
                if num_kv_cache > 0 and usable_kv_tokens < required_kv_tokens:
                    skip_reason = (
                        f"KV cache guard ({usable_kv_tokens} usable tokens, "
                        f"{KV_CACHE_GUARD_FACTOR:.2f} * {num_kv_cache}) cannot hold "
                        f"batch_size={batch_size}, input_len={input_len}, "
                        f"output_len={output_len} "
                        f"({required_kv_tokens} tokens)"
                    )
                    print(
                        f"  Skipping configuration: {skip_reason}. "
                        "Scheduler queueing or decode preemption would "
                        "contaminate pure-decode TPS.",
                        flush=True,
                    )
                    results.append(
                        TpsMetric(
                            batch_size=batch_size,
                            input_len=input_len,
                            output_len=output_len,
                            request_logprobs=client.request_logprobs,
                            return_token_ids=client.return_token_ids,
                            weight_version=client.weight_version,
                            status="skipped",
                            skip_reason=skip_reason,
                        )
                    )
                    raise RuntimeError(skip_reason)

                # Check max_model_len limit
                # Extract from nested JSON: vllm_config.model_config.max_model_len
                max_seq_len = capacity["max_model_len"]
                if max_seq_len is not None:
                    max_input_len = max_seq_len - output_len
                    if input_len > max_input_len:
                        skip_reason = (
                            f"input_len={input_len} exceeds max input length "
                            f"{max_input_len} for max_model_len={max_seq_len} "
                            f"and output_len={output_len}"
                        )
                        print(f"  Skipping configuration: {skip_reason}")
                        results.append(
                            TpsMetric(
                                batch_size=batch_size,
                                input_len=input_len,
                                output_len=output_len,
                                request_logprobs=client.request_logprobs,
                                return_token_ids=client.return_token_ids,
                                weight_version=client.weight_version,
                                status="skipped",
                                skip_reason=skip_reason,
                            )
                        )
                        raise RuntimeError(skip_reason)

                config_idx += 1
                print(f"\nProgress: {config_idx}/{total_configs}")

                max_attempts = config_retries + 1
                last_error = None
                config_succeeded = False
                for attempt_idx in range(1, max_attempts + 1):
                    try:
                        result = await run_single_benchmark(
                            client,
                            batch_size,
                            input_len,
                            output_len,
                            num_iters,
                            seed,
                            vocab_size,
                            max_connections,
                            transformer_config,
                            prefill_total_gpu_flops,
                            decode_total_gpu_flops,
                            stats_last_iters=stats_last_iters,
                        )
                        if disable_mfu:
                            result.prefill_flops = result.decode_flops = None
                            result.prefill_mfu = result.decode_mfu = None
                        results.append(result)

                        print(
                            f"  Results: Prefill TPS={result.prefill_tps:.2f}, "
                            "Decode TPS E2E Window="
                            f"{result.decode_tps_e2e_window:.2f}"
                        )
                        if transformer_config:
                            print(
                                "  FLOPS: "
                                f"Prefill={result.prefill_flops:.2f} TFLOPS, "
                                f"Decode={result.decode_flops:.2f} TFLOPs"
                            )
                            print(f"  MFU: Prefill={result.prefill_mfu:.2%}, Decode={result.decode_mfu:.2%}")
                        config_succeeded = True
                        break
                    except Exception as e:
                        last_error = e
                        print(f"  Configuration attempt {attempt_idx}/{max_attempts} failed: {e}")
                        # Do not mutate or query cleanup endpoints; stop immediately.
                        raise
                        if attempt_idx >= max_attempts:
                            break
                        await asyncio.sleep(min(15, 5 * attempt_idx))

                if config_succeeded:
                    continue

                print(f"  Configuration test failed after retries: {last_error}")
                failed_diagnostics = {}
                if isinstance(last_error, BenchmarkQualityError):
                    failed_diagnostics = last_error.result
                results.append(
                    TpsMetric(
                        batch_size=batch_size,
                        input_len=input_len,
                        output_len=output_len,
                        prefill_tps=0.0,
                        decode_tps_e2e_window=0.0,
                        max_connections=max_connections,
                        request_logprobs=client.request_logprobs,
                        return_token_ids=client.return_token_ids,
                        weight_version=client.weight_version,
                        status="failed",
                        error=str(last_error),
                        decode_window_start=failed_diagnostics.get("decode_window_start", 0.0),
                        decode_window_end=failed_diagnostics.get("decode_window_end", 0.0),
                        decode_window_duration=failed_diagnostics.get("decode_window_duration", 0.0),
                        decode_window_tokens=failed_diagnostics.get("decode_window_tokens", 0),
                        decode_window_chunks=failed_diagnostics.get("decode_window_chunks", 0),
                        decode_window_token_coverage=failed_diagnostics.get("decode_window_token_coverage", 0.0),
                        decode_window_itl_count=failed_diagnostics.get("decode_window_itl_count", 0),
                        itl_avg=failed_diagnostics.get("itl_avg", 0.0),
                        itl_p50=failed_diagnostics.get("itl_p50", 0.0),
                        itl_p90=failed_diagnostics.get("itl_p90", 0.0),
                        itl_p99=failed_diagnostics.get("itl_p99", 0.0),
                        response_chunk_tokens_avg=failed_diagnostics.get("response_chunk_tokens_avg", 0.0),
                        response_chunk_tokens_p10=failed_diagnostics.get("response_chunk_tokens_p10", 0.0),
                        response_chunk_tokens_p50=failed_diagnostics.get("response_chunk_tokens_p50", 0.0),
                        response_chunk_tokens_p90=failed_diagnostics.get("response_chunk_tokens_p90", 0.0),
                        response_chunk_tokens_max=failed_diagnostics.get("response_chunk_tokens_max", 0),
                        complete_requests=failed_diagnostics.get("complete_requests", 0),
                        incomplete_requests=failed_diagnostics.get("incomplete_requests", 0),
                        total_completion_tokens=failed_diagnostics.get("total_completion_tokens", 0),
                        decode_window_quality=failed_diagnostics.get("decode_window_quality", {}),
                    )
                )

    max_seq_len = capacity["max_model_len"]
    if disable_mfu:
        single_gpu_flops = total_gpu_flops = None
        prefill_total_gpu_flops = decode_total_gpu_flops = None

    # Convert to benchmark_static_performance.py format
    final_results = {
        "memory_settings": {
            "max_model_len": max_seq_len,
            "num_kv_cache": num_kv_cache,
            "kv_cache_capacity_source": kv_cache_capacity_source,
            "kv_cache_capacity_available": num_kv_cache > 0,
            "server_info_cache_config": server_info_cache_config,
            "gpu_memory_utilization": gpu_memory_utilization,
            "single_gpu_flops": single_gpu_flops,
            "total_gpu_flops": total_gpu_flops,
            "prefill_total_gpu_flops": prefill_total_gpu_flops,
            "decode_total_gpu_flops": decode_total_gpu_flops,
        },
        "system_config": {
            "model": model_name,
            "mfu_enabled": not disable_mfu,
            "mfu_disabled_reason": "Ascend: no validated peak FLOPS / MTP MFU model" if disable_mfu else None,
            "capacity_guard_semantics": (
                "necessary global capacity check only; actual per-DP routing/residency requires external evidence"
            ),
            "measurement_semantics": "client E2E overlap, not proven pure decode",
            "itl_semantics": "SSE event interval, not per-token ITL",
            "tp_size": tp_size,
            "dp_size": dp_size,
            "pp_size": 1,  # Default
            "prefill_gpu_count": prefill_gpu_count,
            "decode_gpu_count": decode_gpu_count,
            "dtype": precision,
            "request_logprobs": request_logprobs,
            "return_token_ids": return_token_ids,
            "weight_version": weight_version,
        },
    }
    if stats_last_iters is not None:
        final_results["system_config"].update(
            {
                "num_iters": num_iters,
                "stats_last_iters": stats_last_iters,
            }
        )

    # Add each configuration result with the expected key format
    for result in results:
        config_key = f"bs{result.batch_size}_input{result.input_len}_output{result.output_len}"
        final_results[config_key] = {
            "decode_latency": result.decode_latency,
            "prefill_latency": result.prefill_latency,
            "decode_tps_e2e_window": result.decode_tps_e2e_window,
            "prefill_tps": result.prefill_tps,
            "decode_window_start": result.decode_window_start,
            "decode_window_end": result.decode_window_end,
            "decode_window_duration": result.decode_window_duration,
            "decode_window_tokens": result.decode_window_tokens,
            "decode_window_chunks": result.decode_window_chunks,
            "decode_window_token_coverage": result.decode_window_token_coverage,
            "decode_window_itl_count": result.decode_window_itl_count,
            "itl_avg": result.itl_avg,
            "itl_p50": result.itl_p50,
            "itl_p90": result.itl_p90,
            "itl_p99": result.itl_p99,
            "response_chunk_tokens_avg": result.response_chunk_tokens_avg,
            "response_chunk_tokens_p10": result.response_chunk_tokens_p10,
            "response_chunk_tokens_p50": result.response_chunk_tokens_p50,
            "response_chunk_tokens_p90": result.response_chunk_tokens_p90,
            "response_chunk_tokens_max": result.response_chunk_tokens_max,
            "decode_flops": result.decode_flops,
            "prefill_flops": result.prefill_flops,
            "decode_mfu": result.decode_mfu,
            "prefill_mfu": result.prefill_mfu,
            "complete_requests": getattr(result, "complete_requests", 0),
            "incomplete_requests": getattr(result, "incomplete_requests", 0),
            "total_completion_tokens": getattr(result, "total_completion_tokens", 0),
            "decode_window_quality": getattr(result, "decode_window_quality", {}),
            "request_logprobs": getattr(result, "request_logprobs", None),
            "return_token_ids": getattr(result, "return_token_ids", False),
            "weight_version": getattr(result, "weight_version", None),
            "status": getattr(result, "status", "completed"),
            "skip_reason": getattr(result, "skip_reason", None),
            "error": getattr(result, "error", None),
        }

    require_completed_results(final_results)
    return final_results


def save_results_to_json(results: dict[str, Any], output_file: str):
    """Save results to JSON file in benchmark_static_performance.py format"""

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"\nResults saved to: {output_file}")


def print_results_table_from_dict(results: dict[str, Any]):
    """Print results in table format from the new dictionary structure"""
    print("\n" + "=" * 120)
    print("Benchmark Results")
    print("=" * 120)

    # Extract system info
    memory_settings = results.get("memory_settings", {})
    system_config = results.get("system_config", {})

    print(f"Model: {system_config.get('model', 'Unknown')}")
    print(
        f"System Config: TP={system_config.get('tp_size', 1)}, "
        f"DP={system_config.get('dp_size', 1)}, "
        f"PP={system_config.get('pp_size', 1)}, "
        f"Precision={system_config.get('dtype', 'unknown')}"
    )
    print(f"Memory: KV Cache={memory_settings.get('num_kv_cache')} (explicit metadata)")
    if not system_config.get("mfu_enabled", True):
        print("FLOPS/MFU: N/A (explicitly disabled; no NVIDIA probe)")
    else:
        print(f"GPU FLOPS: {memory_settings.get('single_gpu_flops')} TFLOPS/GPU")
    print("")

    # Check if FLOPS data is available
    has_flops_data = False
    config_keys = [k for k in results if k.startswith("bs")]
    if config_keys:
        sample_metrics = results[config_keys[0]]
        has_flops_data = (sample_metrics.get("prefill_flops") or 0) > 0 or (sample_metrics.get("decode_flops") or 0) > 0

    if has_flops_data:
        # Full table with FLOPS and MFU
        print(f"{'Config':<25} {'Prefill':<35} {'Decode':<35}")
        print(
            f"{'':25} {'TPS':<8} {'Lat(s)':<8} {'FLOPS':<8} {'MFU':<8} {'TPS':<8} {'Lat(s)':<8} {'FLOPS':<8} {'MFU':<8}"
        )
        print("-" * 120)

        # Sort configuration keys for consistent display
        config_keys.sort()

        for config_key in config_keys:
            metrics = results[config_key]
            status = metrics.get("status", "completed")
            if status != "completed":
                reason = metrics.get("skip_reason") or metrics.get("error") or ""
                print(f"{config_key:<25} {status:<35} {reason[:35]:<35}")
                continue
            prefill = metrics
            decode = metrics
            print(
                f"{config_key:<25} "
                f"{prefill['prefill_tps']:<8.1f} "
                f"{prefill['prefill_latency']:<8.4f} "
                f"{prefill['prefill_flops']:<8.2f} "
                f"{prefill['prefill_mfu']:<8.2%} "
                f"{decode['decode_tps_e2e_window']:<8.1f} "
                f"{decode['decode_latency']:<8.4f} "
                f"{decode['decode_flops']:<8.2f} "
                f"{decode['decode_mfu']:<8.2%}"
            )
    else:
        # Simple table without FLOPS and MFU
        print(f"{'Config':<25} {'Prefill TPS':<15} {'Decode TPS E2E':<15} {'Prefill Lat(s)':<15} {'Decode Lat(s)':<15}")
        print("-" * 100)

        # Sort configuration keys for consistent display
        config_keys.sort()

        for config_key in config_keys:
            metrics = results[config_key]
            status = metrics.get("status", "completed")
            if status != "completed":
                reason = metrics.get("skip_reason") or metrics.get("error") or ""
                print(f"{config_key:<25} {status:<15} {reason[:55]:<55}")
                continue
            print(
                f"{config_key:<25} "
                f"{metrics['prefill_tps']:<15.2f} "
                f"{metrics['decode_tps_e2e_window']:<15.2f} "
                f"{metrics['prefill_latency']:<15.4f} "
                f"{metrics['decode_latency']:<15.4f}"
            )


def parse_int_list(value: str) -> list[int]:
    """Parse comma-separated integer list"""
    return [int(x.strip()) for x in value.split(",")]


def parse_case_filter_json(value: str) -> set[tuple[int, int]]:
    """Parse JSON static case filter from step_benchmark_runner.py."""
    try:
        raw_cases = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"--case-filter-json must be valid JSON: {exc}") from exc

    if not isinstance(raw_cases, list) or not raw_cases:
        raise argparse.ArgumentTypeError("--case-filter-json must be a non-empty JSON list")

    parsed_cases = set()
    for index, raw_case in enumerate(raw_cases):
        if not isinstance(raw_case, dict):
            raise argparse.ArgumentTypeError(f"--case-filter-json[{index}] must be an object")
        if "output_len" in raw_case or "output_lens" in raw_case:
            raise argparse.ArgumentTypeError("--case-filter-json supports batch_size/input_len only")
        batch_size = raw_case.get("batch_size")
        input_len = raw_case.get("input_len")
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise argparse.ArgumentTypeError(f"--case-filter-json[{index}].batch_size must be a positive integer")
        if not isinstance(input_len, int) or isinstance(input_len, bool) or input_len <= 0:
            raise argparse.ArgumentTypeError(f"--case-filter-json[{index}].input_len must be a positive integer")
        parsed_cases.add((batch_size, input_len))

    return parsed_cases


async def main():
    parser = argparse.ArgumentParser(
        description=("Benchmark Prefill/Decode TPS for different batch sizes and input lengths")
    )

    # Service connection parameters
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host address")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument(
        "--max-connections",
        type=int,
        default=1024,
        help="Maximum number of concurrent HTTP connections (default: 1000). "
        "Increase this value for high batch size testing to avoid connection pool "
        "limitations.",
    )

    # Test configuration parameters
    parser.add_argument(
        "--batch-sizes",
        type=parse_int_list,
        default=[1, 8, 32],
        help="List of batch sizes to test, comma-separated (e.g., 1,8,32)",
    )
    parser.add_argument(
        "--input-lens",
        type=parse_int_list,
        default=[1024, 4096],
        help="List of input lengths to test, comma-separated (e.g., 128,512,1024)",
    )
    parser.add_argument(
        "--output-lens",
        type=parse_int_list,
        default=None,
        help=(
            "Optional list of output lengths to test, comma-separated "
            "(e.g., 64,128,256). If omitted, max_tokens is auto-calculated."
        ),
    )
    parser.add_argument(
        "--case-filter-json",
        type=parse_case_filter_json,
        default=None,
        help=(
            "Optional JSON list of static cases to run. Each item must contain "
            "batch_size and input_len. output_lens remain config-level."
        ),
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=8192,
        help="Maximum number of batched tokens for auto-adjusting max_tokens",
    )
    parser.add_argument(
        "--num-iters",
        type=int,
        default=1,
        help="Number of test iterations per configuration",
    )
    parser.add_argument(
        "--stats-last-iters",
        type=int,
        default=None,
        help=("Aggregate statistics from only the trailing N iterations. By default, all iterations are included."),
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=65536,
        help="Vocabulary size for token generation",
    )

    parser.add_argument(
        "--tp-size",
        type=int,
        default=1,
        help="Tensor parallelism size (for MFU calculation)",
    )
    parser.add_argument(
        "--dp-size",
        type=int,
        default=1,
        help="Data parallelism size (for MFU calculation)",
    )
    parser.add_argument(
        "--prefill-gpu-count",
        type=int,
        default=None,
        help="Actual GPU count used by the prefill side (for PD MFU calculation)",
    )
    parser.add_argument(
        "--decode-gpu-count",
        type=int,
        default=None,
        help="Actual GPU count used by the decode side (for PD MFU calculation)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="bf16",
        help="Model precision (affects FLOPS calculation for fp8)",
    )
    parser.add_argument(
        "--logprobs",
        type=int,
        default=None,
        help=(
            "Optional completions request logprobs value. For example, "
            "--logprobs 0 returns sampled-token logprobs without top logprobs."
        ),
    )
    parser.add_argument(
        "--return-token-ids",
        action="store_true",
        help=("Compatibility flag. TPS token counting requires token_ids, so return_token_ids=true is always sent."),
    )
    parser.add_argument(
        "--spec-factor",
        type=float,
        default=1.0,
        help=(
            "Speculative decoding amplification factor: 1 + N*accept_rate. "
            "Used to scale auto-derived output_len so decode steps (not "
            "tokens) dominate the ITL distribution. Default 1.0 (no spec)."
        ),
    )
    parser.add_argument(
        "--weight-version",
        type=int,
        default=None,
        help=(
            "Set server-side weight_version through /set_model_weight before "
            "benchmarking. The server must run with VLLM_SERVER_DEV_MODE=1 "
            "and VLLM_ENABLE_WEIGHT_VERSION=1."
        ),
    )

    # Random seed parameter
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible results")

    # Timing analysis parameters (now auto-calculated)
    # parser.add_argument("--timestamp-threshold", type=float, default=None,
    #     help="Threshold in seconds for batch decode detection.")

    # Output parameters
    parser.add_argument("--output-file", type=str, default=None, help="Path to save results JSON file")
    parser.add_argument(
        "--request-total-timeout",
        type=int,
        default=900,
        help="Per-request total timeout in seconds",
    )
    parser.add_argument(
        "--request-sock-connect-timeout",
        type=int,
        default=30,
        help="Socket connect timeout in seconds",
    )
    parser.add_argument(
        "--request-sock-read-timeout",
        type=int,
        default=300,
        help="Socket read timeout in seconds",
    )
    parser.add_argument(
        "--config-retries",
        type=int,
        default=0,
        help="Number of retry attempts for each benchmark configuration",
    )
    parser.add_argument(
        "--failure-cleanup-mode",
        choices=["off"],
        default="off",
        help="Safe copy always stops on first failure; no pause/resume/cleanup API calls.",
    )
    parser.add_argument(
        "--failure-cleanup-timeout",
        type=int,
        default=300,
        help="Seconds to wait for /active_requests to reach zero after failure",
    )
    parser.add_argument(
        "--failure-cleanup-poll-interval",
        type=int,
        default=5,
        help="Polling interval in seconds while waiting for failure cleanup",
    )

    parser.add_argument("--disable-mfu", action="store_true", help="No NVIDIA probe; FLOPS and MFU null/N/A")
    parser.add_argument(
        "--capacity-metadata", help="Explicit verified per-DP logical KV capacity JSON (no server schema assumed)"
    )
    parser.add_argument(
        "--raw-artifacts-dir",
        required=True,
        help="New non-existing run directory for raw SSE and request/iteration JSON",
    )
    args = parser.parse_args()

    print("vLLM Prefill/Decode TPS Benchmark")
    print(f"Target service: {args.host}:{args.port}")
    print(f"Model: {args.model}")
    print(f"Batch sizes: {args.batch_sizes}")
    print(f"Input lengths: {args.input_lens}")
    if args.output_lens:
        print(f"Output lengths: {args.output_lens}")
    else:
        print(f"Max batched tokens: {args.max_num_batched_tokens} (for auto max_tokens calculation)")
    if args.case_filter_json is not None:
        print(f"Static case filter pairs: {sorted(args.case_filter_json)}")
    print(f"Iterations: {args.num_iters}")
    print(f"Statistics iterations: last {args.stats_last_iters or args.num_iters} of {args.num_iters}")
    print(f"Max connections: {args.max_connections}")
    print(
        "Request timeouts: "
        f"total={args.request_total_timeout}s, "
        f"connect={args.request_sock_connect_timeout}s, "
        f"read={args.request_sock_read_timeout}s"
    )
    print(
        "Failure cleanup: "
        f"mode={args.failure_cleanup_mode}, "
        f"timeout={args.failure_cleanup_timeout}s, "
        f"poll={args.failure_cleanup_poll_interval}s"
    )
    print(f"Parallel config: TP={args.tp_size}, DP={args.dp_size}")
    print(f"Request logprobs: {args.logprobs}")
    print(f"Return token IDs: {args.return_token_ids}")
    print(f"Set weight_version: {args.weight_version}")
    if args.prefill_gpu_count or args.decode_gpu_count:
        print(f"Stage GPU counts: Prefill={args.prefill_gpu_count}, Decode={args.decode_gpu_count}")
    # print(f"Timestamp threshold: {args.timestamp_threshold*1000:.1f}ms")
    print("\n💡 Enhanced Streaming API mode:")
    if args.output_lens:
        print("   - Uses explicit output_lens from the benchmark configuration")
    else:
        print("   - Auto-adjusts max_tokens based on chunked_prefill with decode priority")
        print(
            "   - Formula: max_tokens = "
            "min(6144 * spec_factor, "
            "3 * ceil(batch_size * input_len / max_num_batched_tokens) "
            "* spec_factor)"
        )
    print("   - Uses verified explicit capacity/max_model_len metadata; server_info is probe evidence only")
    print("   - Uses streaming API with precise timestamp analysis")
    print("   - Simplified logging for cleaner output")

    artifact_directory_preexisted = Path(args.raw_artifacts_dir).exists()
    try:
        # Run benchmark
        results = await run_benchmark_suite(
            host=args.host,
            port=args.port,
            model_name=args.model,
            batch_sizes=args.batch_sizes,
            input_lens=args.input_lens,
            output_lens=args.output_lens,
            max_num_batched_tokens=args.max_num_batched_tokens,
            case_filter=args.case_filter_json,
            num_iters=args.num_iters,
            seed=args.seed,
            vocab_size=args.vocab_size,
            max_connections=args.max_connections,
            tp_size=args.tp_size,
            dp_size=args.dp_size,
            prefill_gpu_count=args.prefill_gpu_count,
            decode_gpu_count=args.decode_gpu_count,
            precision=args.precision,
            request_logprobs=args.logprobs,
            return_token_ids=args.return_token_ids,
            weight_version=args.weight_version,
            spec_factor=args.spec_factor,
            request_total_timeout=args.request_total_timeout,
            request_sock_connect_timeout=args.request_sock_connect_timeout,
            request_sock_read_timeout=args.request_sock_read_timeout,
            config_retries=args.config_retries,
            failure_cleanup_mode=args.failure_cleanup_mode,
            failure_cleanup_timeout=args.failure_cleanup_timeout,
            failure_cleanup_poll_interval=args.failure_cleanup_poll_interval,
            stats_last_iters=args.stats_last_iters,
            disable_mfu=args.disable_mfu,
            capacity_metadata=args.capacity_metadata,
            raw_artifacts_dir=args.raw_artifacts_dir,
        )

        # Print results
        print_results_table_from_dict(results)

        # Save results to file
        if args.output_file:
            save_results_to_json(results, args.output_file)

        failed_configs = [
            key for key, value in results.items() if isinstance(value, dict) and value.get("status") == "failed"
        ]
        if failed_configs:
            print("\nBenchmark failed configurations: " + ", ".join(sorted(failed_configs)))
            return 1

        print("\nBenchmark completed!")
        print("\n💡 Performance notes:")
        if args.output_lens:
            print("   - output_lens were taken directly from the CLI/config")
        else:
            print("   - max_tokens auto-calculated for chunked_prefill decode priority")
        print("   - max_seq_len taken from verified explicit metadata")
        print("   - TPS is client E2E overlap only; ITL is SSE-event interval; pure decode needs external traces")

    except Exception as e:
        if not artifact_directory_preexisted and Path(args.raw_artifacts_dir).is_dir():
            (Path(args.raw_artifacts_dir) / "failure.json").write_text(
                json.dumps(
                    {"status": "failed", "error": f"{type(e).__name__}: {e}", "wall_time_unix": time.time()}, indent=2
                ),
                encoding="utf-8",
            )
        print(f"Benchmark failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(asyncio.run(main()))
