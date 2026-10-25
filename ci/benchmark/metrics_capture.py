"""Read-only Prometheus evidence and strict all-DP speculative-token deltas."""

import asyncio
import hashlib
import json
import math
import os
import re
import sys
import time
from decimal import Decimal, InvalidOperation
from pathlib import Path

DRAFT = "vllm:spec_decode_num_draft_tokens_total"
ACCEPTED = "vllm:spec_decode_num_accepted_tokens_total"
SCHEDULER = {
    "running": "vllm:num_requests_running",
    "waiting": "vllm:num_requests_waiting",
    "preemptions": "vllm:num_preemptions_total",
    "kv_usage": "vllm:kv_cache_usage_perc",
}
TRACKED = {DRAFT, ACCEPTED, *SCHEDULER.values()}
SAMPLE = re.compile(
    r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(.*?)\})?\s+"
    r"([^\s]+)(?:\s+[-+]?\d+(?:\.\d+)?)?(?:\s+#.*)?$"
)
LABEL = re.compile(r'\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*"((?:\\.|[^"\\])*)"\s*')


def parse_labels(text):
    labels = {}
    position = 0
    while position < len(text):
        match = LABEL.match(text, position)
        if match is None:
            raise ValueError("Malformed Prometheus labels")
        key, escaped = match.groups()
        if key in labels:
            raise ValueError("Duplicate Prometheus label")
        value = []
        index = 0
        while index < len(escaped):
            char = escaped[index]
            if char == "\\":
                index += 1
                char = {"\\": "\\", '"': '"', "n": "\n"}.get(escaped[index])
                if char is None:
                    raise ValueError("Unsupported Prometheus label escape")
            value.append(char)
            index += 1
        labels[key] = "".join(value)
        position = match.end()
        if position == len(text):
            break
        if text[position] != ",":
            raise ValueError("Malformed Prometheus label separator")
        position += 1
        if not text[position:].strip():
            raise ValueError("Trailing Prometheus label separator")
    return labels


def parse_metrics(text):
    """Preserve tracked series and errors; never infer missing labels or values."""
    series, errors = [], []
    for line_number, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name = re.split(r"[\s{]", line, maxsplit=1)[0]
        if name not in TRACKED:
            continue
        try:
            match = SAMPLE.fullmatch(line)
            if match is None:
                raise ValueError("Malformed tracked sample")
            name, label_text, numeric = match.groups()
            labels = parse_labels(label_text or "")
            value = Decimal(numeric)
            if not value.is_finite() or value < 0:
                raise ValueError("Metric must be finite and nonnegative")
            if name != SCHEDULER["kv_usage"] and value != value.to_integral_value():
                raise ValueError("Counter/request count must be integral")
            if name == SCHEDULER["kv_usage"]:
                value = float(value)
                if not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError("KV usage must be in [0, 1]")
            else:
                value = int(value)
            series.append({"name": name, "labels": labels, "value": value})
        except (ValueError, InvalidOperation) as error:
            errors.append({"line": line_number, "metric": name, "error": str(error)})
    return {"series": series, "errors": errors}


def metric_by_dp(parsed, metric, model, dp_size):
    """Require one exact model/engine series per DP; no API/TP replica sums."""
    values, identities, errors = {}, {}, []
    errors.extend(
        f"{metric}: parse error at line {error['line']}"
        for error in parsed.get("errors", [])
        if error["metric"] == metric
    )
    for row in parsed.get("series", []):
        if row["name"] != metric:
            continue
        labels = row["labels"]
        if "model_name" not in labels:
            errors.append(f"{metric}: missing model_name label")
            continue
        if labels["model_name"] != model:
            continue
        engine = labels.get("engine")
        if engine not in {str(index) for index in range(dp_size)}:
            errors.append(f"{metric}: unknown/missing engine label {engine!r}")
            continue
        if engine in values:
            errors.append(f"{metric}: duplicate DP {engine} series")
            continue
        values[engine] = row["value"]
        identities[engine] = labels
    missing = sorted({str(index) for index in range(dp_size)} - set(values))
    if missing:
        errors.append(f"{metric}: missing DP series {missing}")
    return {"valid": not errors, "per_dp": values, "identities": identities, "errors": errors}


def snapshot_observations(parsed, model, dp_size):
    observations = {}
    for key, metric in SCHEDULER.items():
        result = metric_by_dp(parsed, metric, model, dp_size)
        if result["valid"] and key != "kv_usage":
            result["all_dp_sum"] = sum(result["per_dp"].values())
        # KV fractions are per-engine observations, not additive capacity.
        observations[key] = result
    return observations


def scheduler_summary(snapshots, model, dp_size):
    """Summarize observed gauges/counters without claiming continuous residency."""
    result = {}
    for key, metric in SCHEDULER.items():
        maps = []
        invalid = []
        for index, snapshot in enumerate(snapshots):
            current = (
                metric_by_dp(snapshot["parsed"], metric, model, dp_size) if snapshot.get("valid") else {"valid": False}
            )
            if not current["valid"]:
                invalid.append(index)
            else:
                maps.append(current)
        summary = {
            "valid": bool(snapshots) and not invalid,
            "invalid_snapshot_indices": invalid,
            "observed_complete_snapshots": len(maps),
            "per_dp": {},
        }
        for engine in (str(index) for index in range(dp_size)):
            values = [row["per_dp"][engine] for row in maps]
            if values:
                summary["per_dp"][engine] = {"observed_min": min(values), "observed_max": max(values)}
        if key == "preemptions":
            summary["delta"] = None
            if summary["valid"]:
                changed = any(
                    after["identities"] != before["identities"]
                    or any(after["per_dp"][engine] < before["per_dp"][engine] for engine in before["per_dp"])
                    for before, after in zip(maps, maps[1:])
                )
                if changed:
                    summary["valid"] = False
                    summary["error"] = "Preemption counter reset/decrease or series identity changed"
                else:
                    summary["delta"] = sum(
                        maps[-1]["per_dp"][engine] - maps[0]["per_dp"][engine] for engine in maps[0]["per_dp"]
                    )
        result[key] = summary
    result["semantics"] = "Observed 1s samples only; no pure-decode or simultaneous-residency assertion"
    return result


def acceptance_delta(snapshots, model, dp_size):
    """Bracketed cumulative-counter delta; any observed reset/unknown invalidates."""
    result = {
        "valid": False,
        "ratio": None,
        "draft_tokens": None,
        "accepted_tokens": None,
        "per_dp": {},
        "errors": [],
        "scope": "all-DP service counters across the full measured iteration, not pure decode",
    }
    if len(snapshots) < 2 or snapshots[0].get("phase") != "before" or snapshots[-1].get("phase") != "after":
        result["errors"].append("Missing before/after metrics brackets")
        return result
    maps = []
    for snapshot in snapshots:
        if not snapshot.get("valid"):
            result["errors"].append(f"Invalid metrics snapshot {snapshot.get('sample_index')}")
            continue
        pair = {}
        for metric in (DRAFT, ACCEPTED):
            current = metric_by_dp(snapshot["parsed"], metric, model, dp_size)
            result["errors"].extend(current["errors"])
            pair[metric] = current
        if pair[DRAFT]["valid"] and pair[ACCEPTED]["valid"]:
            for engine in pair[DRAFT]["identities"]:
                if pair[DRAFT]["identities"][engine] != pair[ACCEPTED]["identities"][engine]:
                    result["errors"].append(f"DP {engine}: draft/accepted series labels differ")
        maps.append(pair)
    if result["errors"]:
        return result
    for engine in (str(index) for index in range(dp_size)):
        for metric in (DRAFT, ACCEPTED):
            for before, after in zip(maps, maps[1:]):
                if before[metric]["identities"][engine] != after[metric]["identities"][engine]:
                    result["errors"].append(f"DP {engine} {metric}: series identity changed")
                if after[metric]["per_dp"][engine] < before[metric]["per_dp"][engine]:
                    result["errors"].append(f"DP {engine} {metric}: counter reset/decrease")
        drafted = maps[-1][DRAFT]["per_dp"][engine] - maps[0][DRAFT]["per_dp"][engine]
        accepted = maps[-1][ACCEPTED]["per_dp"][engine] - maps[0][ACCEPTED]["per_dp"][engine]
        if accepted > drafted:
            result["errors"].append(f"DP {engine}: accepted delta exceeds draft delta")
        result["per_dp"][engine] = {"draft_tokens": drafted, "accepted_tokens": accepted}
    if result["errors"]:
        return result
    drafted = sum(row["draft_tokens"] for row in result["per_dp"].values())
    accepted = sum(row["accepted_tokens"] for row in result["per_dp"].values())
    if drafted == 0:
        result["errors"].append("No draft-token increase; acceptance is undefined")
        return result
    result.update(valid=True, draft_tokens=drafted, accepted_tokens=accepted, ratio=accepted / drafted)
    return result


def weighted_acceptance(iterations):
    """Weight by drafted tokens; do not silently exclude invalid iterations."""
    invalid = [index + 1 for index, result in enumerate(iterations) if not result.get("valid")]
    result = {
        "valid": False,
        "ratio": None,
        "draft_tokens": None,
        "accepted_tokens": None,
        "iteration_count": len(iterations),
        "invalid_iterations": invalid,
        "aggregation": "sum(accepted delta) / sum(draft delta), not mean(iteration ratios)",
    }
    if not iterations or invalid:
        return result
    drafted = sum(row["draft_tokens"] for row in iterations)
    accepted = sum(row["accepted_tokens"] for row in iterations)
    if drafted > 0:
        result.update(valid=True, draft_tokens=drafted, accepted_tokens=accepted, ratio=accepted / drafted)
    return result


class MetricsCapture:
    """One sequential telemetry connection; nominal 1s polling, no retries."""

    def __init__(self, fetch, directory, stem, model, dp_size):
        self.fetch = fetch
        self.directory = Path(directory)
        self.stem = stem
        self.model = model
        self.dp_size = dp_size
        self.snapshots = []
        self.stop = asyncio.Event()
        self.task = None

    async def snapshot(self, phase):
        index = len(self.snapshots)
        filename = f"{self.stem}-metrics-{index:04d}-{phase}"
        started = time.perf_counter()
        row = {
            "sample_index": index,
            "phase": phase,
            "started_perf_counter": started,
            "started_wall_time_unix": time.time(),
            "valid": False,
            "raw_file": filename + ".txt",
            "json_file": filename + ".json",
        }
        raw = b""
        try:
            status, headers, raw = await self.fetch()
            row.update(http_status=status, http_headers=headers)
            if status != 200:
                row["error"] = f"Metrics HTTP status {status}; no retry"
            else:
                row["parsed"] = parse_metrics(raw.decode("utf-8"))
                row["observations"] = snapshot_observations(row["parsed"], self.model, self.dp_size)
                row["valid"] = True
        except Exception as error:
            row["error"] = f"{type(error).__name__}: {error}; no retry"
        row["finished_perf_counter"] = time.perf_counter()
        row["finished_wall_time_unix"] = time.time()
        row["duration_seconds"] = row["finished_perf_counter"] - started
        row["raw_sha256"] = hashlib.sha256(raw).hexdigest()
        (self.directory / row["raw_file"]).write_bytes(raw)
        (self.directory / row["json_file"]).write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
        self.snapshots.append(row)

    async def start(self):
        await self.snapshot("before")
        self.task = asyncio.create_task(self.poll())

    async def poll(self):
        next_sample = time.perf_counter() + 1.0
        while not self.stop.is_set():
            try:
                await asyncio.wait_for(self.stop.wait(), timeout=max(0.0, next_sample - time.perf_counter()))
                return
            except TimeoutError:
                pass
            if not self.stop.is_set():
                await self.snapshot("during")
            next_sample += 1.0
            # A slow scrape never causes overlapping requests or catch-up bursts.
            if next_sample <= time.perf_counter():
                next_sample = time.perf_counter() + 1.0

    async def finish(self, iteration_status):
        self.stop.set()
        if self.task is not None:
            await self.task
        await self.snapshot("after")
        acceptance = acceptance_delta(self.snapshots, self.model, self.dp_size)
        if iteration_status != "completed":
            acceptance["valid"] = False
            acceptance["ratio"] = None
            acceptance["errors"].append(f"Iteration {iteration_status}")
        report = {
            "iteration_status": iteration_status,
            "model_name": self.model,
            "expected_dp_size": self.dp_size,
            "sample_interval_seconds": 1.0,
            "snapshots": self.snapshots,
            "acceptance": acceptance,
            "scheduler": scheduler_summary(self.snapshots, self.model, self.dp_size),
            "sampling": {
                "snapshot_count": len(self.snapshots),
                "during_snapshot_count": sum(row["phase"] == "during" for row in self.snapshots),
                "max_start_gap_seconds": max(
                    (
                        after["started_perf_counter"] - before["started_perf_counter"]
                        for before, after in zip(self.snapshots, self.snapshots[1:])
                    ),
                    default=None,
                ),
                "invalid_snapshot_indices": [index for index, row in enumerate(self.snapshots) if not row["valid"]],
            },
            "limitations": [
                "Service-wide counters require exclusive workload attribution; not request-specific.",
                "Scrapes are not atomic across DP ranks and can lag engine logging.",
                "Only observed resets are detectable; missing/unknown observations invalidate acceptance.",
                "One-second running/waiting/KV samples do not prove pure decode or BS128 residency.",
            ],
        }
        (self.directory / f"{self.stem}-metrics-iteration.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        return report


class ProcessMetricsCapture:
    """Production collector in a separate interpreter, independent of client GIL."""

    def __init__(self, url, directory, stem, model, dp_size, worker_script=None):
        if not re.fullmatch(r"[A-Za-z0-9._-]+", stem):
            raise ValueError("Unsafe metrics artifact stem")
        self.url = url
        self.directory = Path(directory).resolve(strict=True)
        self.stem = stem
        self.model = model
        self.dp_size = dp_size
        self.worker_script = Path(worker_script or Path(__file__).with_name("telemetry_worker.py"))
        self.ready_path = self.directory / f"{stem}-metrics-process-ready.json"
        self.stop_path = self.directory / f"{stem}-metrics-process-stop.json"
        self.report_path = self.directory / f"{stem}-metrics-iteration.json"
        self.process = None
        self.stderr_stream = None

    async def start(self):
        for path in (self.ready_path, self.stop_path, self.report_path):
            if path.exists() or path.is_symlink():
                raise ValueError(f"Refusing to reuse metrics process artifacts: {path}")
        self.stderr_stream = (self.directory / f"{self.stem}-metrics-process.log").open("xb")
        environment = os.environ.copy()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        try:
            self.process = await asyncio.create_subprocess_exec(
                sys.executable,
                str(self.worker_script),
                "--url",
                self.url,
                "--directory",
                str(self.directory),
                "--stem",
                self.stem,
                "--model",
                self.model,
                "--dp-size",
                str(self.dp_size),
                "--parent-pid",
                str(os.getpid()),
                stdout=self.stderr_stream,
                stderr=self.stderr_stream,
                env=environment,
            )
            deadline = time.perf_counter() + 30.0
            while not self.ready_path.exists():
                if self.process.returncode is not None:
                    raise RuntimeError(f"Metrics worker exited before before-scrape: {self.process.returncode}")
                if time.perf_counter() >= deadline:
                    raise TimeoutError("Metrics worker did not complete before-scrape within 30s")
                await asyncio.sleep(0.05)
            ready = json.loads(self.ready_path.read_text())
            if ready.get("pid") != self.process.pid or ready.get("phase") != "before":
                raise RuntimeError("Invalid metrics worker ready identity")
        except BaseException:
            await self._stop_process("startup_failed")
            raise

    async def _stop_process(self, status):
        try:
            if self.process is not None:
                if self.process.returncode is None:
                    temporary = self.stop_path.with_suffix(".tmp")
                    temporary.write_text(json.dumps({"iteration_status": status}) + "\n", encoding="utf-8")
                    temporary.replace(self.stop_path)
                    try:
                        await asyncio.wait_for(self.process.wait(), timeout=30.0)
                    except TimeoutError:
                        self.process.kill()
                        await self.process.wait()
                        raise TimeoutError("Metrics worker failed to finish; acceptance must remain invalid")
        finally:
            if self.stderr_stream is not None:
                self.stderr_stream.close()
                self.stderr_stream = None

    async def finish(self, iteration_status):
        if self.process is None:
            raise RuntimeError("Metrics process was not started")
        await self._stop_process(iteration_status)
        if self.process.returncode != 0 or not self.report_path.is_file():
            raise RuntimeError(f"Metrics process failed (exit {self.process.returncode}); no valid acceptance report")
        report = json.loads(self.report_path.read_text())
        if (
            report.get("model_name") != self.model
            or report.get("expected_dp_size") != self.dp_size
            or report.get("iteration_status") != iteration_status
            or report.get("collector", {}).get("pid") != self.process.pid
        ):
            raise RuntimeError("Metrics process report identity/status mismatch")
        # Read-back binds every returned observation to the original raw body.
        for index, row in enumerate(report["snapshots"]):
            expected = f"{self.stem}-metrics-{index:04d}-{row['phase']}"
            if row.get("sample_index") != index or (
                row.get("raw_file") != expected + ".txt" or row.get("json_file") != expected + ".json"
            ):
                raise RuntimeError("Metrics process snapshot identity mismatch")
            raw = (self.directory / row["raw_file"]).read_bytes()
            stored = json.loads((self.directory / row["json_file"]).read_text())
            if hashlib.sha256(raw).hexdigest() != row["raw_sha256"] or stored != row:
                raise RuntimeError("Metrics process raw/JSON integrity mismatch")
            if row["valid"] and (
                row.get("http_status") != 200 or parse_metrics(raw.decode("utf-8")) != row.get("parsed")
            ):
                raise RuntimeError("Metrics process parsed evidence mismatch")
        verified = acceptance_delta(report["snapshots"], self.model, self.dp_size)
        if iteration_status != "completed":
            verified["valid"] = False
            verified["ratio"] = None
            verified["errors"].append(f"Iteration {iteration_status}")
        if verified != report["acceptance"]:
            raise RuntimeError("Metrics process acceptance does not match raw counters")
        return report
