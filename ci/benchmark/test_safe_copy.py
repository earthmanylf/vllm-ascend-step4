"""Offline synthetic-response tests: no sockets, no GPU probes, no benchmark."""

import asyncio
import json
import os
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

# aiohttp is not installed locally; transport is deliberately a fake module.
sys.modules.setdefault("aiohttp", types.ModuleType("aiohttp"))
sys.path.insert(0, str(Path(__file__).parent))
import benchmark_pd_tps as bench  # noqa: E402 - install fake transport before importing the client
import metrics_capture as metrics  # noqa: E402 - imports share the synthetic transport
import telemetry_worker  # noqa: E402 - imports share the synthetic transport


class FakeResponse:
    status = 200
    headers = {"Content-Type": "text/event-stream"}

    def __init__(self, parts):
        self.parts = parts
        self.content = self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def iter_chunked(self, size):
        for part in self.parts:
            yield part


class FakeSession:
    def __init__(self, parts):
        self.parts = parts

    def post(self, url, **kwargs):
        self.payload = kwargs["json"]
        return FakeResponse(self.parts)


def event(value):
    return ("data: " + json.dumps(value) + "\n\n").encode()


def stream(token_ids=(1, 2, 3), count=3, done=True, finish="length"):
    parts = [
        event({"choices": [{"text": "abc", "token_ids": list(token_ids)}]}),
        event({"choices": [{"text": "", "token_ids": [], "finish_reason": finish}]}),
        event({"choices": [], "usage": {"completion_tokens": count}}),
    ]
    if done:
        parts.append(b"data: [DONE]\n\n")
    return parts


class SafetyTests(unittest.IsolatedAsyncioTestCase):
    async def request(self, parts, expected=3):
        with tempfile.TemporaryDirectory() as directory:
            client = bench.VLLMBenchmarkClient("unused", 1, "step5", raw_artifacts_dir=directory)
            client.session = FakeSession(parts)
            try:
                result = await client.generate_completion_stream([9, 8], expected, request_id="req_0")
            finally:
                records = list(Path(directory).glob("*.json"))
                self.assertEqual(len(records), 1)
                record = json.loads(records[0].read_text())
                self.assertEqual(next(Path(directory).glob("*.sse")).read_bytes(), b"".join(parts))
                self.assertEqual(record["payload"]["min_tokens"], expected)
                self.assertEqual(record["payload"]["max_tokens"], expected)
                self.assertEqual(record["payload"]["model"], "step5")
                self.assertTrue(record["payload"]["ignore_eos"])
                self.last_record = record
            return result

    async def test_multi_token_count_and_actual_last_token(self):
        result = await self.request(stream())
        self.assertEqual(result["completion_tokens"], 3)
        self.assertEqual(result["output_tokens"], 1)  # Original event-count field retained.
        self.assertEqual(result["last_token_timestamp"], result["token_data"][-1]["absolute_timestamp"])
        self.assertLess(result["last_token_timestamp"], result["finish_events"][0]["timestamp"])
        self.assertEqual(len(self.last_record["reads"]), 4)

    async def test_exact_count_rejects_under_and_over(self):
        for ids in [(1,), (1, 2, 3, 4)]:
            with self.assertRaisesRegex(RuntimeError, "Exact output"):
                await self.request(stream(ids, len(ids)))
            self.assertEqual(self.last_record["status"], "failed")

    async def test_done_finish_usage_required(self):
        for parts, message in [
            (stream(done=False), "without"),
            (stream(finish="stop"), "finish_reason"),
            (stream(count=4), "disagrees"),
        ]:
            with self.assertRaisesRegex(RuntimeError, message):
                await self.request(parts)

    async def test_malformed_and_error_events(self):
        for parts in [[b"data: {bad}\n\n"], [event({"error": "failure"})], [b"event: error\n\n"]]:
            with self.assertRaises(RuntimeError):
                await self.request(parts)
            self.assertEqual(self.last_record["status"], "failed")

    async def test_missing_token_ids_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "token_ids"):
            await self.request([event({"choices": [{"text": "x", "finish_reason": "length"}]}), b"data: [DONE]\n\n"])

    async def test_all_skipped_empty_unvalidated_rejected(self):
        for results in [
            {},
            {"bs1": {"status": "skipped"}},
            {"bs1": {"status": "completed", "decode_window_quality": {"valid": True}}},
        ]:
            with self.assertRaises(RuntimeError):
                bench.require_completed_results(results)

    async def test_capacity_no_guesses(self):
        with self.assertRaises(ValueError):
            bench.load_capacity_metadata(None, 8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capacity.json"
            valid = {
                "per_dp_capacity_tokens": {str(i): 100000 for i in range(8)},
                "max_model_len": 1048576,
                "source": "synthetic test only",
            }
            path.write_text(json.dumps(valid))
            self.assertEqual(bench.load_capacity_metadata(str(path), 8), valid)
            valid["per_dp_capacity_tokens"]["0"] = True
            path.write_text(json.dumps(valid))
            with self.assertRaises(ValueError):
                bench.load_capacity_metadata(str(path), 8)

    async def test_forbidden_cleanup_retries_weight_mutation(self):
        for extra in [
            {"failure_cleanup_mode": "abort"},
            {"failure_cleanup_mode": "off", "config_retries": 1},
            {"failure_cleanup_mode": "off", "weight_version": 1},
        ]:
            with self.assertRaises(ValueError):
                await bench.run_benchmark_suite("unused", 1, "step5", [1], [1], [2048], 16384, **extra)


def metrics_text(draft=100, accepted=50):
    rows = ["# TYPE vllm:spec_decode_num_draft_tokens counter"]
    for engine in range(8):
        labels = f'engine="{engine}",model_name="step5"'
        for name, value in (
            (metrics.DRAFT, draft),
            (metrics.ACCEPTED, accepted),
            (metrics.SCHEDULER["running"], engine),
            (metrics.SCHEDULER["waiting"], 0),
            (metrics.SCHEDULER["preemptions"], 3),
            (metrics.SCHEDULER["kv_usage"], 0.25),
        ):
            rows.append(f"{name}{{{labels}}} {value}")
    return "\n".join(rows) + "\n"


def metric_snapshot(phase, draft=100, accepted=50):
    return {"phase": phase, "valid": True, "parsed": metrics.parse_metrics(metrics_text(draft, accepted))}


class MetricsTests(unittest.TestCase):
    def test_parses_all_eight_dp_and_scheduler_observations(self):
        parsed = metrics.parse_metrics(metrics_text() + "unrelated_metric 999\n")
        self.assertEqual(parsed["errors"], [])
        observation = metrics.snapshot_observations(parsed, "step5", 8)
        self.assertEqual(observation["running"]["all_dp_sum"], 28)
        self.assertEqual(observation["preemptions"]["all_dp_sum"], 24)
        self.assertEqual(observation["kv_usage"]["per_dp"], {str(i): 0.25 for i in range(8)})
        self.assertNotIn("all_dp_sum", observation["kv_usage"])
        result = metrics.acceptance_delta([metric_snapshot("before"), metric_snapshot("after", 110, 56)], "step5", 8)
        self.assertTrue(result["valid"])
        self.assertEqual((result["draft_tokens"], result["accepted_tokens"], result["ratio"]), (80, 48, 0.6))

    def test_labels_escapes_order_scientific_counters_and_exemplars(self):
        parsed = metrics.parse_metrics(
            metrics.DRAFT + '{model_name="step5",engine="0",note="a\\\\b\\n\\"c"} 1.2e2 123 # {trace_id="sample"} 1\n'
        )
        self.assertEqual(parsed["errors"], [])
        self.assertEqual(parsed["series"][0]["value"], 120)
        self.assertEqual(parsed["series"][0]["labels"]["note"], 'a\\b\n"c')
        self.assertEqual(metrics.parse_labels('engine="0", model_name="step5"'), {"engine": "0", "model_name": "step5"})
        # Decimal parsing avoids float rounding for cumulative integer counters.
        large = metrics.parse_metrics(metrics.DRAFT + " 9007199254740993\n")
        self.assertEqual(large["series"][0]["value"], 9007199254740993)

    def test_missing_duplicate_or_aggregate_only_engine_series_invalid(self):
        original = metrics_text()
        invalid = [
            original.replace('engine="7"', 'engine="8"'),
            original.replace('engine="7",', ""),
            original.replace(',model_name="step5"', ""),
            original + original.splitlines()[1] + "\n",
            f'{metrics.DRAFT}{{model_name="step5"}} 800\n{metrics.ACCEPTED}{{model_name="step5"}} 400\n',
            "",
        ]
        for text in invalid:
            with self.subTest(text=text[:80]):
                after = {"phase": "after", "valid": True, "parsed": metrics.parse_metrics(text)}
                result = metrics.acceptance_delta([metric_snapshot("before"), after], "step5", 8)
                self.assertFalse(result["valid"])
                self.assertIsNone(result["ratio"])

    def test_wrong_model_is_not_aggregated_and_replica_duplicates_invalid(self):
        parsed = metrics.parse_metrics(
            metrics_text() + metrics_text(999, 888).replace('model_name="step5"', 'model_name="other"')
        )
        result = metrics.metric_by_dp(parsed, metrics.DRAFT, "step5", 8)
        self.assertTrue(result["valid"])
        self.assertEqual(sum(result["per_dp"].values()), 800)
        duplicate = metrics_text() + (f'{metrics.DRAFT}{{engine="0",model_name="step5",tp_rank="1"}} 100\n')
        result = metrics.metric_by_dp(metrics.parse_metrics(duplicate), metrics.DRAFT, "step5", 8)
        self.assertFalse(result["valid"])

    def test_malformed_nonfinite_negative_and_fractional_counters_invalid(self):
        for replacement in ("NaN", "+Inf", "-1", "1.5", "bad"):
            with self.subTest(value=replacement):
                text = metrics_text().replace("} 100\n", "} " + replacement + "\n", 1)
                after = {"phase": "after", "valid": True, "parsed": metrics.parse_metrics(text)}
                result = metrics.acceptance_delta([metric_snapshot("before"), after], "step5", 8)
                self.assertFalse(result["valid"])
        text = metrics_text().replace('engine="0",model_name="step5"', 'engine="0",engine="0"')
        self.assertTrue(metrics.parse_metrics(text)["errors"])

    def test_observed_intermediate_reset_invalid_even_if_final_counter_recovers(self):
        result = metrics.acceptance_delta(
            [
                metric_snapshot("before"),
                metric_snapshot("during", 10, 5),
                metric_snapshot("after", 110, 56),
            ],
            "step5",
            8,
        )
        self.assertFalse(result["valid"])
        self.assertTrue(any("reset/decrease" in error for error in result["errors"]))
        self.assertIsNone(result["ratio"])

    def test_zero_drafts_accepted_excess_identity_change_and_scrape_failure_invalid(self):
        invalid = [
            [metric_snapshot("before"), metric_snapshot("after")],
            [metric_snapshot("before"), metric_snapshot("after", 101, 52)],
            [metric_snapshot("before"), {"phase": "during", "valid": False}, metric_snapshot("after", 110, 56)],
            [metric_snapshot("after", 110, 56)],
        ]
        changed = metric_snapshot("after", 110, 56)
        changed["parsed"]["series"][0]["labels"]["worker"] = "replacement"
        invalid.append([metric_snapshot("before"), changed])
        for snapshots in invalid:
            with self.subTest(snapshots=snapshots):
                result = metrics.acceptance_delta(snapshots, "step5", 8)
                self.assertFalse(result["valid"])
                self.assertIsNone(result["ratio"])

    def test_weighted_ratio_and_invalid_iteration_never_silently_filtered(self):
        rows = [
            {"valid": True, "draft_tokens": 10, "accepted_tokens": 9, "ratio": 0.9},
            {"valid": True, "draft_tokens": 90, "accepted_tokens": 9, "ratio": 0.1},
        ]
        result = metrics.weighted_acceptance(rows)
        self.assertTrue(result["valid"])
        self.assertEqual(result["ratio"], 0.18)
        result = metrics.weighted_acceptance(rows + [{"valid": False}])
        self.assertFalse(result["valid"])
        self.assertEqual(result["invalid_iterations"], [3])
        self.assertIsNone(result["ratio"])

    def test_scheduler_summary_retains_all_dp_and_invalidates_preemption_reset(self):
        snapshots = [metric_snapshot("before"), metric_snapshot("after", 110, 56)]
        for row in snapshots[1]["parsed"]["series"]:
            if row["name"] == metrics.SCHEDULER["running"] and row["labels"]["engine"] == "0":
                row["value"] = 16
            if row["name"] == metrics.SCHEDULER["preemptions"]:
                row["value"] = 4
        result = metrics.scheduler_summary(snapshots, "step5", 8)
        self.assertEqual(result["running"]["per_dp"]["0"], {"observed_min": 0, "observed_max": 16})
        self.assertEqual(result["preemptions"]["delta"], 8)
        for row in snapshots[1]["parsed"]["series"]:
            if row["name"] == metrics.SCHEDULER["preemptions"] and row["labels"]["engine"] == "7":
                row["value"] = 0
        result = metrics.scheduler_summary(snapshots, "step5", 8)
        self.assertFalse(result["preemptions"]["valid"])
        self.assertIsNone(result["preemptions"]["delta"])
        result = metrics.scheduler_summary(snapshots + [{"valid": False}], "step5", 8)
        self.assertFalse(result["kv_usage"]["valid"])
        self.assertEqual(result["kv_usage"]["invalid_snapshot_indices"], [2])


class MetricsCaptureTests(unittest.IsolatedAsyncioTestCase):
    async def test_measured_iteration_integration_preserves_tps_and_weighted_acceptance(self):
        counts = iter([(100, 50), (110, 59), (110, 59), (200, 68)])

        async def fetch():
            draft, accepted = next(counts)
            return 200, {}, metrics_text(draft, accepted).encode()

        with tempfile.TemporaryDirectory() as directory:
            client = bench.VLLMBenchmarkClient("unused", 1, "step5", raw_artifacts_dir=directory)
            client.create_metrics_capture = lambda stem: metrics.MetricsCapture(fetch, directory, stem, "step5", 8)

            async def streaming(prompts, output_len):
                client.batch_sequence += 1
                result = bench.TpsMetric(
                    batch_size=len(prompts), input_len=len(prompts[0]), output_len=output_len
                ).to_dict()
                result.update(
                    prefill_tps=10 * client.batch_sequence,
                    decode_tps_e2e_window=30 * client.batch_sequence,
                )
                return result

            client.test_streaming_benchmark = streaming
            with patch.object(bench.asyncio, "sleep", new=AsyncMock()):
                result = await bench.run_single_benchmark(client, 1, 2, 3, num_iters=2, vocab_size=100)
            self.assertEqual(result.prefill_tps, 15)
            self.assertEqual(result.decode_tps_e2e_window, 45)
            self.assertEqual(result.spec_decode_acceptance["ratio"], 0.18)
            self.assertEqual(result.spec_decode_acceptance["draft_tokens"], 800)
            self.assertTrue(result.metrics_scheduler_summary["running"]["valid"])
            self.assertEqual(len(result.metrics_iterations), 2)
            self.assertEqual(len(list(Path(directory).glob("*-metrics-iteration.json"))), 2)

    async def test_raw_scrapes_before_during_after_and_json_are_persisted(self):
        calls = []

        async def fetch():
            calls.append(len(calls))
            return (
                200,
                {"Content-Type": "text/plain"},
                metrics_text(100 + 10 * len(calls), 50 + 5 * len(calls)).encode(),
            )

        with tempfile.TemporaryDirectory() as directory:
            capture = metrics.MetricsCapture(fetch, directory, "measured", "step5", 8)
            await capture.start()
            # Allow the actual nominal 1s polling task to collect a during sample.
            await asyncio.sleep(1.05)
            report = await capture.finish("completed")
            self.assertEqual([row["phase"] for row in capture.snapshots], ["before", "during", "after"])
            self.assertTrue(report["acceptance"]["valid"])
            self.assertEqual(report["acceptance"]["ratio"], 0.5)
            self.assertEqual(len(calls), 3)
            self.assertEqual(report["sampling"]["during_snapshot_count"], 1)
            self.assertTrue(report["scheduler"]["kv_usage"]["valid"])
            for row in capture.snapshots:
                raw = (Path(directory) / row["raw_file"]).read_bytes()
                stored = json.loads((Path(directory) / row["json_file"]).read_text())
                self.assertEqual(stored, row)
                self.assertEqual(
                    raw, metrics_text(110 + 10 * row["sample_index"], 55 + 5 * row["sample_index"]).encode()
                )
            self.assertTrue((Path(directory) / "measured-metrics-iteration.json").exists())

    async def test_http_or_transport_failure_invalid_no_retry(self):
        calls = []

        async def fetch():
            calls.append(1)
            if len(calls) == 1:
                return 503, {}, b"metrics unavailable"
            raise RuntimeError("connection failed")

        with tempfile.TemporaryDirectory() as directory:
            capture = metrics.MetricsCapture(fetch, directory, "failure", "step5", 8)
            await capture.start()
            report = await capture.finish("failed")
            self.assertEqual(len(calls), 2)
            self.assertFalse(report["acceptance"]["valid"])
            self.assertIsNone(report["acceptance"]["ratio"])
            self.assertEqual((Path(directory) / capture.snapshots[0]["raw_file"]).read_bytes(), b"metrics unavailable")
            self.assertEqual((Path(directory) / capture.snapshots[1]["raw_file"]).read_bytes(), b"")


class ProcessTelemetryTests(unittest.IsolatedAsyncioTestCase):
    def fixture_worker(self, directory, failed=False):
        """Real child interpreter with synthetic transport; never opens sockets."""
        module_dir = str(Path(metrics.__file__).resolve().parent)
        script = Path(directory) / "synthetic_worker.py"
        script.write_text(
            "import sys\n"
            f"sys.path.insert(0, {module_dir!r})\n"
            "import telemetry_worker\n"
            f"TEXT = {metrics_text()!r}\n"
            f"FAILED = {failed!r}\n"
            "class SyntheticFetcher:\n"
            "    def __init__(self, url): self.calls = 0\n"
            "    async def __call__(self):\n"
            "        self.calls += 1\n"
            "        if FAILED and self.calls == 2:\n"
            "            return 503, {}, b'synthetic unavailable'\n"
            "        text = TEXT.replace('} 100\\n', '} '+str(100+10*self.calls)+'\\n')\n"
            "        text = text.replace('} 50\\n', '} '+str(50+5*self.calls)+'\\n')\n"
            "        return 200, {'Content-Type': 'text/plain'}, text.encode()\n"
            "telemetry_worker.HTTPMetricsFetcher = SyntheticFetcher\n"
            "telemetry_worker.main()\n",
            encoding="utf-8",
        )
        return script

    async def test_process_keeps_polling_while_parent_event_loop_and_gil_are_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            script = self.fixture_worker(directory)
            capture = metrics.ProcessMetricsCapture(
                "http://unused/metrics", directory, "process-blocked", "step5", 8, worker_script=script
            )
            await capture.start()
            started = time.perf_counter()
            started_wall = time.time()
            # Deliberately no await: mimic large JSON dumps/client flush with
            # a parent GIL-bound loop, not just an asynchronously sleeping task.
            iterations = 0
            while time.perf_counter() - started < 2.25:
                json.dumps({"rows": list(range(200)), "payload": "x" * 1024})
                iterations += 1
            ended_wall = time.time()
            report = await capture.finish("completed")
            self.assertGreater(iterations, 0)
            during = [row for row in report["snapshots"] if row["phase"] == "during"]
            self.assertGreaterEqual(len(during), 2)
            self.assertGreaterEqual(sum(started_wall < row["started_wall_time_unix"] < ended_wall for row in during), 2)
            self.assertLess(report["sampling"]["max_start_gap_seconds"], 1.7)
            self.assertTrue(report["acceptance"]["valid"])
            self.assertEqual(report["acceptance"]["ratio"], 0.5)
            self.assertEqual(report["collector"]["mode"], "independent-python-process")
            self.assertNotEqual(report["collector"]["pid"], report["collector"]["parent_pid"])
            self.assertEqual(report["collector"]["http_timeout_seconds"], 5)
            self.assertEqual([report["snapshots"][0]["phase"], report["snapshots"][-1]["phase"]], ["before", "after"])
            self.assertGreaterEqual(report["snapshots"][-1]["started_wall_time_unix"], ended_wall)

    async def test_abnormal_child_exit_does_not_publish_valid_report(self):
        with tempfile.TemporaryDirectory() as directory:
            script = self.fixture_worker(directory)
            capture = metrics.ProcessMetricsCapture(
                "http://unused/metrics", directory, "process-exit", "step5", 8, worker_script=script
            )
            await capture.start()
            capture.process.kill()
            await asyncio.wait_for(capture.process.wait(), timeout=5.0)
            with self.assertRaisesRegex(RuntimeError, "Metrics process failed"):
                await capture.finish("completed")
            self.assertFalse(capture.report_path.exists())
            self.assertIsNone(capture.stderr_stream)
            self.assertTrue(capture.ready_path.exists())

    async def test_parent_exit_stops_real_orphan_worker_without_success_report(self):
        with tempfile.TemporaryDirectory() as directory:
            script = self.fixture_worker(directory)
            module_dir = str(Path(metrics.__file__).resolve().parent)
            parent_script = Path(directory) / "synthetic_parent.py"
            parent_script.write_text(
                "import asyncio, os, sys\n"
                f"sys.path.insert(0, {module_dir!r})\n"
                "from metrics_capture import ProcessMetricsCapture\n"
                "async def run():\n"
                f"    capture = ProcessMetricsCapture('http://unused/metrics', {directory!r}, "
                f"'orphan', 'step5', 8, worker_script={str(script)!r})\n"
                "    await capture.start()\n"
                "    os._exit(0)\n"
                "asyncio.run(run())\n",
                encoding="utf-8",
            )
            parent = await asyncio.create_subprocess_exec(
                sys.executable,
                str(parent_script),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
                env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
            )
            await asyncio.wait_for(parent.wait(), timeout=8.0)
            self.assertEqual(parent.returncode, 0)
            ready_path = Path(directory) / "orphan-metrics-process-ready.json"
            self.assertTrue(ready_path.is_file())
            ready = json.loads(ready_path.read_text())
            self.assertEqual(ready["parent_pid"], parent.pid)
            log = Path(directory) / "orphan-metrics-process.log"
            deadline = time.perf_counter() + 5.0
            while "Telemetry parent exited" not in log.read_text() and time.perf_counter() < deadline:
                await asyncio.sleep(0.05)
            self.assertIn("Telemetry parent exited", log.read_text())
            self.assertFalse((Path(directory) / "orphan-metrics-iteration.json").exists())
            samples = list(Path(directory).glob("orphan-metrics-*-during.json"))
            await asyncio.sleep(1.1)
            self.assertEqual(samples, list(Path(directory).glob("orphan-metrics-*-during.json")))

    async def test_raw_artifact_tamper_is_not_accepted_as_valid_process_report(self):
        with tempfile.TemporaryDirectory() as directory:
            script = self.fixture_worker(directory)
            capture = metrics.ProcessMetricsCapture(
                "http://unused/metrics", directory, "process-tamper", "step5", 8, worker_script=script
            )
            await capture.start()
            before = Path(directory) / "process-tamper-metrics-0000-before.txt"
            before.write_bytes(b"tampered after scrape")
            with self.assertRaisesRegex(RuntimeError, "raw/JSON integrity mismatch"):
                await capture.finish("completed")
            self.assertEqual(capture.process.returncode, 0)
            self.assertIsNone(capture.stderr_stream)

    async def test_child_failed_sample_remains_invalid_no_retry_or_filtering(self):
        with tempfile.TemporaryDirectory() as directory:
            script = self.fixture_worker(directory, failed=True)
            capture = metrics.ProcessMetricsCapture(
                "http://unused/metrics", directory, "process-failed", "step5", 8, worker_script=script
            )
            await capture.start()
            await asyncio.sleep(1.15)
            report = await capture.finish("completed")
            self.assertFalse(report["acceptance"]["valid"])
            self.assertIsNone(report["acceptance"]["ratio"])
            self.assertEqual(report["sampling"]["invalid_snapshot_indices"], [1])
            self.assertEqual(report["snapshots"][1]["http_status"], 503)
            self.assertEqual(report["snapshots"][-1]["phase"], "after")
            self.assertFalse(metrics.weighted_acceptance([report["acceptance"]])["valid"])

    async def test_child_startup_failure_is_not_reported_as_valid_telemetry(self):
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "failed_worker.py"
            script.write_text("raise RuntimeError('synthetic startup failure')\n")
            capture = metrics.ProcessMetricsCapture(
                "http://unused/metrics", directory, "bad-start", "step5", 8, worker_script=script
            )
            with self.assertRaisesRegex(RuntimeError, "before before-scrape"):
                await capture.start()
            self.assertFalse(capture.report_path.exists())
            self.assertIsNotNone(capture.process.returncode)
            self.assertIsNone(capture.stderr_stream)

    async def test_http_fetcher_uses_real_body_timeout_no_redirect_or_proxy(self):
        fetch = telemetry_worker.HTTPMetricsFetcher("http://unused/metrics")
        response = unittest.mock.MagicMock()
        response.status = 200
        response.headers = {"Content-Type": "text/plain"}
        response.read.return_value = metrics_text().encode()
        response.__enter__.return_value = response
        with patch.object(fetch.opener, "open", return_value=response) as opened:
            status, headers, raw = await fetch()
        self.assertEqual(status, 200)
        self.assertEqual(raw, metrics_text().encode())
        self.assertEqual(headers["Content-Type"], "text/plain")
        self.assertEqual(opened.call_args.kwargs["timeout"], 5.0)
        request = opened.call_args.args[0]
        self.assertEqual(request.get_method(), "GET")
        self.assertEqual(request.get_header("Connection"), "close")
        self.assertIsNone(telemetry_worker.NoRedirect().redirect_request(None, None, 302, "", {}, "http://other"))
        with self.assertRaisesRegex(ValueError, "direct HTTP"):
            telemetry_worker.HTTPMetricsFetcher("http://unused/pause")


if __name__ == "__main__":
    unittest.main()
