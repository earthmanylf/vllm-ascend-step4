"""Offline synthetic-response tests: no sockets, no GPU probes, no benchmark."""

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

# aiohttp is not installed locally; transport is deliberately a fake module.
sys.modules.setdefault("aiohttp", types.ModuleType("aiohttp"))
sys.path.insert(0, str(Path(__file__).parent))
import benchmark_pd_tps as bench  # noqa: E402 -- transport stub must be installed before import


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


if __name__ == "__main__":
    unittest.main()
