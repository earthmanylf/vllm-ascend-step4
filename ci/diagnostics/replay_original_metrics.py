"""Replay untouched original aggregation on recorded SSE timing, without network."""

import argparse
import ast
import asyncio
import contextlib
import copy
import hashlib
import io
import json
import math
import statistics
from pathlib import Path
from types import SimpleNamespace

parser = argparse.ArgumentParser()
parser.add_argument("--original", type=Path, required=True)
parser.add_argument("--run", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
tree = ast.parse(args.original.read_text())
client = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "VLLMBenchmarkClient")
method = next(n for n in client.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "test_streaming_benchmark")
percentile = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_percentile")
error = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "BenchmarkQualityError")
ns = {"asyncio": asyncio, "statistics": statistics, "math": math, "Any": object}
exec(
    compile(
        ast.fix_missing_locations(ast.Module(body=[percentile, error, method], type_ignores=[])),
        str(args.original),
        "exec",
    ),
    ns,
)
original_method = ns["test_streaming_benchmark"]
summary = json.loads((args.run / "summary.json").read_text())
rows = []


async def replay():
    for p in sorted(args.run.glob("*-measured-iteration.json")):
        j = json.loads(p.read_text())
        requests = j["requests"]
        bs = len(requests)
        # Restore the original client's last-choice timestamp (possibly finish)
        # instead of the safe-copy last-nonempty-token timestamp.
        orig_records = []
        finish_deltas = []
        for r in requests:
            matches = list(args.run.glob(p.name.removesuffix("-iteration.json") + "-" + r["request_id"] + "-*.json"))
            assert len(matches) == 1, (p.name, r["request_id"], matches)
            raw = json.loads(matches[0].read_text())
            assert raw["status"] == "completed" and raw["done"]
            original = copy.deepcopy(r)
            last_choice_timestamp = raw["chunks"][-1]["absolute_timestamp"]
            finish_deltas.append(last_choice_timestamp - r["last_token_timestamp"])
            original["last_token_timestamp"] = last_choice_timestamp
            original["is_complete"] = r["completion_tokens"] >= r["expected_tokens"]
            orig_records.append(original)
        lookup = {r["request_id"]: r for r in orig_records}

        async def generate(prompt, max_tokens, request_id, lookup=lookup):
            return copy.deepcopy(lookup[request_id])

        fake = SimpleNamespace(generate_completion_stream=generate, weight_version=None)
        with contextlib.redirect_stdout(io.StringIO()):
            result = await original_method(fake, [[0] for _ in requests], requests[0]["expected_tokens"])
        metrics = j["metrics"]
        first = max(r["first_token_timestamp"] for r in requests)
        last = min(r["last_token_timestamp"] for r in requests)
        rows.append(
            {
                "artifact": p.name,
                "batch_size": bs,
                "original_replayed_prefill_tps": result["prefill_tps"],
                "recorded_prefill_tps": metrics["prefill_tps"],
                "original_replayed_decode_tps": result["decode_tps_e2e_window"],
                "recorded_decode_tps": metrics["decode_tps_e2e_window"],
                "relative_decode_difference": result["decode_tps_e2e_window"] / metrics["decode_tps_e2e_window"] - 1,
                "original_replayed_itl_ms": result["itl_avg"] * 1000,
                "recorded_itl_ms": metrics["itl_avg"] * 1000,
                "last_first_token_minus_first_request_seconds": first
                - min(r["request_start_timestamp"] for r in requests),
                "actual_output_window_seconds": last - first,
                "all_requests_first_token_at_or_before_window_start": all(
                    r["first_token_timestamp"] <= first for r in requests
                ),
                "all_requests_last_output_at_or_after_window_end": all(
                    r["last_token_timestamp"] >= last for r in requests
                ),
                "max_finish_vs_last_actual_token_delta_seconds": max(finish_deltas),
                "min_finish_vs_last_actual_token_delta_seconds": min(finish_deltas),
                "window_token_coverage": metrics["decode_window_token_coverage"],
            }
        )


asyncio.run(replay())
aggregate = []
for bs in (1, 2, 4, 8, 16, 32, 64):
    selected = [r for r in rows if r["batch_size"] == bs]
    assert len(selected) == 3
    key = f"bs{bs}_input131072_output2048"
    record = summary[key]
    aggregate.append(
        {
            "batch_size": bs,
            "original_replayed_prefill_tps_mean": statistics.mean(r["original_replayed_prefill_tps"] for r in selected),
            "table_prefill_tps": record["prefill_tps"],
            "original_replayed_decode_tps_mean": statistics.mean(r["original_replayed_decode_tps"] for r in selected),
            "table_decode_tps": record["decode_tps_e2e_window"],
            "original_replayed_itl_ms_mean": statistics.mean(r["original_replayed_itl_ms"] for r in selected),
            "table_itl_ms": record["itl_avg"] * 1000,
        }
    )
report = {
    "original_filename": args.original.name,
    "original_sha256": hashlib.sha256(args.original.read_bytes()).hexdigest(),
    "run_identity": json.loads((args.run / "run.json").read_text()),
    "scope": (
        "Offline replay of original metric method on recorded request timing; "
        "not a fresh execution of untouched original client"
    ),
    "prefill_metric": "sum(input_tokens)/max(TTFT), including client/queue/mixed-execution effects",
    "decode_window": "(max(all first actual token timestamps), min(all last timestamps)]",
    "interpretation": (
        "Excludes the measured requests initial TTFT/prefill interval, "
        "but client timestamps do not prove worker/kernel-only decode or absence of other engine work"
    ),
    "rows": rows,
    "aggregate": aggregate,
}
args.output.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(aggregate, indent=2))
print("MAX_ABS_RELATIVE_DECODE_DELTA", max(abs(r["relative_decode_difference"]) for r in rows))
