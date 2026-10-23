"""Pause/admit/resume fixed DP batches and switch only draft mode in one process."""

import argparse
import concurrent.futures
import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True, type=Path)
parser.add_argument("--size", type=int, default=16)
parser.add_argument("--phases", default="full-a,full-b,eager-a,eager-b,full-c")
parser.add_argument("--tokens", type=int, default=32)
parser.add_argument("--trace", choices=("true", "false"), default="true")
parser.add_argument("--ordered-waves", action="store_true")
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
inputs = [[100 + (i * 17 + j) % 1000 for j in range(1024)] for i in range(args.size)]


def call(path, payload=None, headers=None, timeout=240):
    request = urllib.request.Request(
        "http://127.0.0.1:19082" + path,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=timeout) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"HTTP {error.code}: {error.read().decode()}") from error


def request(index, phase, barrier):
    payload = {
        "model": "step5",
        "prompt": inputs[index],
        "temperature": 0,
        "seed": 0,
        "max_tokens": args.tokens,
        "min_tokens": args.tokens,
        "ignore_eos": True,
        "return_token_ids": True,
        "return_tokens_as_token_ids": True,
        "logprobs": 5,
        "stream": False,
        "request_id": f"controlled-{phase}-{index}",
    }
    barrier.wait()
    if args.ordered_waves:
        # Preserve request-row order within each DP; traces must still verify
        # the actual ordered first batch rather than trusting this delay.
        time.sleep((index // 8) * 0.75)
    status, response = call("/v1/completions", payload, {"X-data-parallel-rank": str(index % 8)})
    row = {"index": index, "phase": phase, "dp": index % 8, "status": status, "response": response}
    (args.output / f"{phase}-request{index}.json").write_text(json.dumps(row, indent=2))
    choice = response["choices"][0]
    assert len(choice["token_ids"]) == args.tokens and choice["finish_reason"] == "length"
    return row


phases = {}
for phase in args.phases.split(","):
    mode = "eager" if phase.startswith("eager") else "full"
    _, configuration = call(
        "/collective_rpc",
        {
            "method": "controlled_mtp_configure",
            "args": [mode, phase, args.trace],
            "timeout": 30,
        },
    )
    (args.output / f"{phase}-configuration.json").write_text(json.dumps(configuration, indent=2))
    status, paused = call("/pause?mode=keep&clear_cache=false", {})
    assert status == 200 and paused["status"] == "paused"
    try:
        barrier = threading.Barrier(args.size + 1)
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.size) as pool:
            futures = [pool.submit(request, index, phase, barrier) for index in range(args.size)]
            barrier.wait()
            time.sleep(3)
            status, resumed = call("/resume", {})
            assert status == 200
            phases[phase] = [future.result(timeout=240) for future in futures]
    finally:
        call("/resume", {})
    print("PHASE_COMPLETED", phase, flush=True)

comparison = []
names = list(phases)
pairs = [(names[0], phase) for phase in names[1:]]
for prefix in ("eager", "full"):
    matching = [name for name in names if name.startswith(prefix)]
    pairs += [(a, b) for a, b in zip(matching, matching[1:]) if (a, b) not in pairs]
for baseline, phase in pairs:
    results = phases[phase]
    rows = []
    for before, after in zip(phases[baseline], results):
        a, b = before["response"]["choices"][0], after["response"]["choices"][0]
        differences = [i for i, (x, y) in enumerate(zip(a["token_ids"], b["token_ids"])) if x != y]
        row = {
            "index": before["index"],
            "exact": not differences,
            "first_difference": differences[0] if differences else None,
        }
        if differences:
            index = differences[0]
            row["baseline_top_logprobs"] = a["logprobs"]["top_logprobs"][index]
            row["phase_top_logprobs"] = b["logprobs"]["top_logprobs"][index]
        rows.append(row)
    comparison.append(
        {
            "baseline": baseline,
            "phase": phase,
            "exact": sum(row["exact"] for row in rows),
            "total": len(rows),
            "requests": rows,
        }
    )
summary = {
    "size": args.size,
    "phases": list(phases),
    "comparisons": comparison,
    "trace": args.trace,
    "ordered_waves": args.ordered_waves,
    "control": "single process/image; fixed DP; scheduler paused while admitting each batch; target FULL unchanged",
}
(args.output / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2), flush=True)
