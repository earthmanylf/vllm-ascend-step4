"""Controlled real-model token comparisons with explicit DP routing and logits."""

import argparse
import concurrent.futures
import json
import threading
import time
import urllib.request
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True, type=Path)
parser.add_argument("--kind", choices=("serial", "pinned-batch", "unpinned"), default="serial")
parser.add_argument("--rounds", type=int, default=2)
parser.add_argument("--indices", default="0,1,10,11,12,13,14")
parser.add_argument("--tokens", type=int, default=32)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
indices = [int(x) for x in args.indices.split(",")]
inputs = [[100 + (i * 17 + j) % 1000 for j in range(1024)] for i in range(16)]


def run(index, round_index, barrier=None):
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
    }
    headers = {"Content-Type": "application/json"}
    if args.kind != "unpinned":
        headers["X-data-parallel-rank"] = str(index % 8 if args.kind == "pinned-batch" else 0)
    req = urllib.request.Request(
        "http://127.0.0.1:19082/v1/completions",
        data=json.dumps(payload).encode(),
        headers=headers,
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    if barrier:
        barrier.wait()
    started = time.monotonic()
    with opener.open(req, timeout=240) as response:
        result = json.load(response)
    row = {
        "index": index,
        "round": round_index,
        "dp": headers.get("X-data-parallel-rank"),
        "elapsed": time.monotonic() - started,
        "response": result,
    }
    (args.output / f"round{round_index}-request{index}.json").write_text(json.dumps(row, indent=2))
    choice = result["choices"][0]
    assert len(choice["token_ids"]) == args.tokens and choice["finish_reason"] == "length"
    return row


rounds = []
for round_index in range(args.rounds):
    if args.kind == "serial":
        results = [run(i, round_index) for i in indices]
    else:
        barrier = threading.Barrier(len(indices))
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(indices)) as pool:
            results = list(
                pool.map(lambda i, round_index=round_index, barrier=barrier: run(i, round_index, barrier), indices)
            )
    rounds.append(results)
    print("completed round", round_index, flush=True)
comparisons = []
for round_index, results in enumerate(rounds[1:], start=1):
    for before, after in zip(rounds[0], results):
        a, b = before["response"]["choices"][0], after["response"]["choices"][0]
        differences = [i for i, (x, y) in enumerate(zip(a["token_ids"], b["token_ids"])) if x != y]
        first = differences[0] if differences else None
        row = {"index": before["index"], "round": round_index, "exact": not differences, "first_difference": first}
        if first is not None:
            row.update(
                {
                    "before_top_logprobs": a["logprobs"]["top_logprobs"][first],
                    "after_top_logprobs": b["logprobs"]["top_logprobs"][first],
                }
            )
        comparisons.append(row)
summary = {
    "kind": args.kind,
    "indices": indices,
    "comparisons": comparisons,
    "exact_count": sum(r["exact"] for r in comparisons),
    "total": len(comparisons),
}
(args.output / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2), flush=True)
