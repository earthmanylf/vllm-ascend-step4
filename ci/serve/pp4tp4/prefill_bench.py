#!/usr/bin/env python3
"""Measure prefill latency/throughput for a running Step5 service.

Each request is ``input_len`` random token ids with ``max_tokens=1``, so the
wall time up to the first token is dominated by prefill. Batch sizes are fired
concurrently; every round is repeated ``--num-iters`` times and reported
separately so run-to-run noise is visible instead of averaged away.

    python3 prefill_bench.py --host 10.0.0.1 --port 10086 --model step5 \
        --input-len 131072 --batch-sizes 1,2,4,8,16,32,64 --num-iters 3

The client talks HTTP only; it does not import vLLM or touch the NPUs.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time

import aiohttp


def build_prompt(input_len: int, seed: int) -> list[int]:
    """Deterministic token ids; the model never needs them to be valid text."""
    state = seed & 0xFFFFFFFF
    tokens = []
    for _ in range(input_len):
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        tokens.append(state % 120000)
    return tokens


async def one_request(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    prompt: list[int],
    start_at: float,
    timeout: float,
) -> dict:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": 1,
        "min_tokens": 1,
        "temperature": 0.0,
        "ignore_eos": True,
        "stream": True,
    }
    first_token_at = None
    try:
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
            if resp.status != 200:
                body = await resp.text()
                return {"ok": False, "error": f"HTTP {resp.status}: {body[:200]}"}
            async for raw in resp.content:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices") or []
                if choices and (choices[0].get("text") is not None or
                                choices[0].get("delta", {}).get("content") is not None):
                    if first_token_at is None:
                        first_token_at = time.monotonic()
                    break
    except Exception as exc:  # noqa: BLE001 - surface as a failed request
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    if first_token_at is None:
        return {"ok": False, "error": "stream closed without a token"}
    return {"ok": True, "ttft": first_token_at - start_at}


async def run_round(
    args, session: aiohttp.ClientSession, bs: int, seed: int
) -> dict:
    prompt = build_prompt(args.input_len, seed)
    url = f"http://{args.host}:{args.port}/v1/completions"
    start = time.monotonic()
    tasks = [
        one_request(session, url, args.model, prompt, start, args.request_timeout)
        for _ in range(bs)
    ]
    results = await asyncio.gather(*tasks)
    wall = time.monotonic() - start
    failures = [r for r in results if not r["ok"]]
    ttfts = sorted(r["ttft"] for r in results if r["ok"])
    row = {
        "input_len": args.input_len,
        "batch_size": bs,
        "round": seed,
        "complete_requests": len(ttfts),
        "failed_requests": len(failures),
        "wall_s": round(wall, 3),
    }
    if ttfts:
        prompt_tokens = args.input_len * len(ttfts)
        row.update(
            ttft_min_s=round(ttfts[0], 3),
            ttft_median_s=round(statistics.median(ttfts), 3),
            ttft_max_s=round(ttfts[-1], 3),
            # Whole-batch throughput: everything is admitted together and the
            # slowest request bounds the batch's prefill completion.
            prefill_tps_wall=round(prompt_tokens / ttfts[-1], 1),
            # Per-request throughput, averaged: isolates single-stream prefill.
            prefill_tps_per_request=round(
                statistics.mean(args.input_len / t for t in ttfts), 1),
            prefill_tps_total_wall=round(prompt_tokens / wall, 1),
        )
    if failures:
        row["status"] = "failed"
        row["error"] = failures[0]["error"]
    else:
        row["status"] = "completed"
    return row


async def main_async(args) -> list[dict]:
    connector = aiohttp.TCPConnector(limit=max(args.batch_sizes) + 8,
                                     force_close=False)
    async with aiohttp.ClientSession(connector=connector) as session:
        rows = []
        for bs in args.batch_sizes:
            for round_idx in range(args.num_iters):
                row = await run_round(args, session, bs, seed=args.seed + round_idx)
                rows.append(row)
                print(json.dumps(row), flush=True)
                if row["status"] != "completed":
                    return rows
                await asyncio.sleep(args.cooldown)
        return rows


def summarize(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[int, int], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["input_len"], row["batch_size"]), []).append(row)
    summary = []
    for (input_len, bs), group in sorted(grouped.items()):
        done = [g for g in group if g["status"] == "completed"]
        if not done:
            summary.append({"input_len": input_len, "batch_size": bs, "status": "failed"})
            continue
        best = max(done, key=lambda g: g["prefill_tps_wall"])
        summary.append({
            "input_len": input_len,
            "batch_size": bs,
            "status": "completed",
            "rounds": len(done),
            "ttft_min_s": statistics.median(g["ttft_min_s"] for g in done),
            "ttft_max_s": statistics.median(g["ttft_max_s"] for g in done),
            "prefill_tps_wall_best": best["prefill_tps_wall"],
            "prefill_tps_wall_median": round(
                statistics.median(g["prefill_tps_wall"] for g in done), 1),
            "prefill_tps_per_request_median": round(
                statistics.median(g["prefill_tps_per_request"] for g in done), 1),
        })
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=10086)
    parser.add_argument("--model", default="step5")
    parser.add_argument("--input-len", type=int, default=131072)
    parser.add_argument("--batch-sizes", default="1,2,4,8,16,32,64",
                        help="comma-separated batch sizes, ascending")
    parser.add_argument("--num-iters", type=int, default=3)
    parser.add_argument("--cooldown", type=float, default=5.0,
                        help="seconds between rounds (let the scheduler drain)")
    parser.add_argument("--request-timeout", type=float, default=3600.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output-file", default=None)
    args = parser.parse_args()
    args.batch_sizes = [int(x) for x in args.batch_sizes.split(",") if x.strip()]

    rows = asyncio.run(main_async(args))
    summary = summarize(rows)
    print("\n=== summary ===")
    header = f"{'input':>7} {'bs':>3} {'ttft_med':>9} {'ttft_max':>9} {'tps_wall':>9} {'tps/req':>9}"
    print(header)
    for row in summary:
        if row["status"] != "completed":
            print(f"{row['input_len']:>7} {row['batch_size']:>3} failed")
            continue
        print(f"{row['input_len']:>7} {row['batch_size']:>3} "
              f"{row['ttft_min_s']:>9.3f} {row['ttft_max_s']:>9.3f} "
              f"{row['prefill_tps_wall_median']:>9.1f} "
              f"{row['prefill_tps_per_request_median']:>9.1f}")

    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as stream:
            json.dump({"args": vars(args), "rows": rows, "summary": summary},
                      stream, indent=2)
            stream.write("\n")
    return 0 if all(r["status"] == "completed" for r in summary) else 1


if __name__ == "__main__":
    sys.exit(main())
