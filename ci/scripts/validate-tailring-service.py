"""Bounded real-model smoke and fixed-input MTP correctness requests, not a benchmark."""

import argparse
import concurrent.futures
import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", required=True, type=Path)
parser.add_argument("--base-url", default="http://127.0.0.1:19082")
parser.add_argument("--batch-only", action="store_true")
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def call(path, payload=None, timeout=300):
    body = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(args.base_url + path, data=body, headers={"Content-Type": "application/json"})
    try:
        with opener.open(req, timeout=timeout) as response:
            raw = response.read()
            return response.status, json.loads(raw) if raw else None
    except urllib.error.HTTPError as error:
        raw = error.read().decode(errors="replace")
        return error.code, {"error_body": raw}


def save(name, value):
    (args.output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


health, _ = call("/health", timeout=15)
models_status, models = call("/v1/models", timeout=15)
save("health-models.json", {"health": health, "models_status": models_status, "models": models})
if health != 200 or models_status != 200:
    raise RuntimeError("Service not ready")
if not args.batch_only:
    messages = [
        ("math", "计算 17×19，只回答最终数字。"),
        ("capital", "法国的首都是哪里？请简短回答。"),
    ]
    for name, text in messages:
        started = time.monotonic()
        status, response = call(
            "/v1/chat/completions",
            {
                "model": "step5",
                "messages": [{"role": "user", "content": text}],
                "temperature": 0,
                "seed": 0,
                "max_tokens": 192,
            },
        )
        save(name + ".json", {"status": status, "elapsed_seconds": time.monotonic() - started, "response": response})
        if status != 200:
            raise RuntimeError(f"Real chat request failed: {name}")

inputs = [[100 + (i * 17 + j) % 1000 for j in range(1024)] for i in range(16)]
save("inputs.json", inputs)


def request(index):
    status, response = call(
        "/v1/completions",
        {
            "model": "step5",
            "prompt": inputs[index],
            "temperature": 0,
            "seed": 0,
            "max_tokens": 32,
            "min_tokens": 32,
            "ignore_eos": True,
            "return_token_ids": True,
            "stream": False,
        },
    )
    save(f"batch-response-{index}.json", {"status": status, "response": response})
    if status != 200:
        return {"request": index, "pass": False, "status": status}
    choice = response["choices"][0]
    tokens = choice.get("token_ids") or []
    return {
        "request": index,
        "pass": choice["finish_reason"] == "length" and len(tokens) == 32,
        "token_ids": tokens,
        "finish_reason": choice["finish_reason"],
    }


with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
    results = list(pool.map(request, range(16)))
result = {
    "execution_pass": all(row["pass"] for row in results),
    "requests": results,
    "input_sha256": hashlib.sha256(json.dumps(inputs).encode()).hexdigest(),
    "paired_eager_comparison": "not yet performed",
    "scope": "fixed BS16 execution check, not accuracy/performance/production certification",
}
save("batch-result.json", result)
print(json.dumps(result, indent=2))
if not result["execution_pass"]:
    raise RuntimeError("BS16 execution check failed")
