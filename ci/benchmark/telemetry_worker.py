"""Independent read-only /metrics process; no vLLM, model or completion imports."""

import argparse
import asyncio
import json
import os
import signal
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from metrics_capture import MetricsCapture


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, new_url):
        return None


class HTTPMetricsFetcher:
    """Fresh connection per scrape, no proxy, redirect, retry or timeout inflation."""

    def __init__(self, url):
        parsed = urllib.parse.urlsplit(url)
        if (
            parsed.scheme not in ("http", "https")
            or parsed.path != "/metrics"
            or (parsed.username or parsed.password or parsed.query or parsed.fragment)
        ):
            raise ValueError("Telemetry accepts only a direct HTTP(S) /metrics URL")
        self.url = url
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())

    async def __call__(self):
        # Blocking HTTP is intentional: this process serves only telemetry.
        # The completion process can serialize/flush for seconds without
        # preventing this interpreter from reading the socket or polling.
        request = urllib.request.Request(self.url, headers={"Accept": "text/plain", "Connection": "close"})
        try:
            with self.opener.open(request, timeout=5.0) as response:
                return response.status, dict(response.headers), response.read()
        except urllib.error.HTTPError as error:
            return error.code, dict(error.headers), error.read()


def write_json_atomic(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def parent_alive(parent_pid):
    if os.getppid() != parent_pid:
        return False
    try:
        os.kill(parent_pid, 0)
    except ProcessLookupError:
        return False
    return True


async def run_worker(args):
    directory = args.directory.resolve(strict=True)
    capture = MetricsCapture(HTTPMetricsFetcher(args.url), directory, args.stem, args.model, args.dp_size)
    ready_path = directory / f"{args.stem}-metrics-process-ready.json"
    stop_path = directory / f"{args.stem}-metrics-process-stop.json"
    pid = os.getpid()
    started = time.time()
    interrupted = asyncio.Event()
    loop = asyncio.get_running_loop()
    for number in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(number, interrupted.set)
    await capture.start()
    write_json_atomic(ready_path, {"pid": pid, "parent_pid": args.parent_pid, "phase": "before"})
    status = "cancelled"
    try:
        while not interrupted.is_set():
            if not parent_alive(args.parent_pid):
                raise RuntimeError("Telemetry parent exited; refusing orphan collection")
            if capture.task.done():
                # Surface raw-artifact write failures instead of publishing a
                # superficially valid report with missing during observations.
                capture.task.result()
                raise RuntimeError("Telemetry polling stopped unexpectedly")
            if stop_path.exists():
                status = json.loads(stop_path.read_text())["iteration_status"]
                if status not in ("completed", "failed", "cancelled", "startup_failed"):
                    raise ValueError("Invalid telemetry stop status")
                break
            await asyncio.sleep(0.05)
        report = await capture.finish(status)
        report["collector"] = {
            "mode": "independent-python-process",
            "pid": pid,
            "parent_pid": args.parent_pid,
            "started_wall_time_unix": started,
            "http_timeout_seconds": 5.0,
            "retry_count": 0,
            "artifact_flush_isolation": "separate interpreter/event loop/GIL from completion client",
        }
        write_json_atomic(directory / f"{args.stem}-metrics-iteration.json", report)
    finally:
        capture.stop.set()
        if capture.task is not None and not capture.task.done():
            capture.task.cancel()
            await asyncio.gather(capture.task, return_exceptions=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--stem", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dp-size", required=True, type=int)
    parser.add_argument("--parent-pid", required=True, type=int)
    args = parser.parse_args()
    if args.dp_size <= 0 or args.parent_pid <= 0:
        parser.error("DP size and parent PID must be positive")
    asyncio.run(run_worker(args))


if __name__ == "__main__":
    main()
