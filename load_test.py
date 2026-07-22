"""Bounded load test for the production POST plus SSE chat API."""

from __future__ import annotations

import argparse
import asyncio
import json
from statistics import mean, median
from time import perf_counter
from uuid import uuid4

import httpx


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * fraction))] if ordered else 0.0


async def one_request(client: httpx.AsyncClient, backend_url: str, number: int) -> dict:
    started = perf_counter()
    first_token = None
    accepted = await client.post(
        f"{backend_url}/v1/chat",
        json={
            "message": f"What is a Factory? Test request {number}.",
            "session_id": str(uuid4()),
            "diagram_enabled": False,
        },
    )
    if accepted.status_code != 202:
        return {"status": f"http_{accepted.status_code}", "latency": perf_counter() - started}
    request_id = accepted.json()["request_id"]
    event = "message"
    async with client.stream("GET", f"{backend_url}/v1/chat/{request_id}/stream") as response:
        async for line in response.aiter_lines():
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data = json.loads(line[5:].strip())
                if event == "answer" and first_token is None:
                    first_token = perf_counter() - started
                elif event == "complete":
                    return {"status": "complete", "latency": perf_counter() - started, "ttft": first_token}
                elif event == "error":
                    message = str(data.get("message", "")).casefold()
                    return {"status": "busy" if "busy" in message else "error", "latency": perf_counter() - started}
    return {"status": "incomplete", "latency": perf_counter() - started}


async def run(backend_url: str, users: int) -> None:
    timeout = httpx.Timeout(240)
    limits = httpx.Limits(max_connections=users, max_keepalive_connections=users)
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        results = await asyncio.gather(*(one_request(client, backend_url, number) for number in range(users)))
    latencies = [result["latency"] for result in results]
    ttft = [result["ttft"] for result in results if result.get("ttft") is not None]
    statuses = {status: sum(result["status"] == status for result in results) for status in sorted({result["status"] for result in results})}
    print(json.dumps({
        "users": users,
        "statuses": statuses,
        "success_rate": statuses.get("complete", 0) / users,
        "latency_seconds": {"mean": mean(latencies), "median": median(latencies), "p95": percentile(latencies, 0.95)},
        "time_to_first_token_seconds": {"median": median(ttft) if ttft else None, "p95": percentile(ttft, 0.95) if ttft else None},
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend-url", default="http://127.0.0.1:8000")
    parser.add_argument("--users", type=int, default=50)
    args = parser.parse_args()
    if not 1 <= args.users <= 500:
        raise SystemExit("--users must be between 1 and 500")
    asyncio.run(run(args.backend_url.rstrip("/"), args.users))


if __name__ == "__main__":
    main()
