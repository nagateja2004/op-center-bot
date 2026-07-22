"""Evaluate the running production API for routing, answers, citations, and latency."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
from statistics import mean, median
from time import perf_counter
from typing import Any
from urllib.request import Request, urlopen
from uuid import uuid4


QUESTIONS_PATH = Path(__file__).parent / "tests" / "evaluation_questions.json"


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * fraction))]


def api_request(url: str, payload: dict[str, Any] | None = None):
    body = json.dumps(payload).encode() if payload is not None else None
    return urlopen(Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"} if body else {},
        method="POST" if body else "GET",
    ), timeout=180)


def invoke(question: str, backend_url: str, conversation: dict[str, str]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "message": question,
        "session_id": conversation["session_id"],
        "diagram_enabled": True,
    }
    if conversation.get("conversation_id"):
        payload.update({
            "conversation_id": conversation["conversation_id"],
            "thread_id": conversation["thread_id"],
        })
    with api_request(f"{backend_url}/v1/chat", payload) as response:
        accepted = json.load(response)
    conversation.update({
        "conversation_id": accepted["conversation_id"],
        "thread_id": accepted["thread_id"],
    })
    event, data = "message", []
    with api_request(f"{backend_url}/v1/chat/{accepted['request_id']}/stream") as response:
        for raw_line in response:
            line = raw_line.decode().rstrip("\r\n")
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data.append(line[5:].strip())
            elif not line and data:
                item = json.loads("\n".join(data))
                if event == "complete":
                    return item
                if event == "error":
                    raise RuntimeError(item.get("message", "Evaluation request failed"))
                event, data = "message", []
    raise RuntimeError("Stream ended without a completion event")


def output_hit(case: dict[str, Any], result: dict[str, Any]) -> bool | None:
    terms = [term.casefold() for term in case.get("expected_terms", [])]
    sources = result.get("sources", [])
    searchable = " ".join([
        str(result.get("answer", "")),
        *(f"{source.get('manual', '')} {source.get('chapter', '')} {source.get('section', '')}" for source in sources),
    ]).casefold()
    if not terms:
        return None
    return any(term in searchable for term in terms)


def manual_hit(case: dict[str, Any], result: dict[str, Any]) -> bool | None:
    expected = [value.casefold() for value in case.get("expected_manuals", [])]
    if not expected:
        return None
    actual = " ".join(str(source.get("manual", "")) for source in result.get("sources", [])).casefold()
    return all(value in actual for value in expected)


def citations_valid(result: dict[str, Any]) -> bool:
    source_ids = {str(source.get("source_id")) for source in result.get("sources", [])}
    cited_ids = {f"S{number}" for number in re.findall(r"\[S(\d+)\]", result.get("answer", ""))}
    return bool(source_ids) and cited_ids == source_ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend-url", default=os.getenv("BACKEND_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cases = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    cases = cases[: args.limit] if args.limit else cases
    term_hits: list[bool] = []
    manual_hits: list[bool] = []
    status_hits: list[bool] = []
    citation_hits: list[bool] = []
    diagram_hits: list[bool] = []
    latencies: list[float] = []
    errors = 0
    for case in cases:
        conversation = {"session_id": str(uuid4())}
        try:
            if case.get("context_question"):
                invoke(case["context_question"], args.backend_url.rstrip("/"), conversation)
            started = perf_counter()
            result = invoke(case["question"], args.backend_url.rstrip("/"), conversation)
            latencies.append(perf_counter() - started)
        except Exception as exc:
            errors += 1
            print(f"FAIL {case['id']}: {exc}")
            continue
        term = output_hit(case, result)
        manual = manual_hit(case, result)
        status = result.get("evidence", {}).get("status") == case["expected_status"]
        if term is not None:
            term_hits.append(term)
        if manual is not None:
            manual_hits.append(manual)
        status_hits.append(status)
        if case["expected_status"] == "sufficient":
            citation_hits.append(citations_valid(result))
        if "expected_diagram" in case:
            diagram_hits.append(bool(result.get("diagram", {}).get("generated")) == case["expected_diagram"])
        print(f"{case['id']}: status={status} term={term} manual={manual} cited={citations_valid(result)} latency={latencies[-1]:.2f}s")
    report = {
        "cases": len(cases),
        "completed": len(latencies),
        "errors": errors,
        "answer_term_accuracy": sum(term_hits) / len(term_hits) if term_hits else 0.0,
        "manual_routing_accuracy": sum(manual_hits) / len(manual_hits) if manual_hits else 0.0,
        "status_accuracy": sum(status_hits) / len(status_hits) if status_hits else 0.0,
        "citation_id_accuracy": sum(citation_hits) / len(citation_hits) if citation_hits else 0.0,
        "diagram_render_accuracy": sum(diagram_hits) / len(diagram_hits) if diagram_hits else None,
        "latency_seconds": {
            "mean": mean(latencies) if latencies else 0.0,
            "median": median(latencies) if latencies else 0.0,
            "p95": percentile(latencies, 0.95),
        },
    }
    rendered = json.dumps(report, indent=2)
    print(f"\nEvaluation report\n{rendered}")
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
