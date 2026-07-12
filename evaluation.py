"""CLI evaluation for retrieval, fallbacks, citations, and latency."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from statistics import mean, median
from time import perf_counter
from typing import Any
from uuid import uuid4

from langchain_core.messages import HumanMessage

from src.config import settings
from src.graph import graph


QUESTIONS_PATH = Path(__file__).parent / "tests" / "evaluation_questions.json"


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * fraction))]


def retrieval_hit(case: dict[str, Any], result: dict[str, Any]) -> bool | None:
    terms = [term.casefold() for term in case.get("expected_terms", [])]
    if not terms:
        return None
    evidence = " ".join(
        f"{document['text']} {document['metadata'].get('manual', '')} "
        f"{document['metadata'].get('chapter', '')} {document['metadata'].get('section', '')}"
        for document in result.get("reranked_docs", [])
    ).casefold()
    term_hit = any(term in evidence for term in terms)
    manuals = [manual.casefold() for manual in case.get("expected_manuals", [])]
    manual_hit = not manuals or any(manual in evidence for manual in manuals)
    return term_hit and manual_hit


def citation_covered(result: dict[str, Any]) -> bool:
    source_ids = {
        source.source_id if hasattr(source, "source_id") else source.get("source_id")
        for source in result.get("sources", [])
    }
    answer = result.get("answer", "")
    cited_ids = {f"S{number}" for number in re.findall(r"\[S(\d+)\]", answer)}
    return bool(source_ids) and cited_ids == source_ids


def invoke(question: str, thread_id: str) -> dict[str, Any]:
    return graph.invoke(
        {
            "messages": [HumanMessage(content=question)],
            "retry_count": 0,
            "allow_diagrams": True,
        },
        config={"configurable": {"thread_id": thread_id}},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0, help="Run only the first N cases")
    args = parser.parse_args()
    settings.validate()
    cases = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    if args.limit:
        cases = cases[: args.limit]

    hits: list[bool] = []
    fallback_checks: list[bool] = []
    citation_checks: list[bool] = []
    latencies: list[float] = []
    errors = 0
    for case in cases:
        thread_id = f"eval-{uuid4()}"
        try:
            if case.get("context_question"):
                invoke(case["context_question"], thread_id)
            started = perf_counter()
            result = invoke(case["question"], thread_id)
            latency = perf_counter() - started
        except Exception as exc:
            errors += 1
            print(f"FAIL {case['id']}: {exc}")
            continue

        hit = retrieval_hit(case, result)
        if hit is not None:
            hits.append(hit)
        if case["expected_status"] != "sufficient":
            fallback_checks.append(result.get("evidence_status") == case["expected_status"])
        else:
            citation_checks.append(citation_covered(result))
        latencies.append(latency)
        print(
            f"{case['id']}: status={result.get('evidence_status')} "
            f"hit={hit if hit is not None else 'n/a'} "
            f"cited={len(result.get('sources', []))} latency={latency:.2f}s"
        )

    report = {
        "cases": len(cases),
        "completed": len(latencies),
        "errors": errors,
        "retrieval_hit_rate": sum(hits) / len(hits) if hits else 0.0,
        "fallback_accuracy": (
            sum(fallback_checks) / len(fallback_checks) if fallback_checks else 0.0
        ),
        "citation_coverage": (
            sum(citation_checks) / len(citation_checks) if citation_checks else 0.0
        ),
        "latency_seconds": {
            "mean": mean(latencies) if latencies else 0.0,
            "median": median(latencies) if latencies else 0.0,
            "p95": percentile(latencies, 0.95),
        },
    }
    print("\nEvaluation report")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
