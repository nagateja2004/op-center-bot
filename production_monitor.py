"""Summarize recent production LangSmith traces and flag actionable regressions."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta
import json
import os
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * fraction))]


def _value(item: Any, name: str, default: Any = None) -> Any:
    return item.get(name, default) if isinstance(item, dict) else getattr(item, name, default)


def _feedback_average(stats: Any, key: str) -> float | None:
    if not isinstance(stats, dict) or key not in stats:
        return None
    value = stats[key]
    average = value.get("avg") if isinstance(value, dict) else getattr(value, "avg", None)
    return float(average) if average is not None else None


def analyze_runs(
    runs: Iterable[Any],
    *,
    project: str,
    lookback_minutes: int,
    max_error_rate: float,
    max_p95_latency: float,
    min_feedback_score: float,
    feedback_keys: tuple[str, ...],
    min_runs: int = 0,
    max_average_cost: float = 0.0,
) -> dict[str, Any]:
    items = list(runs)
    latencies = [float(value) for item in items if (value := _value(item, "latency")) is not None]
    costs = [float(value) for item in items if (value := _value(item, "total_cost")) is not None]
    error_runs = [item for item in items if _value(item, "error")]
    negative_feedback: list[Any] = []
    for item in items:
        stats = _value(item, "feedback_stats", {})
        if any(
            average is not None and average < min_feedback_score
            for key in feedback_keys
            if (average := _feedback_average(stats, key)) is not None
        ):
            negative_feedback.append(item)

    count = len(items)
    error_rate = len(error_runs) / count if count else 0.0
    p95_latency = percentile(latencies, 0.95)
    average_cost = mean(costs) if costs else 0.0
    reasons = []
    if count < min_runs:
        reasons.append(f"Only {count} traces were found; expected at least {min_runs}.")
    if error_rate > max_error_rate:
        reasons.append(f"Error rate {error_rate:.1%} exceeds {max_error_rate:.1%}.")
    if p95_latency > max_p95_latency:
        reasons.append(f"p95 latency {p95_latency:.2f}s exceeds {max_p95_latency:.2f}s.")
    if negative_feedback:
        reasons.append(f"{len(negative_feedback)} trace(s) have feedback below {min_feedback_score:.2f}.")
    if max_average_cost > 0 and average_cost > max_average_cost:
        reasons.append(
            f"Average trace cost ${average_cost:.4f} exceeds ${max_average_cost:.4f}."
        )

    flagged = []
    seen_ids = set()
    for item in [*error_runs, *negative_feedback]:
        run_id = str(_value(item, "id", ""))
        if run_id not in seen_ids:
            seen_ids.add(run_id)
            flagged.append(item)
        if len(flagged) == 10:
            break
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "project": project,
        "lookback_minutes": lookback_minutes,
        "runs": count,
        "errors": len(error_runs),
        "error_rate": error_rate,
        "latency_seconds": {
            "p50": median(latencies) if latencies else 0.0,
            "p95": p95_latency,
        },
        "cost_usd": {
            "total": sum(costs),
            "average_per_trace": average_cost,
        },
        "negative_feedback_runs": len(negative_feedback),
        "issue_found": bool(reasons),
        "reasons": reasons,
        "flagged_traces": [
            {
                "id": str(_value(item, "id", "")),
                "name": str(_value(item, "name", "")),
                "url": str(_value(item, "url", "") or ""),
            }
            for item in flagged
        ],
    }


def render_issue(report: dict[str, Any]) -> str:
    reasons = "\n".join(f"- {reason}" for reason in report["reasons"])
    traces = "\n".join(
        f"- [{trace['name'] or trace['id']}]({trace['url']})"
        if trace["url"] else f"- `{trace['id']}` — {trace['name']}"
        for trace in report["flagged_traces"]
    ) or "- No individual trace was selected; inspect the project dashboard."
    return f"""## Automated production observation

Project: `{report['project']}`
Window: {report['lookback_minutes']} minutes
Runs: {report['runs']}
Error rate: {report['error_rate']:.1%}
p50 / p95 latency: {report['latency_seconds']['p50']:.2f}s / {report['latency_seconds']['p95']:.2f}s
Average cost per trace: ${report['cost_usd']['average_per_trace']:.4f}

### Trigger

{reasons}

### Traces for human review

{traces}

### Golden-case decision

Do not add a case automatically. A reviewer must reproduce the problem, validate the expected behavior, edit this issue to include one JSON block in the format below, and then apply the `golden-approved` label.

```json
{{
  "id": "production_001",
  "category": "production_failure",
  "question": "Human-validated user question",
  "expected_status": "sufficient",
  "expected_terms": ["required concept"],
  "expected_manuals": []
}}
```
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("production_monitor_report.json"))
    parser.add_argument("--issue-output", type=Path, default=Path("production_issue.md"))
    parser.add_argument("--lookback-minutes", type=int, default=int(os.getenv("MONITOR_LOOKBACK_MINUTES", "1440")))
    parser.add_argument("--max-error-rate", type=float, default=float(os.getenv("MONITOR_MAX_ERROR_RATE", "0.05")))
    parser.add_argument("--max-p95-latency", type=float, default=float(os.getenv("MONITOR_MAX_P95_LATENCY", "75")))
    parser.add_argument("--min-feedback-score", type=float, default=float(os.getenv("MONITOR_MIN_FEEDBACK_SCORE", "0.5")))
    parser.add_argument("--min-runs", type=int, default=int(os.getenv("MONITOR_MIN_RUNS", "0")))
    parser.add_argument("--max-average-cost", type=float, default=float(os.getenv("MONITOR_MAX_AVERAGE_COST_USD", "0")))
    parser.add_argument("--limit", type=int, default=int(os.getenv("MONITOR_RUN_LIMIT", "500")))
    args = parser.parse_args()
    if args.lookback_minutes <= 0 or args.limit <= 0 or args.min_runs < 0:
        parser.error("lookback, limit, and minimum run values are invalid")
    if not 0 <= args.max_error_rate <= 1 or not 0 <= args.min_feedback_score <= 1:
        parser.error("rate and feedback thresholds must be between 0 and 1")
    if args.max_p95_latency <= 0 or args.max_average_cost < 0:
        parser.error("latency must be positive and cost cannot be negative")

    project = (
        os.getenv("LANGSMITH_PRODUCTION_PROJECT", "opcenter-rag-production").strip()
        or "opcenter-rag-production"
    )
    endpoint = (
        os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com").strip()
        or "https://api.smith.langchain.com"
    )
    api_key = os.getenv("LANGSMITH_API_KEY", "").strip()
    if not api_key:
        parser.error("LANGSMITH_API_KEY is required")
    from langsmith import Client

    client = Client(
        api_key=api_key,
        api_url=endpoint,
    )
    runs = client.list_runs(
        project_name=project,
        is_root=True,
        start_time=datetime.now(UTC) - timedelta(minutes=args.lookback_minutes),
        limit=args.limit,
    )
    feedback_keys = tuple(
        value.strip()
        for value in os.getenv("MONITOR_FEEDBACK_KEYS", "correctness,user_feedback").split(",")
        if value.strip()
    )
    report = analyze_runs(
        runs,
        project=project,
        lookback_minutes=args.lookback_minutes,
        max_error_rate=args.max_error_rate,
        max_p95_latency=args.max_p95_latency,
        min_feedback_score=args.min_feedback_score,
        feedback_keys=feedback_keys,
        min_runs=args.min_runs,
        max_average_cost=args.max_average_cost,
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.issue_output.write_text(render_issue(report), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["issue_found"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
