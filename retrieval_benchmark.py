"""Compare semantic-only and hybrid retrieval against labeled evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from html import escape
import json
import os
import platform
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any, Iterable, Sequence


METRIC_NAMES = ("precision_at_k", "recall_at_k", "f1_at_k", "reciprocal_rank")


def retrieval_question(case: dict[str, Any]) -> str:
    """Resolve a follow-up into a self-contained retrieval query."""
    return " ".join(
        part.strip()
        for part in (case.get("context_question", ""), case["question"])
        if part and part.strip()
    )


def attach_ground_truth(
    cases: Sequence[dict[str, Any]], labels_by_case: dict[str, list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    """Attach exact retrieval labels without mutating the answer-evaluation set."""
    case_ids = {str(case["id"]) for case in cases}
    label_ids = set(labels_by_case)
    if case_ids != label_ids:
        missing = sorted(case_ids - label_ids)
        extra = sorted(label_ids - case_ids)
        raise ValueError(f"ground-truth case mismatch: missing={missing}, extra={extra}")
    return [
        {**case, "expected_evidence": labels_by_case[str(case["id"])]}
        for case in cases
    ]


def score_ranked_ids(
    retrieved_ids: Sequence[str], relevant_ids: set[str], *, k: int
) -> dict[str, float]:
    """Score a ranked evidence-ID list against exact relevance labels."""
    if k <= 0:
        raise ValueError("k must be positive")
    ranked = list(dict.fromkeys(str(item) for item in retrieved_ids))[:k]
    hits = sum(item in relevant_ids for item in ranked)
    precision = hits / k
    recall = hits / len(relevant_ids) if relevant_ids else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    reciprocal_rank = next(
        (1.0 / rank for rank, item in enumerate(ranked, start=1) if item in relevant_ids),
        0.0,
    )
    return {
        "precision_at_k": precision,
        "recall_at_k": recall,
        "f1_at_k": f1,
        "reciprocal_rank": reciprocal_rank,
    }


def _summarize(rows: Iterable[dict[str, Any]]) -> dict[str, float]:
    values = list(rows)
    if not values:
        return {
            **{name: 0.0 for name in METRIC_NAMES},
            "mean_latency_seconds": 0.0,
            "p50_latency_seconds": 0.0,
            "p95_latency_seconds": 0.0,
        }
    latencies = sorted(float(row["latency_seconds"]) for row in values)
    return {
        **{
            name: mean(float(row["metrics"][name]) for row in values)
            for name in METRIC_NAMES
        },
        "mean_latency_seconds": mean(latencies),
        "p50_latency_seconds": _percentile(latencies, 0.50),
        "p95_latency_seconds": _percentile(latencies, 0.95),
    }


def _percentile(sorted_values: Sequence[float], quantile: float) -> float:
    position = (len(sorted_values) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * fraction


def compare_pipelines(
    semantic_results: Sequence[dict[str, Any]],
    hybrid_results: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Return aggregate metrics and the measured hybrid F1 lift."""
    semantic = _summarize(semantic_results)
    hybrid = _summarize(hybrid_results)
    absolute_lift = hybrid["f1_at_k"] - semantic["f1_at_k"]
    relative_lift = (
        absolute_lift / semantic["f1_at_k"]
        if semantic["f1_at_k"] > 0
        else None
    )
    return {
        "semantic_only": semantic,
        "hybrid": hybrid,
        "absolute_f1_lift": absolute_lift,
        "relative_f1_lift": relative_lift,
    }


def _evidence_id(document: dict[str, Any]) -> str:
    return str(document.get("metadata", {}).get("evidence_id") or document["chunk_id"])


def semantic_retrieve(query: str, *, k: int, config: Any) -> list[dict[str, Any]]:
    """Run the dense-vector-only baseline and resolve complete evidence units."""
    from src.retrieval import resolve_evidence_units, vector_search

    candidates = vector_search(query, max(config.vector_top_k, k * 4), config=config)
    return resolve_evidence_units(candidates, limit=k, config=config)


def hybrid_retrieve(query: str, *, k: int, config: Any) -> list[dict[str, Any]]:
    """Run dense + BM25 + weighted RRF + context + cross-encoder retrieval."""
    from src.retrieval import (
        bm25_search,
        expand_context,
        reciprocal_rank_fusion,
        rerank_documents,
        resolve_evidence_units,
        vector_search,
    )

    vector = vector_search(query, config.vector_top_k, config=config)
    lexical = bm25_search(query, config.bm25_top_k, config=config)
    fused = reciprocal_rank_fusion(
        vector,
        lexical,
        limit=config.fused_top_k,
        config=config,
    )
    expanded = expand_context(fused, limit=20, config=config)
    reranked = rerank_documents(query, expanded, limit=k, config=config)
    if reranked and not all(
        "reranker_score" in document["retrieval_scores"] for document in reranked
    ):
        raise RuntimeError("cross-encoder reranking was not applied")
    return resolve_evidence_units(reranked, limit=k, config=config)


def _run_one(
    query: str,
    relevant_ids: set[str],
    *,
    k: int,
    config: Any,
    pipeline: Any,
) -> dict[str, Any]:
    started = perf_counter()
    documents = pipeline(query, k=k, config=config)
    latency = perf_counter() - started
    ranked_ids = [_evidence_id(document) for document in documents]
    return {
        "retrieved_ids": ranked_ids,
        "latency_seconds": latency,
        "metrics": score_ranked_ids(ranked_ids, relevant_ids, k=k),
    }


def run_benchmark(
    cases: Sequence[dict[str, Any]], *, k: int, config: Any
) -> dict[str, Any]:
    """Execute both retrieval variants on the same human-labeled cases."""
    warmup_query = "Opcenter retrieval benchmark warmup"
    semantic_retrieve(warmup_query, k=k, config=config)
    hybrid_retrieve(warmup_query, k=k, config=config)
    rows: list[dict[str, Any]] = []
    semantic_scored: list[dict[str, Any]] = []
    hybrid_scored: list[dict[str, Any]] = []
    for case in cases:
        query = retrieval_question(case)
        labels = case.get("expected_evidence", [])
        relevant_ids = {str(label["evidence_id"]) for label in labels}
        if not relevant_ids:
            rows.append(
                {
                    "id": case["id"],
                    "category": case["category"],
                    "question": query,
                    "scored": False,
                    "reason": "no relevant evidence expected",
                }
            )
            continue
        semantic = _run_one(
            query, relevant_ids, k=k, config=config, pipeline=semantic_retrieve
        )
        hybrid = _run_one(
            query, relevant_ids, k=k, config=config, pipeline=hybrid_retrieve
        )
        semantic_scored.append(semantic)
        hybrid_scored.append(hybrid)
        rows.append(
            {
                "id": case["id"],
                "category": case["category"],
                "question": query,
                "scored": True,
                "relevant_ids": sorted(relevant_ids),
                "semantic_only": semantic,
                "hybrid": hybrid,
            }
        )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "k": k,
            "case_count": len(cases),
            "scored_case_count": len(semantic_scored),
            "semantic_pipeline": "Chroma dense-vector retrieval",
            "hybrid_pipeline": "Chroma + BM25 + weighted RRF + context expansion + cross-encoder reranking",
            "embedding_model": config.embedding_model,
            "embedding_model_revision": config.embedding_model_revision,
            "reranker_model": config.reranker_model,
            "reranker_model_revision": config.reranker_model_revision,
            "vector_top_k": config.vector_top_k,
            "bm25_top_k": config.bm25_top_k,
            "fused_top_k": config.fused_top_k,
            "warmup_query": warmup_query,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
        "comparison": compare_pipelines(semantic_scored, hybrid_scored),
        "cases": rows,
    }


def _percent(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.2%}"


def write_report(
    report: dict[str, Any], output_dir: Path, *, stem: str
) -> tuple[Path, Path]:
    """Persist an auditable JSON report and a compact portfolio-friendly HTML view."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{stem}.json"
    html_path = output_dir / f"{stem}.html"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    comparison = report["comparison"]
    semantic = comparison["semantic_only"]
    hybrid = comparison["hybrid"]
    case_rows = "".join(
        "<tr>"
        f"<td>{escape(str(row['id']))}</td>"
        f"<td>{escape(str(row['category']))}</td>"
        f"<td>{escape(str(row['question']))}</td>"
        f"<td>{_percent(row.get('semantic_only', {}).get('metrics', {}).get('f1_at_k'))}</td>"
        f"<td>{_percent(row.get('hybrid', {}).get('metrics', {}).get('f1_at_k'))}</td>"
        "</tr>"
        for row in report.get("cases", [])
    )
    metric_rows = "".join(
        f"<tr><td>{label}</td><td>{semantic[key]:.4f}</td><td>{hybrid[key]:.4f}</td></tr>"
        for label, key in (
            ("Precision@K", "precision_at_k"),
            ("Recall@K", "recall_at_k"),
            ("F1@K", "f1_at_k"),
            ("MRR", "reciprocal_rank"),
            ("Mean latency (s)", "mean_latency_seconds"),
            ("p50 latency (s)", "p50_latency_seconds"),
            ("p95 latency (s)", "p95_latency_seconds"),
        )
    )
    html_text = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Retrieval benchmark</title>
<style>body{{font:16px system-ui;max-width:1100px;margin:40px auto;padding:0 20px;color:#172033}}.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}}.card{{padding:18px;border:1px solid #d9dfeb;border-radius:12px}}table{{width:100%;border-collapse:collapse;margin-top:28px}}th,td{{padding:10px;border-bottom:1px solid #e4e8f0;text-align:left}}th{{background:#f5f7fb}}</style>
</head><body><h1>Retrieval benchmark</h1>
<p>{report['configuration']['scored_case_count']} scored cases at K={report['configuration']['k']}.</p>
<div class="cards"><div class="card"><h2>Semantic-only</h2><p>F1@K: {_percent(semantic['f1_at_k'])}</p></div>
<div class="card"><h2>Hybrid</h2><p>F1@K: {_percent(hybrid['f1_at_k'])}</p></div>
<div class="card"><h2>Relative lift</h2><p>{_percent(comparison['relative_f1_lift'])}</p></div></div>
<table><thead><tr><th>Metric</th><th>Semantic-only</th><th>Hybrid</th></tr></thead><tbody>{metric_rows}</tbody></table>
<table><thead><tr><th>Case</th><th>Category</th><th>Question</th><th>Semantic F1</th><th>Hybrid F1</th></tr></thead><tbody>{case_rows}</tbody></table>
</body></html>"""
    html_path.write_text(html_text, encoding="utf-8")
    return json_path, html_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("tests/evaluation_questions.json"))
    parser.add_argument(
        "--labels", type=Path, default=Path("tests/retrieval_ground_truth.json")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation_results"))
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--allow-model-download", action="store_true")
    args = parser.parse_args()
    if not args.allow_model_download:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ["LANGSMITH_TRACING"] = "false"
    from src.config import settings

    cases = attach_ground_truth(
        json.loads(args.dataset.read_text(encoding="utf-8")),
        json.loads(args.labels.read_text(encoding="utf-8")),
    )
    report = run_benchmark(cases, k=args.k, config=settings)
    stem = "retrieval-benchmark-latest"
    json_path, html_path = write_report(report, args.output_dir, stem=stem)
    print(f"JSON report: {json_path}")
    print(f"HTML report: {html_path}")
    print(f"Hybrid relative F1 lift: {_percent(report['comparison']['relative_f1_lift'])}")


if __name__ == "__main__":
    main()
