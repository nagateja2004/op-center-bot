import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_ranked_retrieval_metrics_use_exact_evidence_ids() -> None:
    benchmark = importlib.import_module("retrieval_benchmark")

    metrics = benchmark.score_ranked_ids(
        ["evidence-1", "irrelevant", "evidence-2"],
        {"evidence-1", "evidence-2", "evidence-3"},
        k=3,
    )

    assert metrics == {
        "precision_at_k": pytest.approx(2 / 3),
        "recall_at_k": pytest.approx(2 / 3),
        "f1_at_k": pytest.approx(2 / 3),
        "reciprocal_rank": 1.0,
    }


def test_report_compares_macro_f1_without_inventing_an_improvement() -> None:
    benchmark = importlib.import_module("retrieval_benchmark")
    semantic = [
        {"metrics": {"precision_at_k": 0.5, "recall_at_k": 0.5, "f1_at_k": 0.5, "reciprocal_rank": 0.5}, "latency_seconds": 0.2},
        {"metrics": {"precision_at_k": 0.0, "recall_at_k": 0.0, "f1_at_k": 0.0, "reciprocal_rank": 0.0}, "latency_seconds": 0.4},
    ]
    hybrid = [
        {"metrics": {"precision_at_k": 1.0, "recall_at_k": 1.0, "f1_at_k": 1.0, "reciprocal_rank": 1.0}, "latency_seconds": 0.6},
        {"metrics": {"precision_at_k": 0.5, "recall_at_k": 0.5, "f1_at_k": 0.5, "reciprocal_rank": 0.5}, "latency_seconds": 0.8},
    ]

    comparison = benchmark.compare_pipelines(semantic, hybrid)

    assert comparison["semantic_only"]["f1_at_k"] == 0.25
    assert comparison["hybrid"]["f1_at_k"] == 0.75
    assert comparison["absolute_f1_lift"] == 0.5
    assert comparison["relative_f1_lift"] == 2.0
    assert comparison["semantic_only"]["p50_latency_seconds"] == pytest.approx(0.3)
    assert comparison["semantic_only"]["p95_latency_seconds"] == pytest.approx(0.39)


def test_follow_up_retrieval_question_includes_its_context() -> None:
    benchmark = importlib.import_module("retrieval_benchmark")

    question = benchmark.retrieval_question(
        {
            "context_question": "What is a Factory?",
            "question": "How do I define it?",
        }
    )

    assert question == "What is a Factory? How do I define it?"


def test_ground_truth_is_attached_by_case_id() -> None:
    benchmark = importlib.import_module("retrieval_benchmark")
    cases = [{"id": "one", "question": "Q", "category": "direct"}]
    labels = {
        "one": [
            {"evidence_id": "e1", "source_file": "manual.pdf", "pdf_page": 2}
        ]
    }

    merged = benchmark.attach_ground_truth(cases, labels)

    assert merged[0]["expected_evidence"] == labels["one"]
    assert "expected_evidence" not in cases[0]


def test_benchmark_artifacts_include_machine_and_human_readable_reports(
    tmp_path: Path,
) -> None:
    benchmark = importlib.import_module("retrieval_benchmark")
    report = {
        "configuration": {"k": 5, "case_count": 1, "scored_case_count": 1},
        "comparison": {
            "semantic_only": {
                "precision_at_k": 0.2,
                "recall_at_k": 0.2,
                "f1_at_k": 0.2,
                "reciprocal_rank": 0.2,
                "mean_latency_seconds": 0.2,
                "p50_latency_seconds": 0.2,
                "p95_latency_seconds": 0.2,
            },
            "hybrid": {
                "precision_at_k": 0.3,
                "recall_at_k": 0.3,
                "f1_at_k": 0.3,
                "reciprocal_rank": 0.3,
                "mean_latency_seconds": 0.3,
                "p50_latency_seconds": 0.3,
                "p95_latency_seconds": 0.3,
            },
            "absolute_f1_lift": 0.1,
            "relative_f1_lift": 0.5,
        },
        "cases": [],
    }

    json_path, html_path = benchmark.write_report(report, tmp_path, stem="sample")

    assert json.loads(json_path.read_text(encoding="utf-8")) == report
    html = html_path.read_text(encoding="utf-8")
    assert "Semantic-only" in html
    assert "Hybrid" in html
    assert "50.00%" in html


def test_benchmark_warms_both_pipelines_before_measuring(monkeypatch) -> None:
    benchmark = importlib.import_module("retrieval_benchmark")
    calls: list[tuple[str, str]] = []

    def result(pipeline: str):
        def retrieve(query: str, *, k: int, config):
            calls.append((pipeline, query))
            return [{"chunk_id": "e1", "metadata": {}, "retrieval_scores": {}}]

        return retrieve

    monkeypatch.setattr(benchmark, "semantic_retrieve", result("semantic"))
    monkeypatch.setattr(benchmark, "hybrid_retrieve", result("hybrid"))
    config = SimpleNamespace(
        embedding_model="embedding",
        embedding_model_revision="revision",
        reranker_model="reranker",
        reranker_model_revision="revision",
        vector_top_k=12,
        bm25_top_k=12,
        fused_top_k=18,
    )
    cases = [
        {
            "id": "one",
            "category": "direct",
            "question": "Measured question",
            "expected_evidence": [{"evidence_id": "e1"}],
        }
    ]

    benchmark.run_benchmark(cases, k=5, config=config)

    assert calls[:2] == [
        ("semantic", "Opcenter retrieval benchmark warmup"),
        ("hybrid", "Opcenter retrieval benchmark warmup"),
    ]
    assert calls[2:] == [
        ("semantic", "Measured question"),
        ("hybrid", "Measured question"),
    ]


def test_golden_retrieval_cases_have_exact_evidence_and_page_labels() -> None:
    cases = json.loads(
        Path("tests/evaluation_questions.json").read_text(encoding="utf-8")
    )
    labels_by_case = json.loads(
        Path("tests/retrieval_ground_truth.json").read_text(encoding="utf-8")
    )
    evidence_units = {
        unit["evidence_id"]: unit
        for unit in json.loads(
            Path("indexes/evidence_units.json").read_text(encoding="utf-8")
        )
    }

    assert set(labels_by_case) == {case["id"] for case in cases}

    for case in cases:
        labels = labels_by_case[case["id"]]
        assert isinstance(labels, list), case["id"]
        if case["expected_terms"]:
            assert labels, case["id"]
        for label in labels:
            assert label["evidence_id"], case["id"]
            assert label["source_file"].endswith(".pdf"), case["id"]
            assert isinstance(label["pdf_page"], int) and label["pdf_page"] > 0, case["id"]
            unit = evidence_units[label["evidence_id"]]
            assert unit["metadata"]["source_file"] == label["source_file"]
            assert unit["metadata"]["pdf_page"] == label["pdf_page"]
