from collections import Counter
import json
from pathlib import Path

from evaluation import expected_output, render_html


def test_evaluation_question_category_counts() -> None:
    questions = json.loads(Path("tests/evaluation_questions.json").read_text(encoding="utf-8"))
    counts = Counter(question["category"] for question in questions)

    assert len(questions) >= 50
    assert len({question["id"] for question in questions}) == len(questions)
    assert counts["direct"] >= 7
    assert counts["indirect"] >= 8
    assert counts["follow_up"] >= 5
    assert counts["procedure"] >= 5
    assert counts["table_field"] >= 5
    assert counts["comparison"] >= 4
    assert counts["unsupported"] >= 3
    assert counts["irrelevant"] >= 3
    assert counts["electronics"] >= 5
    assert counts["discrete"] >= 5


def test_golden_cases_have_required_labels() -> None:
    questions = json.loads(Path("tests/evaluation_questions.json").read_text(encoding="utf-8"))
    for case in questions:
        assert case["id"]
        assert case["category"]
        assert case["question"].strip()
        assert case["expected_status"] in {
            "sufficient", "partial", "in_scope_insufficient", "out_of_scope"
        }
        assert isinstance(case["expected_terms"], list)


def test_shareable_report_contains_test_case_fields_and_escapes_answers() -> None:
    report = {
        "generated_at": "2026-08-02T00:00:00+00:00",
        "cases": 1,
        "completed": 1,
        "errors": 0,
        "passed": 1,
        "answer_term_accuracy": 1.0,
        "manual_routing_accuracy": 1.0,
        "status_accuracy": 1.0,
        "citation_id_accuracy": 1.0,
        "diagram_render_accuracy": None,
        "latency_seconds": {"mean": 1.0, "median": 1.0, "p95": 1.0},
        "results": [{
            "id": "direct_01", "category": "direct", "passed": True,
            "input": "What is a Factory?", "expected_output": {"evidence_status": "sufficient"},
            "actual_output": "<script>alert(1)</script>", "retrieval_context": [],
            "term_hit": True, "status_hit": True, "latency_seconds": 1.0,
        }],
    }
    html = render_html(report)
    assert "Gold expectations" in html
    assert "Actual model answer" in html
    assert "Retrieved context" in html
    assert "<script>" not in html
    assert expected_output({
        "expected_status": "sufficient", "expected_terms": ["factory"]
    })["any_expected_terms"] == ["factory"]
