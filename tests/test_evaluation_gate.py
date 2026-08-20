import unittest
import json
from pathlib import Path
from types import SimpleNamespace

from evaluation import (
    DEFAULT_THRESHOLDS,
    evidence_status_hit,
    evaluate_gate,
    judge_response,
    parse_semantic_judgment,
    validated_backend_url,
)


def report(**overrides):
    value = {
        "cases": 50,
        "passed": 45,
        "errors": 0,
        "answer_term_accuracy": 0.92,
        "manual_routing_accuracy": 0.94,
        "status_accuracy": 0.96,
        "citation_id_accuracy": 0.98,
        "latency_seconds": {"p95": 12.0},
    }
    value.update(overrides)
    return value


class EvaluationGateTests(unittest.TestCase):
    def test_backend_url_rejects_non_http_and_embedded_credentials(self):
        self.assertEqual(
            validated_backend_url("https://staging.example.com/"),
            "https://staging.example.com",
        )
        for value in ("file:///tmp/result", "https://user:secret@example.com"):
            with self.assertRaises(ValueError):
                validated_backend_url(value)

    def test_golden_dataset_has_at_least_50_unique_labeled_cases(self):
        cases = json.loads(
            Path("tests/evaluation_questions.json").read_text(encoding="utf-8")
        )

        self.assertGreaterEqual(len(cases), 50)
        self.assertEqual(len({case["id"] for case in cases}), len(cases))
        self.assertTrue(
            all(case.get("question") and case.get("expected_status") for case in cases)
        )

    def test_gate_passes_a_healthy_report(self):
        gate = evaluate_gate(report(), DEFAULT_THRESHOLDS)

        self.assertTrue(gate["passed"])
        self.assertEqual(gate["failures"], [])

    def test_gate_reports_threshold_and_baseline_regressions(self):
        candidate = report(
            passed=40,
            errors=1,
            citation_id_accuracy=0.89,
            latency_seconds={"p95": 80.0},
        )
        gate = evaluate_gate(
            candidate,
            DEFAULT_THRESHOLDS,
            baseline=report(),
            max_regression=0.03,
        )

        self.assertFalse(gate["passed"])
        self.assertTrue(any("pass_rate" in failure for failure in gate["failures"]))
        self.assertTrue(any("citation_id_accuracy" in failure for failure in gate["failures"]))
        self.assertTrue(any("p95_latency_seconds" in failure for failure in gate["failures"]))
        self.assertTrue(any("errors" in failure for failure in gate["failures"]))
        self.assertTrue(any("regressed" in failure for failure in gate["failures"]))

    def test_semantic_judgment_is_parsed_and_critical_scores_must_pass(self):
        judgment = parse_semantic_judgment(json.dumps({
            "scores": {
                "relevance": 5,
                "correctness": 4,
                "faithfulness": 4,
                "completeness": 4,
                "clarity": 5,
            },
            "reason": "Grounded and complete.",
        }))

        self.assertTrue(judgment["passed"])
        self.assertEqual(judgment["overall_score"], 4.4)

    def test_gate_fails_semantic_threshold_or_judge_error(self):
        candidate = report(semantic_quality={
            "enabled": True,
            "overall_score": 3.8,
            "pass_rate": 0.80,
            "errors": 1,
        })

        gate = evaluate_gate(candidate, DEFAULT_THRESHOLDS)

        self.assertFalse(gate["passed"])
        self.assertTrue(any("semantic_score" in failure for failure in gate["failures"]))
        self.assertTrue(any("semantic_pass_rate" in failure for failure in gate["failures"]))
        self.assertTrue(any("judge_errors" in failure for failure in gate["failures"]))

    def test_judge_call_is_temperature_zero_json_and_tracks_tokens(self):
        class Completions:
            def create(self, **kwargs):
                self.kwargs = kwargs
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({
                        "scores": {
                            "relevance": 5,
                            "correctness": 4,
                            "faithfulness": 4,
                            "completeness": 4,
                            "clarity": 5,
                        },
                        "reason": "Grounded.",
                    })))],
                    usage=SimpleNamespace(
                        prompt_tokens=100,
                        completion_tokens=20,
                        total_tokens=120,
                    ),
                )

        completions = Completions()
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        judgment = judge_response(
            client,
            "openai/gpt-oss-20b",
            {"question": "What is a Factory?", "expected_status": "sufficient"},
            {
                "answer": "A Factory is a division [S1].",
                "evidence": {"status": "sufficient"},
                "evaluation_context": [
                    {"source_id": "S1", "text": "A Factory is a division."}
                ],
            },
        )

        self.assertEqual(completions.kwargs["temperature"], 0)
        self.assertEqual(completions.kwargs["max_tokens"], 1_000)
        self.assertEqual(completions.kwargs["response_format"], {"type": "json_object"})
        self.assertEqual(judgment["tokens"]["total"], 120)

    def test_judge_retries_one_invalid_response(self):
        class Completions:
            calls = 0

            def create(self, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    raise ValueError("invalid JSON")
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({
                        "scores": {dimension: 5 for dimension in (
                            "relevance", "correctness", "faithfulness", "completeness", "clarity"
                        )},
                        "reason": "Grounded.",
                    })))],
                    usage=SimpleNamespace(
                        prompt_tokens=10,
                        completion_tokens=5,
                        total_tokens=15,
                    ),
                )

        completions = Completions()
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

        judgment = judge_response(
            client,
            "openai/gpt-oss-20b",
            {"question": "What is a Factory?", "expected_status": "sufficient"},
            {
                "answer": "A Factory is a division [S1].",
                "evidence": {"status": "sufficient"},
                "evaluation_context": [
                    {"source_id": "S1", "text": "A Factory is a division."}
                ],
            },
        )

        self.assertEqual(completions.calls, 2)
        self.assertTrue(judgment["passed"])

    def test_case_can_explicitly_accept_partial_or_sufficient_status(self):
        case = {
            "expected_status": "sufficient",
            "accepted_statuses": ["sufficient", "partial"],
        }

        self.assertTrue(evidence_status_hit(case, {"evidence": {"status": "partial"}}))
        self.assertFalse(
            evidence_status_hit(case, {"evidence": {"status": "in_scope_insufficient"}})
        )


if __name__ == "__main__":
    unittest.main()
