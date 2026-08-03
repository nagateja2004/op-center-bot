import unittest
from types import SimpleNamespace

from production_monitor import analyze_runs, render_issue


class ProductionMonitorTests(unittest.TestCase):
    def test_healthy_trace_window_does_not_open_issue(self):
        runs = [
            SimpleNamespace(
                id="1",
                name="opcenter_chat",
                latency=2.0,
                error=None,
                total_cost=0.01,
                feedback_stats={"correctness": {"avg": 1.0}},
                url="https://smith.example/runs/1",
            )
        ]
        report = analyze_runs(
            runs,
            project="production",
            lookback_minutes=60,
            max_error_rate=0.05,
            max_p95_latency=30,
            min_feedback_score=0.5,
            feedback_keys=("correctness",),
        )
        self.assertFalse(report["issue_found"])
        self.assertEqual(report["cost_usd"]["average_per_trace"], 0.01)

    def test_errors_latency_and_feedback_create_review_issue(self):
        runs = [
            SimpleNamespace(
                id="bad-run",
                name="opcenter_chat",
                latency=45.0,
                error="provider timeout",
                total_cost=0.02,
                feedback_stats={"correctness": {"avg": 0.0}},
                url="https://smith.example/runs/bad-run",
            )
        ]
        report = analyze_runs(
            runs,
            project="production",
            lookback_minutes=60,
            max_error_rate=0.05,
            max_p95_latency=30,
            min_feedback_score=0.5,
            feedback_keys=("correctness",),
        )
        issue = render_issue(report)
        self.assertTrue(report["issue_found"])
        self.assertEqual(len(report["reasons"]), 3)
        self.assertIn("golden-approved", issue)
        self.assertNotIn("provider timeout", issue)


if __name__ == "__main__":
    unittest.main()
