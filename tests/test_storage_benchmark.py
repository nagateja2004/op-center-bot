import pytest

from benchmarks.run_retrieval_benchmark import latency_summary, reduction


def test_latency_percentiles_use_all_samples():
    result = latency_summary([1, 2, 3, 4, 5])
    assert result == pytest.approx({'mean_ms': 3, 'p50_ms': 3, 'p95_ms': 4.8, 'p99_ms': 4.96})


def test_reduction_preserves_slowdowns_and_rejects_zero_baseline():
    assert reduction(100, 60) == 40
    assert reduction(100, 120) == -20
    assert reduction(0, 20) is None
