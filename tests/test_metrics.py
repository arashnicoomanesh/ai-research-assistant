"""
Smoke test for evaluation metrics.

Verifies that latency measurement and metrics recording work correctly.
"""
from src.evaluation.metrics import measure_latency, MetricsRecorder, summarize_metrics


class TestMetrics:
    """Tests for evaluation metrics."""

    def test_measure_latency_decorator_preserves_dict_return(self):
        """@measure_latency on dict-returning fn should add latency key."""
        @measure_latency
        def sample_fn(x):
            return {"result": x * 2}

        result = sample_fn(5)
        assert isinstance(result, dict)
        assert "latency" in result
        assert result["latency"] >= 0
        assert result["result"] == 10

    def test_measure_latency_decorator_preserves_return_value(self):
        """Return value should be preserved with latency added."""
        @measure_latency
        def sample_fn():
            return {"answer": "test", "score": 0.9}

        result = sample_fn()
        assert result["answer"] == "test"
        assert result["score"] == 0.9
        assert "latency" in result
        assert result["latency"] >= 0

    def test_metrics_recorder_basic(self):
        """MetricsRecorder should collect and summarize metrics."""
        recorder = MetricsRecorder()

        # Add some synthetic results
        recorder.add("q1", {
            "latency": 10.5,
            "faithful": True,
            "citation_present": True,
            "has_citation": True,
            "retrieval_success": True,
            "strategy": "single_shot",
        })
        recorder.add("q2", {
            "latency": 15.2,
            "faithful": False,
            "citation_present": True,
            "has_citation": True,
            "retrieval_success": True,
            "strategy": "crag_rewrite_merge",
        })
        recorder.add("q3", {
            "latency": 8.1,
            "faithful": True,
            "citation_present": False,
            "has_citation": False,
            "retrieval_success": False,
            "strategy": "no_retrieval",
        })

        summary = summarize_metrics(recorder.get_results())

        assert summary["total_questions"] == 3
        assert summary["avg_latency_seconds"] > 0
        assert abs(summary["faithfulness_rate"] - 2/3) < 0.01
        assert abs(summary["citation_rate"] - 2/3) < 0.01
        assert abs(summary["retrieval_success_rate"] - 2/3) < 0.01
        assert "strategy_distribution" in summary
        assert abs(summary["strategy_distribution"]["single_shot"] - 1/3) < 0.01
        assert abs(summary["strategy_distribution"]["crag_rewrite_merge"] - 1/3) < 0.01
        assert abs(summary["strategy_distribution"]["no_retrieval"] - 1/3) < 0.01

    def test_citation_present_and_has_citation_both_work(self):
        """Both citation_present and has_citation fields should be handled."""
        recorder = MetricsRecorder()

        recorder.add("q1", {
            "latency": 10.0,
            "faithful": True,
            "citation_present": True,
            "has_citation": True,
            "retrieval_success": True,
            "strategy": "single_shot",
        })
        recorder.add("q2", {
            "latency": 10.0,
            "faithful": True,
            "citation_present": False,
            "has_citation": True,
            "retrieval_success": True,
            "strategy": "single_shot",
        })

        summary = summarize_metrics(recorder.get_results())
        assert "citation_rate" in summary
        # Both should count as having citations (either field True)
        assert summary["citation_rate"] == 1.0

    def test_summarize_metrics_with_empty_results(self):
        """Empty results should return zeroed summary."""
        summary = summarize_metrics({})
        assert summary["total_questions"] == 0
        assert summary["avg_latency_seconds"] == 0.0
        assert summary["faithfulness_rate"] == 0.0
        assert summary["citation_rate"] == 0.0
        assert summary["retrieval_success_rate"] == 0.0