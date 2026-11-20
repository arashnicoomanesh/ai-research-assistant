"""
Offline smoke tests for eval gold fields and metrics.

Tests:
- strategy_match true/false/None
- must_cite + citation_ok
- lexical_overlap on known fixture (token Jaccard)
- summarize_metrics includes new rates when fed synthetic results
"""
import pytest
from src.evaluation.metrics import (
    _tokenize,
    lexical_overlap,
    strategy_match,
    citation_compliance,
    summarize_metrics,
    MetricsRecorder,
)


class TestTokenize:
    """Tests for tokenization."""

    def test_tokenize_basic(self):
        tokens = _tokenize("Hello World")
        assert tokens == {"hello", "world"}

    def test_tokenize_punctuation(self):
        tokens = _tokenize("Hello, world!")
        assert tokens == {"hello", "world"}

    def test_tokenize_numbers(self):
        tokens = _tokenize("BERT-2 and GPT-3")
        assert tokens == {"bert", "2", "and", "gpt", "3"}

    def test_tokenize_empty(self):
        tokens = _tokenize("")
        assert tokens == set()

    def test_tokenize_none(self):
        tokens = _tokenize(None)
        assert tokens == set()


class TestLexicalOverlap:
    """Tests for lexical overlap (token Jaccard)."""

    def test_identical_strings(self):
        score = lexical_overlap("hello world", "hello world")
        assert score == 1.0

    def test_no_overlap(self):
        score = lexical_overlap("hello world", "foo bar")
        assert score == 0.0

    def test_partial_overlap(self):
        # hello world vs hello foo -> intersection={hello}, union={hello,world,foo} = 1/3
        score = lexical_overlap("hello world", "hello foo")
        assert abs(score - 1/3) < 0.01

    def test_reference_empty(self):
        score = lexical_overlap("hello world", "")
        assert score == 0.0

    def test_reference_none(self):
        score = lexical_overlap("hello world", None)
        assert score == 0.0

    def test_both_empty(self):
        score = lexical_overlap("", "")
        assert score == 1.0  # Both empty = trivial match

    def test_case_insensitive(self):
        score = lexical_overlap("Hello World", "hello world")
        assert score == 1.0

    def test_known_fixture(self):
        """Known fixture: expected 0.25"""
        # "the cat sat" vs "the dog sat" -> intersection={the, sat}, union={the, cat, sat, dog} = 2/4 = 0.5
        score = lexical_overlap("the cat sat", "the dog sat")
        assert abs(score - 0.5) < 0.01


class TestStrategyMatch:
    """Tests for strategy matching."""

    def test_exact_match(self):
        assert strategy_match("crag_raptor", "crag_raptor") is True

    def test_cached_prefix_stripped(self):
        assert strategy_match("cached_crag_raptor", "crag_raptor") is True

    def test_cached_on_expected(self):
        assert strategy_match("crag_raptor", "cached_crag_raptor") is True

    def test_mismatch(self):
        assert strategy_match("single_shot", "crag_raptor") is False

    def test_expected_none(self):
        assert strategy_match("single_shot", None) is None

    def test_predicted_none(self):
        assert strategy_match(None, "single_shot") is False

    def test_both_none(self):
        assert strategy_match(None, None) is None

    def test_case_insensitive(self):
        assert strategy_match("CRAG_RAPTOR", "crag_raptor") is True


class TestCitationCompliance:
    """Tests for citation compliance."""

    def test_must_cite_true_has_citation(self):
        assert citation_compliance(True, True) is True

    def test_must_cite_true_no_citation(self):
        assert citation_compliance(False, True) is False

    def test_must_cite_false_has_citation(self):
        assert citation_compliance(True, False) is None

    def test_must_cite_false_no_citation(self):
        assert citation_compliance(False, False) is None


class TestSummarizeMetrics:
    """Tests for summarize_metrics with gold fields."""

    def test_empty_results(self):
        summary = summarize_metrics({})
        assert summary["total_questions"] == 0
        assert summary["strategy_match_rate"] == 0.0
        assert summary["citation_ok_rate"] == 0.0
        assert summary["avg_lexical_overlap"] == 0.0

    def test_strategy_match_rate(self):
        results = {
            "q1": {"strategy": "single_shot", "expected_strategy": "single_shot", "strategy_match": True, "must_cite": False, "reference_answer": "ref", "has_citation": False, "citation_present": False},
            "q2": {"strategy": "crag_raptor", "expected_strategy": "single_shot", "strategy_match": False, "must_cite": False, "reference_answer": "ref", "has_citation": False, "citation_present": False},
            "q3": {"strategy": "multi_step", "expected_strategy": None, "strategy_match": None, "must_cite": False, "reference_answer": "ref", "has_citation": False, "citation_present": False},
        }
        summary = summarize_metrics(results)
        # q1: match=True, q2: match=False, q3: expected=None (excluded)
        # rate = 1/2 = 0.5
        assert summary["strategy_match_rate"] == 0.5

    def test_citation_ok_rate(self):
        results = {
            "q1": {"strategy": "single_shot", "must_cite": True, "citation_ok": True, "has_citation": True, "citation_present": True, "expected_strategy": None, "reference_answer": ""},
            "q2": {"strategy": "crag_raptor", "must_cite": True, "citation_ok": False, "has_citation": False, "citation_present": False, "expected_strategy": None, "reference_answer": ""},
            "q3": {"strategy": "multi_step", "must_cite": False, "citation_ok": None, "has_citation": False, "citation_present": False, "expected_strategy": None, "reference_answer": ""},
        }
        summary = summarize_metrics(results)
        # q1: ok=True, q2: ok=False, q3: must_cite=False (excluded)
        # rate = 1/2 = 0.5
        assert summary["citation_ok_rate"] == 0.5

    def test_avg_lexical_overlap(self):
        results = {
            "q1": {"strategy": "single_shot", "reference_answer": "hello world", "answer": "hello world", "lexical_overlap": 1.0, "expected_strategy": None, "must_cite": False, "has_citation": False, "citation_present": False},
            "q2": {"strategy": "crag_raptor", "reference_answer": "hello world", "answer": "foo bar", "lexical_overlap": 0.0, "expected_strategy": None, "must_cite": False, "has_citation": False, "citation_present": False},
            "q3": {"strategy": "multi_step", "reference_answer": "", "answer": "something", "lexical_overlap": 0.0, "expected_strategy": None, "must_cite": False, "has_citation": False, "citation_present": False},
        }
        summary = summarize_metrics(results)
        # q1: overlap=1.0, q2: overlap=0.0, q3: reference="" (excluded)
        # avg = (1.0 + 0.0) / 2 = 0.5
        assert summary["avg_lexical_overlap"] == 0.5

    def test_mixed_with_wrapped_format(self):
        """Test summarize_metrics handles wrapped {results, summary} format."""
        metrics_record = {
            "results": {
                "q1": {"strategy": "single_shot", "expected_strategy": "single_shot", "must_cite": False, "reference_answer": "ref"},
            },
            "summary": {}
        }
        summary = summarize_metrics(metrics_record)
        assert summary["total_questions"] == 1

    def test_all_fields_present(self):
        """Verify all expected summary keys exist."""
        results = {
            "q1": {"strategy": "single_shot", "expected_strategy": "single_shot", "must_cite": True, "has_citation": True, "reference_answer": "ref", "answer": "ref"},
        }
        summary = summarize_metrics(results)
        expected_keys = [
            "total_questions", "avg_latency_seconds", "cache_hit_rate",
            "faithfulness_rate", "citation_rate", "retrieval_success_rate",
            "strategy_distribution", "strategy_match_rate", "citation_ok_rate", "avg_lexical_overlap"
        ]
        for key in expected_keys:
            assert key in summary, f"Missing key: {key}"


class TestMetricsRecorderIntegration:
    """Integration test for MetricsRecorder with gold fields."""

    def test_recorder_preserves_gold_fields(self):
        recorder = MetricsRecorder()
        result = {
            "question_id": "q1",
            "strategy": "single_shot",
            "expected_strategy": "single_shot",
            "must_cite": True,
            "has_citation": True,
            "reference_answer": "hello world",
            "answer": "hello world",
            "latency": 1.0,
            "cache_hit": False,
            "faithful": True,
            "retrieval_success": True,
            "has_citation": True,
            "citation_present": True,
            "strategy_match": True,
            "citation_ok": True,
            "lexical_overlap": 1.0,
        }
        recorder.add("q1", result)
        summary = recorder.get_summary()
        assert summary["total_questions"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])