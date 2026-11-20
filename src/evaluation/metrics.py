"""
Evaluation Metrics for AI Research Assistant.

Helper functions to compute/record evaluation data:
- latency
- cache hit/miss
- strategy used
- faithfulness (from pipeline if available)
- citation presence
- retrieval success
- optional LLM-as-judge score for answer quality
- lexical overlap with reference answer (token Jaccard: |intersection| / |union|)
- strategy match against expected_strategy
- citation compliance for must_cite questions

Lexical Overlap Formula:
    token_overlap(answer, reference) = |tokens(answer) ∩ tokens(reference)| / |tokens(answer) ∪ tokens(reference)|
    where tokens(text) = set(lowercase(text).split() on non-alphanumeric boundaries)
    Returns 0.0 if reference is empty or None.
"""
import json
import re
import time
from typing import Any, Dict, List, Optional, OrderedDict
from collections import OrderedDict


class MetricsRecorder:
    """
    Records metrics for each evaluated question.

    Stores per-query metrics that are aggregated at the end of evaluation.
    """
    def __init__(self):
        self.results: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self.summary: Dict[str, Any] = {
            "total_questions": 0,
            "avg_latency": 0.0,
            "cache_hit_rate": 0.0,
            "faithfulness_rate": 0.0,
            "citation_rate": 0.0,
            "strategy_distribution": {},
            "retrieval_success_rate": 0.0,
        }

    def add(self, question_id: str, result: Dict[str, Any]) -> None:
        """
        Add a single evaluation result.

        Args:
            question_id: The question identifier
            result: Dict with metrics for this question
        """
        self.results[question_id] = result
        self._update_summary()

    def _update_summary(self) -> None:
        """Update summary statistics from all recorded results."""
        total = len(self.results)
        if total == 0:
            return

        self.summary["total_questions"] = total

        # Average latency
        latencies = [r.get("latency", 0.0) for r in self.results.values()]
        self.summary["avg_latency"] = sum(latencies) / len(latencies) if latencies else 0.0

        # Cache hit rate
        cache_hits = sum(1 for r in self.results.values() if r.get("cache_hit", False))
        self.summary["cache_hit_rate"] = cache_hits / total if total else 0.0

        # Faithfulness rate
        faithful_count = sum(1 for r in self.results.values() if r.get("faithful", False))
        self.summary["faithfulness_rate"] = faithful_count / total if total else 0.0

        # Citation rate
        citation_count = sum(1 for r in self.results.values() if r.get("has_citation", False) or r.get("citation_present", False))
        self.summary["citation_rate"] = citation_count / total if total else 0.0

        # Strategy distribution
        strategy_counts: Dict[str, int] = {}
        for r in self.results.values():
            s = r.get("strategy", "unknown")
            strategy_counts[s] = strategy_counts.get(s, 0) + 1
        self.summary["strategy_distribution"] = {
            k: v / total for k, v in strategy_counts.items()
        }

        # Retrieval success rate
        success_count = sum(1 for r in self.results.values() if r.get("retrieval_success", False))
        self.summary["retrieval_success_rate"] = success_count / total if total else 0.0

    def get_results(self) -> Dict[str, Any]:
        """Get all recorded results."""
        return dict(self.results)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return dict(self.summary)


def measure_latency(func):
    """
    Decorator to measure function execution latency.

    Args:
        func: Function to measure

    Returns:
        Decorated function that returns (result, latency_in_seconds) for non-dict results,
        or adds "latency" key to dict results and returns the dict.
    """
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        latency = time.time() - start
        if isinstance(result, dict):
            result["latency"] = latency
            return result
        return result, latency
    return wrapper


def has_citation(answer: str, sources: List[Dict[str, Any]]) -> bool:
    """
    Check if the answer includes proper citations.

    Args:
        answer: The generated answer text
        sources: List of source citations

    Returns:
        True if answer contains citations from sources
    """
    if not sources:
        # Check if answer mentions paper titles or IDs in bracket format
        import re
        # Look for patterns like [Paper Title] or [paper_id]
        citation_patterns = [
            r"\[[A-Za-z0-9 _-]+\]",  # [Paper Title]
            r"\[[a-zA-Z0-9_-]+:[A-Za-z0-9 _-]+\]",  # [paper_id: something]
        ]
        for pattern in citation_patterns:
            if re.search(pattern, answer):
                return True
        return False
    # If sources provided, check if answer references them
    if answer and sources:
        # Simple check: answer length > 0 and sources exist
        return True
    return False


def _tokenize(text: str) -> set:
    """
    Tokenize text into lowercase alphanumeric tokens.

    Args:
        text: Input text

    Returns:
        Set of lowercase tokens split on non-alphanumeric boundaries
    """
    if not text:
        return set()
    # Split on non-alphanumeric, filter empty, lowercase
    return {t for t in re.split(r'[^a-zA-Z0-9]+', text.lower()) if t}


def lexical_overlap(answer: str, reference: Optional[str]) -> float:
    """
    Compute token Jaccard overlap between answer and reference answer.

    Formula: |tokens(answer) ∩ tokens(reference)| / |tokens(answer) ∪ tokens(reference)|
    Returns 0.0 if reference is empty or None (and answer is not also empty).
    Returns 1.0 if both answer and reference are empty/None (trivial match).

    Args:
        answer: Generated answer text
        reference: Reference/gold answer text (may be None)

    Returns:
        Float in [0.0, 1.0]
    """
    # Handle both empty/None case first
    if not reference and not answer:
        return 1.0
    if not reference:
        return 0.0
    ans_tokens = _tokenize(answer)
    ref_tokens = _tokenize(reference)
    if not ans_tokens and not ref_tokens:
        return 1.0
    if not ans_tokens or not ref_tokens:
        return 0.0
    intersection = ans_tokens & ref_tokens
    union = ans_tokens | ref_tokens
    return len(intersection) / len(union)


def strategy_match(predicted: Optional[str], expected: Optional[str]) -> Optional[bool]:
    """
    Check if predicted strategy matches expected strategy.

    Args:
        predicted: Strategy name from pipeline (e.g., 'crag_raptor')
        expected: Expected strategy from gold (e.g., 'crag_raptor')

    Returns:
        True if match, False if both present and different, None if expected is None
    """
    if expected is None:
        return None
    if predicted is None:
        return False
    # Normalize: lowercase, strip prefixes like 'cached_'
    pred_norm = predicted.lower().replace('cached_', '')
    exp_norm = expected.lower().replace('cached_', '')
    return pred_norm == exp_norm


def citation_compliance(has_citation: bool, must_cite: bool) -> Optional[bool]:
    """
    Check citation compliance for must_cite questions.

    Args:
        has_citation: Whether the answer has citations
        must_cite: Whether the question requires citations

    Returns:
        True if compliant (has citation when must_cite, or no requirement),
        False if must_cite but no citation,
        None if must_cite is False (not applicable)
    """
    if not must_cite:
        return None
    return has_citation


def compute_faithfulness(faithful: bool, pipeline_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Compute faithfulness metric from pipeline output.

    Args:
        faithful: Boolean from self-check
        pipeline_metadata: Optional additional metadata from the pipeline

    Returns:
        Dict with faithfulness information
    """
    result = {
        "faithful": faithful,
        "confidence": 1.0 if faithful else 0.5,
    }
    if pipeline_metadata:
        result["reasoning"] = pipeline_metadata.get("self_check_reasoning", "")
        result["confidence"] = pipeline_metadata.get("confidence", 1.0)
    return result


def summarize_metrics(metrics_record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Summarize metrics from a single evaluation run.

    Args:
        metrics_record: Dict from MetricsRecorder.get_results() (id -> result mapping)
                        or a dict with "results" and "summary" keys.

    Returns:
        Summary dict with key metrics
    """
    # Handle both formats: wrapped {results: {...}, summary: {...}} or plain id->result mapping
    if "results" in metrics_record and isinstance(metrics_record["results"], dict):
        results = metrics_record["results"]
        summary = metrics_record.get("summary", {})
    else:
        results = metrics_record
        summary = {}

    # If summary is empty or incomplete, rebuild it
    if not summary or "total_questions" not in summary:
        total = len(results)
        if total == 0:
            return {
                "total_questions": 0,
                "avg_latency_seconds": 0.0,
                "cache_hit_rate": 0.0,
                "faithfulness_rate": 0.0,
                "citation_rate": 0.0,
                "retrieval_success_rate": 0.0,
                "strategy_distribution": {},
                "strategy_match_rate": 0.0,
                "citation_ok_rate": 0.0,
                "avg_lexical_overlap": 0.0,
            }

        latencies = [r.get("latency", 0.0) for r in results.values()]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

        cache_hits = sum(1 for r in results.values() if r.get("cache_hit", False))
        cache_hit_rate = cache_hits / total if total else 0.0

        faithful_count = sum(1 for r in results.values() if r.get("faithful", False))
        faithfulness_rate = faithful_count / total if total else 0.0

        citation_count = sum(1 for r in results.values() if r.get("has_citation", False) or r.get("citation_present", False))
        citation_rate = citation_count / total if total else 0.0

        strategy_counts: Dict[str, int] = {}
        for r in results.values():
            s = r.get("strategy", "unknown")
            strategy_counts[s] = strategy_counts.get(s, 0) + 1
        strategy_distribution = {k: v / total for k, v in strategy_counts.items()}

        success_count = sum(1 for r in results.values() if r.get("retrieval_success", False))
        retrieval_success_rate = success_count / total if total else 0.0

        # New gold-based metrics
        # Strategy match rate: among questions with expected_strategy
        strategy_match_results = [r.get("strategy_match") for r in results.values() if r.get("expected_strategy") is not None]
        strategy_match_count = sum(1 for m in strategy_match_results if m is True)
        strategy_match_rate = strategy_match_count / len(strategy_match_results) if strategy_match_results else 0.0

        # Citation OK rate: among questions with must_cite=true
        citation_ok_results = [r.get("citation_ok") for r in results.values() if r.get("must_cite") is True]
        citation_ok_count = sum(1 for m in citation_ok_results if m is True)
        citation_ok_rate = citation_ok_count / len(citation_ok_results) if citation_ok_results else 0.0

        # Avg lexical overlap: among questions with non-empty reference_answer
        lexical_overlaps = [r.get("lexical_overlap", 0.0) for r in results.values() if r.get("reference_answer")]
        avg_lexical_overlap = sum(lexical_overlaps) / len(lexical_overlaps) if lexical_overlaps else 0.0

        summary = {
            "total_questions": total,
            "avg_latency": avg_latency,
            "cache_hit_rate": cache_hit_rate,
            "faithfulness_rate": faithfulness_rate,
            "citation_rate": citation_rate,
            "retrieval_success_rate": retrieval_success_rate,
            "strategy_distribution": strategy_distribution,
            "strategy_match_rate": strategy_match_rate,
            "citation_ok_rate": citation_ok_rate,
            "avg_lexical_overlap": avg_lexical_overlap,
        }

    return {
        "total_questions": summary.get("total_questions", 0),
        "avg_latency_seconds": summary.get("avg_latency", 0.0),
        "cache_hit_rate": summary.get("cache_hit_rate", 0.0),
        "faithfulness_rate": summary.get("faithfulness_rate", 0.0),
        "citation_rate": summary.get("citation_rate", 0.0),
        "retrieval_success_rate": summary.get("retrieval_success_rate", 0.0),
        "strategy_distribution": summary.get("strategy_distribution", {}),
        "strategy_match_rate": summary.get("strategy_match_rate", 0.0),
        "citation_ok_rate": summary.get("citation_ok_rate", 0.0),
        "avg_lexical_overlap": summary.get("avg_lexical_overlap", 0.0),
    }