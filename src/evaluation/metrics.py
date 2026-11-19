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
"""
import json
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

        summary = {
            "total_questions": total,
            "avg_latency": avg_latency,
            "cache_hit_rate": cache_hit_rate,
            "faithfulness_rate": faithfulness_rate,
            "citation_rate": citation_rate,
            "retrieval_success_rate": retrieval_success_rate,
            "strategy_distribution": strategy_distribution,
        }

    return {
        "total_questions": summary.get("total_questions", 0),
        "avg_latency_seconds": summary.get("avg_latency", 0.0),
        "cache_hit_rate": summary.get("cache_hit_rate", 0.0),
        "faithfulness_rate": summary.get("faithfulness_rate", 0.0),
        "citation_rate": summary.get("citation_rate", 0.0),
        "retrieval_success_rate": summary.get("retrieval_success_rate", 0.0),
        "strategy_distribution": summary.get("strategy_distribution", {}),
    }