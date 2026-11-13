"""
Evaluation package for AI Research Assistant.

Provides metrics computation, evaluation orchestration, and reporting.
"""
from src.evaluation.metrics import (
    MetricsRecorder,
    measure_latency,
    has_citation,
    compute_faithfulness,
    summarize_metrics,
)
from src.evaluation.evaluator import (
    load_questions,
    run_evaluation,
    print_summary,
    main,
)

__all__ = [
    "MetricsRecorder",
    "measure_latency",
    "has_citation",
    "compute_faithfulness",
    "summarize_metrics",
    "load_questions",
    "run_evaluation",
    "print_summary",
    "main",
]