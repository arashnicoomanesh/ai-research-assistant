"""
Evaluation harness for AI Research Assistant.

Loads questions from JSONL, runs each through the existing pipeline,
collects structured results, and produces reports.
"""
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.core.llm import get_llm
from src.core.embeddings import get_embeddings
from src.retrieval.hybrid_retriever import get_retriever
from src.retrieval.grader import get_grader
from src.retrieval.query_rewriter import get_rewriter
from src.retrieval.semantic_cache import get_cache
from src.retrieval.corrective_retriever import get_corrective_retriever
from src.retrieval.multi_step_retriever import get_multi_step_retriever
from src.retrieval.raptor_retriever import get_raptor_retriever
from src.routing.adaptive_router import get_router
from src.generation.answer_generator import get_generator
from src.evaluation.metrics import (
    MetricsRecorder,
    measure_latency,
    has_citation,
    compute_faithfulness,
    summarize_metrics,
)

logger = logging.getLogger("eval")


def load_questions(jsonl_path: str) -> List[Dict[str, Any]]:
    """
    Load evaluation questions from JSONL file.

    Args:
        jsonl_path: Path to questions JSONL file

    Returns:
        List of question dicts
    """
    questions = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    questions.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping malformed line: {e}")
    logger.info(f"Loaded {len(questions)} questions from {jsonl_path}")
    return questions


@measure_latency
def run_single_question(
    question_data: Dict[str, Any],
    assistant,
) -> Dict[str, Any]:
    """
    Run a single question through the pipeline and collect metrics.

    Args:
        question_data: Question data from JSONL
        assistant: ResearchAssistant instance

    Returns:
        Dict with metrics and results for this question
    """
    question_id = question_data["id"]
    question = question_data["question"]
    question_type = question_data["type"]
    must_cite = question_data.get("must_cite", False)

    result: Dict[str, Any] = {
        "question_id": question_id,
        "question": question,
        "type": question_type,
        "must_cite": must_cite,
        "strategy": None,
        "answer": "",
        "sources": [],
        "faithful": None,
        "latency": 0.0,
        "cache_hit": False,
        "retrieval_success": False,
        "citation_present": False,
        "has_citation": False,
        "crag_metadata": None,
        "raptor_metadata": None,
    }

    # Step 1: Check cache
    cached = assistant.cache.get(question)
    if cached:
        result["cache_hit"] = True
        result["answer"] = cached["answer"]
        result["sources"] = cached["sources"]
        result["strategy"] = cached["strategy"]
        result["faithful"] = True
        result["latency"] = 0.01  # Cache lookup is fast
        # Still record citation info
        cite = has_citation(result["answer"], result["sources"])
        result["citation_present"] = cite
        result["has_citation"] = cite
        return result

    # Step 2: Run through pipeline
    import time
    start_time = time.time()

    try:
        answer = assistant.ask(question)

        latency = time.time() - start_time
        result["latency"] = latency

        result["answer"] = answer.answer
        result["sources"] = answer.sources
        result["strategy"] = answer.strategy
        result["faithful"] = answer.faithful
        cite = has_citation(answer.answer, answer.sources)
        result["citation_present"] = cite
        result["has_citation"] = cite

        # Record CRAG/MP/RAptor metadata
        ms_meta = getattr(answer, "multi_step_metadata", None)
        if ms_meta:
            result["raptor_metadata"] = ms_meta
        if "crag_" in answer.strategy:
            result["crag_metadata"] = {
                "correction_applied": answer.strategy.replace("crag_", ""),
            }

        # Record successful retrieval
        result["retrieval_success"] = len(answer.sources) > 0

    except Exception as e:
        latency = time.time() - start_time
        result["latency"] = latency
        result["answer"] = f"Error: {str(e)}"
        result["sources"] = []
        result["strategy"] = "error"
        result["faithful"] = False

    return result


def run_evaluation(
    questions_path: str,
    output_path: Optional[str] = None,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run evaluation on all questions from JSONL file.

    Args:
        questions_path: Path to questions JSONL file
        output_path: Optional path to save detailed JSON report
        limit: Optional limit on number of questions to evaluate

    Returns:
        Dict with overall results and summary
    """
    # Load questions
    questions = load_questions(questions_path)

    # Apply limit if specified
    if limit:
        questions = questions[:limit]

    # Initialize pipeline components
    llm = get_llm()
    embeddings = get_embeddings()
    retriever = get_retriever()
    grader = get_grader()
    rewriter = get_rewriter()
    cache = get_cache(embeddings=embeddings)
    corrective_retriever = get_corrective_retriever(
        retriever=retriever,
        grader=grader,
        rewriter=rewriter,
    )
    multi_step_retriever = get_multi_step_retriever(
        corrective_retriever=corrective_retriever,
    )
    raptor_retriever = get_raptor_retriever(embeddings=embeddings)
    router = get_router()
    generator = get_generator()

    # Create assistant instance
    from src.main import ResearchAssistant
    assistant = ResearchAssistant()

    # Initialize metrics recorder
    metrics = MetricsRecorder()

    logger.info(f"Starting evaluation of {len(questions)} questions")

    # Run each question
    for i, question_data in enumerate(questions, 1):
        logger.info(f"Processing question {i}/{len(questions)}: {question_data['id']}")

        # Run through pipeline
        result = run_single_question(question_data, assistant)

        # Record in metrics
        metrics.add(question_data["id"], result)

        # Log progress
        cache_info = "HIT" if result["cache_hit"] else "MISS"
        logger.info(
            f"Q{question_data['id']}: strategy={result['strategy']}, "
            f"cache={cache_info}, faithful={result['faithful']}, "
            f"latency={result['latency']:.2f}s"
        )

    # Generate summary
    summary = summarize_metrics(metrics.get_results())

    # Prepare final results
    overall_results = {
        "questions_evaluated": len(questions),
        "summary": summary,
        "detailed_results": dict(metrics.get_results()),
    }

    # Save detailed report if output path specified
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(overall_results, f, indent=2, ensure_ascii=False)
        logger.info(f"Detailed report saved to {output_path}")

    return overall_results


def print_summary(overall_results: Dict[str, Any]) -> None:
    """
    Print a human-readable summary table.

    Args:
        overall_results: Results dict from run_evaluation
    """
    summary = overall_results.get("summary", {})
    detailed = overall_results.get("detailed_results", {})

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    print(f"\nTotal questions: {summary.get('total_questions', 0)}")
    print(f"Average latency: {summary.get('avg_latency_seconds', 0.0):.2f}s")
    print(f"Cache hit rate: {summary.get('cache_hit_rate', 0.0):.1%}")
    print(f"Faithfulness rate: {summary.get('faithfulness_rate', 0.0):.1%}")
    print(f"Citation rate: {summary.get('citation_rate', 0.0):.1%}")
    print(f"Retrieval success rate: {summary.get('retrieval_success_rate', 0.0):.1%}")

    print(f"\nStrategy distribution:")
    for strategy, rate in summary.get("strategy_distribution", {}).items():
        print(f"  {strategy}: {rate:.1%}")

    print(f"\nPer-question details:")
    for qid, result in detailed.items():
        print(f"  {qid}: strategy={result.get('strategy', 'N/A')}, "
              f"latency={result.get('latency', 0):.2f}s, "
              f"cache_hit={result.get('cache_hit', False)}, "
              f"faithful={result.get('faithful', False)}")

    print("\n" + "=" * 60)


def main():
    """Main entry point for evaluation harness."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run evaluation harness for AI Research Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_eval.py                          # Run all questions
  python scripts/run_eval.py --limit 5               # Run first 5 questions
  python scripts/run_eval.py --output report.json    # Save detailed report
        """
    )
    parser.add_argument(
        "questions_path",
        nargs="?",
        default="data/eval/questions.jsonl",
        help="Path to evaluation questions JSONL file",
    )
    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=None,
        help="Limit number of questions to evaluate",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="data/eval/reports/evaluation_report.json",
        help="Path to save detailed JSON report",
    )

    args = parser.parse_args()

    # Run evaluation
    overall_results = run_evaluation(
        questions_path=args.questions_path,
        output_path=args.output,
        limit=args.limit,
    )

    # Print summary
    print_summary(overall_results)


if __name__ == "__main__":
    main()