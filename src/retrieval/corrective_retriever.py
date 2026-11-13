"""
Corrective Retriever for CRAG (Corrective RAG).

Orchestrates the corrective retrieval logic:
1. Initial retrieval
2. Document grading
3. Overall retrieval quality evaluation
4. Corrective actions based on evaluation:
   - "correct": Refine and return best chunks
   - "ambiguous": Rewrite query + second retrieval, merge results
   - "incorrect": Strong rewrite + re-retrieve, return empty if still weak
"""
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from src.core.llm import OllamaLLM, get_llm
from src.core.config import config
from src.retrieval.hybrid_retriever import HybridRetriever, get_retriever
from src.retrieval.grader import DocumentGrader, get_grader
from src.retrieval.query_rewriter import QueryRewriter, get_rewriter
from src.retrieval.retrieval_evaluator import RetrievalEvaluator, get_evaluator, RetrievalEvaluation

logger = logging.getLogger(__name__)


@dataclass
class CorrectiveRetrievalResult:
    """Result of corrective retrieval with metadata."""
    chunks: List[Dict[str, Any]]
    evaluation: RetrievalEvaluation
    correction_applied: str  # "none", "refine", "rewrite_merge", "strong_rewrite"
    rewrite_used: Optional[str] = None
    total_retrievals: int = 1


class CorrectiveRetriever:
    """
    Implements Corrective RAG (CRAG) retrieval logic.

    Flow:
    1. Retrieve initial chunks (child + abstract)
    2. Grade individual chunks for relevance
    3. Evaluate overall retrieval quality
    4. Apply corrections based on evaluation:
       - correct: Select top relevant chunks
       - ambiguous: Rewrite query, retrieve again, merge and re-grade
       - incorrect: Strong rewrite, re-retrieve, if still weak return minimal
    """

    @staticmethod
    def _to_dict(chunk: Any) -> Dict[str, Any]:
        """Convert chunk to dict (handles both dict and dataclass)."""
        if isinstance(chunk, dict):
            return chunk
        # Assume dataclass with attributes (GradedChunk)
        return {
            "chunk_id": getattr(chunk, "chunk_id", None),
            "paper_id": getattr(chunk, "paper_id", ""),
            "title": getattr(chunk, "title", ""),
            "section": getattr(chunk, "section", ""),
            "chunk_type": getattr(chunk, "chunk_type", ""),
            "text": getattr(chunk, "text", ""),
            "parent_id": getattr(chunk, "parent_id", None),
            "chunk_index": getattr(chunk, "chunk_index", 0),
            "token_count": getattr(chunk, "token_count", 0),
            "relevance_score": getattr(chunk, "relevance_score", 0),
            "relevance_label": getattr(chunk, "relevance_label", ""),
            "reasoning": getattr(chunk, "reasoning", ""),
        }

    def __init__(
        self,
        retriever: Optional[HybridRetriever] = None,
        grader: Optional[DocumentGrader] = None,
        rewriter: Optional[QueryRewriter] = None,
        evaluator: Optional[RetrievalEvaluator] = None,
        relevance_threshold: float = 0.5,
        max_chunks_final: int = 8,
        max_rewrite_attempts: int = 2,
    ):
        """
        Initialize corrective retriever.

        Args:
            retriever: HybridRetriever instance (creates default if None)
            grader: DocumentGrader instance (creates default if None)
            rewriter: QueryRewriter instance (creates default if None)
            evaluator: RetrievalEvaluator instance (creates default if None)
            relevance_threshold: Minimum relevance score to keep a chunk
            max_chunks_final: Maximum chunks to return
            max_rewrite_attempts: Max query rewrite attempts for correction
        """
        self.retriever = retriever or get_retriever()
        self.grader = grader or get_grader()
        self.rewriter = rewriter or get_rewriter()
        self.evaluator = evaluator or get_evaluator()
        self.relevance_threshold = relevance_threshold
        self.max_chunks_final = max_chunks_final
        self.max_rewrite_attempts = max_rewrite_attempts

    def retrieve(
        self,
        query: str,
        paper_id: Optional[str] = None,
        chunk_types: Optional[List[str]] = None,
    ) -> CorrectiveRetrievalResult:
        """
        Perform corrective retrieval for a query.

        Args:
            query: User query
            paper_id: Optional paper filter
            chunk_types: Chunk types to search (default: child + abstract)

        Returns:
            CorrectiveRetrievalResult with final chunks and metadata
        """
        if chunk_types is None:
            chunk_types = ["child", "abstract"]

        logger.info(f"Starting corrective retrieval for: {query[:60]}...")

        # Step 1: Initial retrieval
        retrieval_result = self.retriever.retrieve_multi_type(
            query=query,
            chunk_types=chunk_types,
            paper_id=paper_id,
        )

        if not retrieval_result.chunks:
            logger.warning("No chunks retrieved initially")
            return CorrectiveRetrievalResult(
                chunks=[],
                evaluation=None,
                correction_applied="none",
                total_retrievals=1,
            )

        # Step 2: Grade individual chunks
        graded_chunks = self.grader.grade(query, retrieval_result.chunks)

        # Convert GradedChunk objects to dicts for uniform handling
        graded_chunks = [self._to_dict(c) for c in graded_chunks]

        # Step 3: Evaluate overall retrieval quality
        evaluation = self.evaluator.evaluate(query, graded_chunks)

        # Step 4: Apply corrective action
        rewrite_used = None
        if evaluation.evaluation == "correct":
            final_chunks = self._refine_chunks(graded_chunks)
            correction = "refine"

        elif evaluation.evaluation == "ambiguous":
            final_chunks, rewrite_used = self._handle_ambiguous(query, graded_chunks, chunk_types, paper_id)
            correction = "rewrite_merge"

        else:  # "incorrect"
            final_chunks, rewrite_used = self._handle_incorrect(query, graded_chunks, chunk_types, paper_id)
            correction = "strong_rewrite"

        # Limit final chunks
        final_chunks = self._limit_final_chunks(final_chunks)

        logger.info(f"Corrective retrieval complete: {correction}, {len(final_chunks)} final chunks, eval={evaluation.evaluation}")

        return CorrectiveRetrievalResult(
            chunks=final_chunks,
            evaluation=evaluation,
            correction_applied=correction,
            rewrite_used=rewrite_used,
            total_retrievals=2 if correction != "refine" else 1,
        )

    def _refine_chunks(self, graded_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Refine: select top relevant chunks, already graded."""
        # Already filtered by grader, just sort by relevance
        sorted_chunks = sorted(
            graded_chunks,
            key=lambda x: x.get("relevance_score", 0),
            reverse=True
        )
        return sorted_chunks[:self.max_chunks_final]

    def _handle_ambiguous(
        self,
        query: str,
        initial_chunks: List[Dict[str, Any]],
        chunk_types: List[str],
        paper_id: Optional[str],
    ) -> tuple[List[Dict[str, Any]], Optional[str]]:
        """
        Handle ambiguous retrieval:
        1. Rewrite query for better specificity
        2. Retrieve with rewritten query
        3. Merge and re-grade
        """
        logger.info("Retrieval ambiguous - applying query rewrite correction")

        # Rewrite query
        rewritten = self.rewriter.rewrite(query)
        search_query = rewritten.rewritten
        logger.info(f"Ambiguous correction: rewritten query: '{search_query}'")

        # Second retrieval with rewritten query
        second_result = self.retriever.retrieve_multi_type(
            query=search_query,
            chunk_types=chunk_types,
            paper_id=paper_id,
        )

        if not second_result.chunks:
            logger.warning("Second retrieval returned no chunks, using initial")
            return self._refine_chunks(initial_chunks), search_query

        # Grade second retrieval
        second_graded = self.grader.grade(query, second_result.chunks)  # Grade against original query
        second_graded = [self._to_dict(c) for c in second_graded]

        # Merge initial and second results (deduplicate by chunk_id)
        merged = self._merge_chunks(initial_chunks, second_graded)

        # Re-grade merged set against original query
        regraded = self.grader.grade(query, merged)
        regraded = [self._to_dict(c) for c in regraded]

        return self._refine_chunks(regraded), search_query

    def _handle_incorrect(
        self,
        query: str,
        initial_chunks: List[Dict[str, Any]],
        chunk_types: List[str],
        paper_id: Optional[str],
    ) -> tuple[List[Dict[str, Any]], Optional[str]]:
        """
        Handle incorrect retrieval:
        1. Generate a stronger, more specific rewritten query
        2. Re-retrieve
        3. If still weak, return empty or minimal context
        """
        logger.info("Retrieval incorrect - applying strong rewrite correction")

        # Use alternative queries from rewriter or generate a more specific one
        rewritten = self.rewriter.rewrite(query)

        # Try alternatives first, then the main rewritten query
        candidates = [rewritten.rewritten] + rewritten.alternatives

        best_chunks = []
        best_rewrite = None

        for attempt, search_query in enumerate(candidates[:self.max_rewrite_attempts]):
            logger.info(f"Incorrect correction attempt {attempt + 1}: '{search_query}'")

            result = self.retriever.retrieve_multi_type(
                query=search_query,
                chunk_types=chunk_types,
                paper_id=paper_id,
            )

            if not result.chunks:
                continue

            graded = self.grader.grade(query, result.chunks)
            graded = [self._to_dict(c) for c in graded]

            if graded:
                best_chunks = graded
                best_rewrite = search_query
                break  # Found relevant chunks

        if not best_chunks:
            logger.warning("All rewrite attempts failed - returning empty context")
            return [], best_rewrite

        # Final refinement
        final = self._refine_chunks(best_chunks)
        return final, best_rewrite

    def _merge_chunks(
        self,
        chunks1: List[Dict[str, Any]],
        chunks2: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Merge two chunk lists, deduplicating by chunk_id."""
        seen = set()
        merged = []

        for chunk in chunks1 + chunks2:
            cid = chunk.get("chunk_id")
            if cid and cid not in seen:
                seen.add(cid)
                merged.append(chunk)

        return merged

    def _limit_final_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Limit to max final chunks, sorted by relevance."""
        sorted_chunks = sorted(
            chunks,
            key=lambda x: x.get("relevance_score", x.get("score", 0)),
            reverse=True
        )
        return sorted_chunks[:self.max_chunks_final]


def get_corrective_retriever(
    retriever: Optional[HybridRetriever] = None,
    grader: Optional[DocumentGrader] = None,
    rewriter: Optional[QueryRewriter] = None,
    evaluator: Optional[RetrievalEvaluator] = None,
    relevance_threshold: float = 0.5,
    max_chunks_final: int = 8,
    max_rewrite_attempts: int = 2,
) -> CorrectiveRetriever:
    """Get a corrective retriever instance."""
    return CorrectiveRetriever(
        retriever, grader, rewriter, evaluator,
        relevance_threshold, max_chunks_final, max_rewrite_attempts
    )