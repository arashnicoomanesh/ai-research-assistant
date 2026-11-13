"""
Retrieval Evaluator for Corrective RAG (CRAG).

Evaluates overall retrieval quality given a query and retrieved chunks.
Returns one of: "correct", "ambiguous", "incorrect" with reasoning.
"""
import json
import logging
from typing import List, Dict, Any, Optional, Literal
from dataclasses import dataclass
from src.core.llm import OllamaLLM, get_llm
from src.core.config import config

logger = logging.getLogger(__name__)


EvaluationResult = Literal["correct", "ambiguous", "incorrect"]


@dataclass
class RetrievalEvaluation:
    """Result of retrieval quality evaluation."""
    evaluation: EvaluationResult
    confidence: float
    reasoning: str
    num_chunks: int
    avg_relevance: float


class RetrievalEvaluator:
    """
    Evaluates whether retrieved chunks are sufficient to answer the query.

    Uses LLM to assess overall retrieval quality, not individual chunk relevance.
    This is the "corrective" step in CRAG that decides if retrieval needs fixing.
    
    In CPU_SAFE_MODE (config.cpu_safe.enabled), uses heuristic fallback based on
    average relevance scores instead of LLM evaluation.
    """

    EVALUATOR_SYSTEM_PROMPT = """You are a retrieval quality evaluator for a research paper QA system.
Your task is to evaluate whether the retrieved document chunks are sufficient to answer the user's query.

Given:
- A user query
- A list of retrieved chunks with their text and metadata

Classify the retrieval into exactly ONE of these categories:

1. "correct" - The retrieved chunks contain sufficient relevant information to answer the query well.
   The chunks directly address the query topic with relevant facts, methods, results, or comparisons.

2. "ambiguous" - The retrieved chunks are partially relevant but insufficient.
   - Some chunks are relevant but key information is missing
   - Chunks are on the right topic but too generic
   - Mixed relevant/irrelevant chunks
   - Need more specific or comprehensive retrieval

3. "incorrect" - The retrieved chunks are largely irrelevant or fail to address the query.
   - Chunks discuss different topics/papers
   - No relevant information for the query
   - Completely off-target retrieval

Output MUST be valid JSON with this exact structure:
{
  "evaluation": "correct|ambiguous|incorrect",
  "confidence": 0.0-1.0,
  "reasoning": "Brief explanation of the evaluation"
}"""

    USER_PROMPT_TEMPLATE = """Query: {query}

Retrieved Chunks ({num_chunks} total):
{chunks_text}

Evaluate the overall retrieval quality. Return ONLY the JSON structure specified."""

    def __init__(
        self,
        llm: Optional[OllamaLLM] = None,
        max_chunks_to_show: int = 10,
        max_chunk_chars: int = 800,
    ):
        """
        Initialize retrieval evaluator.

        Args:
            llm: LLM client (creates default if None)
            max_chunks_to_show: Maximum chunks to include in evaluation prompt
            max_chunk_chars: Maximum characters per chunk in prompt
        """
        self.llm = llm or get_llm()
        self.max_chunks_to_show = max_chunks_to_show
        self.max_chunk_chars = max_chunk_chars
        self.cpu_safe = config.cpu_safe.enabled
        self.enable_llm_evaluator = config.cpu_safe.enable_retrieval_evaluator

    @staticmethod
    def _to_dict(chunk: Any) -> Dict[str, Any]:
        """Convert chunk to dict (handles both dict and dataclass)."""
        if isinstance(chunk, dict):
            return chunk
        # Assume dataclass with attributes
        return {
            "text": getattr(chunk, "text", ""),
            "title": getattr(chunk, "title", "Unknown"),
            "section": getattr(chunk, "section", "Unknown"),
            "relevance_score": getattr(chunk, "relevance_score", getattr(chunk, "score", 0)),
            "chunk_id": getattr(chunk, "chunk_id", None),
            "paper_id": getattr(chunk, "paper_id", None),
            "chunk_type": getattr(chunk, "chunk_type", None),
        }

    def evaluate(
        self,
        query: str,
        chunks: List[Any],
    ) -> RetrievalEvaluation:
        """
        Evaluate retrieval quality for a query.

        Args:
            query: User query
            chunks: List of retrieved chunks (dicts or GradedChunk objects)

        Returns:
            RetrievalEvaluation with decision and metadata
        """
        if not chunks:
            logger.info("No chunks to evaluate - returning 'incorrect'")
            return RetrievalEvaluation(
                evaluation="incorrect",
                confidence=1.0,
                reasoning="No chunks retrieved",
                num_chunks=0,
                avg_relevance=0.0,
            )

        # Convert all chunks to dict for uniform handling
        chunk_dicts = [self._to_dict(c) for c in chunks]

        # CPU-safe mode: use heuristic fallback
        if self.cpu_safe and not self.enable_llm_evaluator:
            logger.info(f"CPU_SAFE_MODE: Skipping LLM evaluator, using heuristic for {len(chunks)} chunks")
            return self._fallback_evaluation(chunk_dicts, 
                sum(c.get("relevance_score", c.get("score", 0)) for c in chunk_dicts) / len(chunk_dicts) if chunk_dicts else 0.0)

        # Limit chunks for prompt
        eval_chunks = chunk_dicts[:self.max_chunks_to_show]

        # Build chunks text for prompt
        chunks_text = ""
        for i, chunk in enumerate(eval_chunks):
            text = chunk.get("text", "")[:self.max_chunk_chars]
            title = chunk.get("title", "Unknown")
            section = chunk.get("section", "Unknown")
            score = chunk.get("relevance_score", chunk.get("score", 0))
            chunks_text += f"\n--- Chunk {i+1} (score: {score:.3f}) ---\n"
            chunks_text += f"Paper: {title} [{section}]\n"
            chunks_text += f"Text: {text}\n"

        # Calculate average relevance
        scores = [
            c.get("relevance_score", c.get("score", 0))
            for c in chunk_dicts
        ]
        avg_relevance = sum(scores) / len(scores) if scores else 0.0

        prompt = self.USER_PROMPT_TEMPLATE.format(
            query=query,
            num_chunks=len(eval_chunks),
            chunks_text=chunks_text,
        )

        try:
            response = self.llm.generate(
                prompt=prompt,
                system_prompt=self.EVALUATOR_SYSTEM_PROMPT,
                temperature=0.0,
                max_tokens=512,
            )

            result = self._parse_response(response, chunk_dicts, avg_relevance)
            logger.info(f"Retrieval evaluation: {result.evaluation} (confidence: {result.confidence:.2f})")
            return result

        except Exception as e:
            logger.warning(f"Retrieval evaluation failed: {e}")
            # Fallback: use average relevance as heuristic
            return self._fallback_evaluation(chunk_dicts, avg_relevance)

    def _parse_response(
        self,
        response: str,
        chunk_dicts: List[Dict[str, Any]],
        avg_relevance: float,
    ) -> RetrievalEvaluation:
        """Parse LLM JSON response."""
        try:
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                data = json.loads(json_str)
            else:
                raise ValueError("No JSON found in response")

            evaluation = data.get("evaluation", "ambiguous")
            if evaluation not in ("correct", "ambiguous", "incorrect"):
                evaluation = "ambiguous"

            return RetrievalEvaluation(
                evaluation=evaluation,
                confidence=float(data.get("confidence", 0.5)),
                reasoning=data.get("reasoning", "No reasoning provided"),
                num_chunks=len(chunk_dicts),
                avg_relevance=avg_relevance,
            )

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"Failed to parse evaluator response: {e}")
            return self._fallback_evaluation(chunk_dicts, avg_relevance)

    def _fallback_evaluation(
        self,
        chunks: List[Dict[str, Any]],
        avg_relevance: float,
    ) -> RetrievalEvaluation:
        """Heuristic fallback evaluation based on average relevance."""
        if avg_relevance >= 0.7:
            return RetrievalEvaluation(
                evaluation="correct",
                confidence=0.6,
                reasoning=f"Heuristic: avg relevance {avg_relevance:.2f} >= 0.7",
                num_chunks=len(chunks),
                avg_relevance=avg_relevance,
            )
        elif avg_relevance >= 0.4:
            return RetrievalEvaluation(
                evaluation="ambiguous",
                confidence=0.5,
                reasoning=f"Heuristic: avg relevance {avg_relevance:.2f} in [0.4, 0.7)",
                num_chunks=len(chunks),
                avg_relevance=avg_relevance,
            )
        else:
            return RetrievalEvaluation(
                evaluation="incorrect",
                confidence=0.6,
                reasoning=f"Heuristic: avg relevance {avg_relevance:.2f} < 0.4",
                num_chunks=len(chunks),
                avg_relevance=avg_relevance,
            )


def get_evaluator(
    llm: Optional[OllamaLLM] = None,
    max_chunks_to_show: int = 10,
    max_chunk_chars: int = 800,
) -> RetrievalEvaluator:
    """Get an evaluator instance."""
    return RetrievalEvaluator(llm, max_chunks_to_show, max_chunk_chars)