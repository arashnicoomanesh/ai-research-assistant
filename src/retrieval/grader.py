"""
Document Grader for AI Research Papers.

Uses LLM (Ollama) to evaluate relevance of retrieved chunks to a query.
Returns only relevant chunks with relevance scores.
Uses structured JSON prompting for reliable output.
"""
import json
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from src.core.llm import OllamaLLM, get_llm
from src.core.config import config

logger = logging.getLogger(__name__)


@dataclass
class GradedChunk:
    """A chunk with relevance grading."""
    chunk_id: str
    paper_id: str
    title: str
    section: str
    chunk_type: str
    text: str
    parent_id: Optional[str]
    relevance_score: float  # 0.0 to 1.0
    relevance_label: str    # "relevant", "partially_relevant", "not_relevant"
    reasoning: str


class DocumentGrader:
    """
    Grades retrieved document chunks for relevance to a query.

    Uses LLM to evaluate each chunk and returns only relevant ones.
    Employs structured JSON prompting for reliable parsing.
    
    In CPU_SAFE_MODE (config.cpu_safe.enabled), skips LLM grading and
    ranks chunks by vector retrieval score only.
    """

    # System prompt for grading
    GRADER_SYSTEM_PROMPT = """You are a relevance grader for a research paper retrieval system.
Your task is to evaluate whether a retrieved text chunk is relevant to answering the user's query.

Guidelines:
- Score 1.0: Directly answers the query, contains key facts/methods/results
- Score 0.7-0.9: Highly relevant, contains supporting information
- Score 0.4-0.6: Partially relevant, tangentially related
- Score 0.0-0.3: Not relevant, unrelated or too generic

Output MUST be valid JSON with this exact structure:
{
  "chunks": [
    {
      "chunk_id": "string",
      "relevance_score": 0.0-1.0,
      "relevance_label": "relevant|partially_relevant|not_relevant",
      "reasoning": "Brief explanation"
    }
  ]
}"""

    USER_PROMPT_TEMPLATE = """Query: {query}

Retrieved Chunks:
{chunks_text}

Grade each chunk for relevance to the query. Return ONLY the JSON structure specified."""

    def __init__(
        self,
        llm: Optional[OllamaLLM] = None,
        relevance_threshold: float = 0.5,
        batch_size: int = 5,
    ):
        """
        Initialize document grader.

        Args:
            llm: LLM client (creates default if None)
            relevance_threshold: Minimum score to keep a chunk (0.0-1.0)
            batch_size: Number of chunks to grade per LLM call
        """
        self.llm = llm or get_llm()
        self.relevance_threshold = relevance_threshold
        self.batch_size = batch_size
        self.cpu_safe = config.cpu_safe.enabled
        self.enable_llm_grader = config.cpu_safe.enable_llm_grader

    def grade(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
    ) -> List[GradedChunk]:
        """
        Grade chunks for relevance to query.

        Args:
            query: User query
            chunks: List of chunk dicts from retriever

        Returns:
            List of GradedChunk objects (only those above threshold)
        """
        if not chunks:
            return []

        # CPU-safe mode: skip LLM grading, use retrieval scores
        if self.cpu_safe and not self.enable_llm_grader:
            logger.info(f"CPU_SAFE_MODE: Skipping LLM grader, ranking {len(chunks)} chunks by retrieval score")
            return self._fallback_grade_by_score(chunks)

        logger.info(f"Grading {len(chunks)} chunks for query: {query[:50]}...")

        all_graded = []

        # Process in batches to avoid context limits
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i:i + self.batch_size]
            graded_batch = self._grade_batch(query, batch)
            all_graded.extend(graded_batch)

        # Filter by threshold
        relevant_chunks = [
            gc for gc in all_graded
            if gc.relevance_score >= self.relevance_threshold
        ]

        logger.info(f"Graded: {len(all_graded)} total, {len(relevant_chunks)} above threshold ({self.relevance_threshold})")

        return relevant_chunks

    def _grade_batch(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
    ) -> List[GradedChunk]:
        """Grade a batch of chunks."""
        # Build chunks text for prompt
        chunks_text = ""
        for idx, chunk in enumerate(chunks):
            chunk_id = chunk.get("chunk_id", f"chunk_{idx}")
            text = chunk.get("text", "")[:1500]  # Limit text length
            title = chunk.get("title", "Unknown")
            section = chunk.get("section", "Unknown")
            chunks_text += f"\n--- Chunk {idx} (ID: {chunk_id}) ---\n"
            chunks_text += f"Paper: {title}\n"
            chunks_text += f"Section: {section}\n"
            chunks_text += f"Text: {text}\n"

        prompt = self.USER_PROMPT_TEMPLATE.format(
            query=query,
            chunks_text=chunks_text,
        )

        try:
            response = self.llm.generate(
                prompt=prompt,
                system_prompt=self.GRADER_SYSTEM_PROMPT,
                temperature=0.0,  # Deterministic for grading
                max_tokens=2048,
            )

            # Parse JSON response
            result = self._parse_response(response, chunks)

            return result

        except Exception as e:
            logger.error(f"Failed to grade batch: {e}")
            # Return all chunks as partially relevant on error
            return self._fallback_grade(chunks)

    def _parse_response(
        self,
        response: str,
        original_chunks: List[Dict[str, Any]],
    ) -> List[GradedChunk]:
        """Parse LLM JSON response into GradedChunk objects."""
        try:
            # Extract JSON from response (handle potential extra text)
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                data = json.loads(json_str)
            else:
                raise ValueError("No JSON found in response")

            graded = []
            for item in data.get("chunks", []):
                chunk_id = item.get("chunk_id", "")
                # Find original chunk
                orig_chunk = next(
                    (c for c in original_chunks if c.get("chunk_id") == chunk_id),
                    None
                )
                if orig_chunk:
                    graded.append(GradedChunk(
                        chunk_id=chunk_id,
                        paper_id=orig_chunk.get("paper_id", ""),
                        title=orig_chunk.get("title", ""),
                        section=orig_chunk.get("section", ""),
                        chunk_type=orig_chunk.get("chunk_type", ""),
                        text=orig_chunk.get("text", ""),
                        parent_id=orig_chunk.get("parent_id"),
                        relevance_score=float(item.get("relevance_score", 0.0)),
                        relevance_label=item.get("relevance_label", "not_relevant"),
                        reasoning=item.get("reasoning", ""),
                    ))

            return graded

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"Failed to parse grader response: {e}. Using fallback.")
            return self._fallback_grade(original_chunks)

    def _fallback_grade(self, chunks: List[Dict[str, Any]]) -> List[GradedChunk]:
        """Fallback grading when LLM fails - mark all as partially relevant."""
        return [
            GradedChunk(
                chunk_id=c.get("chunk_id", f"chunk_{i}"),
                paper_id=c.get("paper_id", ""),
                title=c.get("title", ""),
                section=c.get("section", ""),
                chunk_type=c.get("chunk_type", ""),
                text=c.get("text", ""),
                parent_id=c.get("parent_id"),
                relevance_score=0.5,
                relevance_label="partially_relevant",
                reasoning="Fallback grading (LLM error)",
            )
            for i, c in enumerate(chunks)
        ]

    def _fallback_grade_by_score(self, chunks: List[Dict[str, Any]]) -> List[GradedChunk]:
        """Fallback grading using retrieval scores - no LLM call."""
        # Sort by retrieval score (from vector search)
        sorted_chunks = sorted(
            chunks,
            key=lambda x: x.get("relevance_score", x.get("score", 0)),
            reverse=True
        )
        return [
            GradedChunk(
                chunk_id=c.get("chunk_id", f"chunk_{i}"),
                paper_id=c.get("paper_id", ""),
                title=c.get("title", ""),
                section=c.get("section", ""),
                chunk_type=c.get("chunk_type", ""),
                text=c.get("text", ""),
                parent_id=c.get("parent_id"),
                relevance_score=min(1.0, c.get("relevance_score", c.get("score", 0.5))),
                relevance_label="relevant" if c.get("relevance_score", c.get("score", 0)) >= 0.7 else "partially_relevant",
                reasoning="CPU-safe fallback: ranked by vector retrieval score",
            )
            for i, c in enumerate(sorted_chunks)
            if c.get("relevance_score", c.get("score", 0)) >= self.relevance_threshold
        ]

    def grade_with_scores(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        keep_all: bool = False,
    ) -> List[GradedChunk]:
        """
        Grade chunks and return all with scores (optionally filtered).

        Args:
            query: User query
            chunks: List of chunk dicts
            keep_all: If True, return all chunks with scores; if False, filter by threshold

        Returns:
            List of GradedChunk objects
        """
        graded = self.grade(query, chunks)
        if keep_all:
            return graded
        return [g for g in graded if g.relevance_score >= self.relevance_threshold]


def get_grader(
    llm: Optional[OllamaLLM] = None,
    relevance_threshold: float = 0.5,
    batch_size: int = 5,
) -> DocumentGrader:
    """Get a grader instance."""
    return DocumentGrader(llm, relevance_threshold, batch_size)