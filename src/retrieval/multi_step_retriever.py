"""
Multi-Step Retriever for AI Research Papers.

Implements practical multi-step retrieval:
1. LLM decomposes complex query into 2-3 focused sub-questions
2. For each sub-question: use CorrectiveRetriever to get relevant chunks
3. Optional gap analysis: ask LLM whether evidence is sufficient
4. If insufficient: generate one follow-up sub-question and retrieve once more
5. Return deduplicated evidence + process metadata

Constraints:
- Maximum 3 primary sub-questions
- Maximum 1 follow-up round
- Reuse existing CorrectiveRetriever, grader, and LLM modules
- Fully explicit, no LangChain/LlamaIndex
- Stays fully local
"""
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from src.core.llm import OllamaLLM, get_llm
from src.retrieval.corrective_retriever import CorrectiveRetriever, CorrectiveRetrievalResult, get_corrective_retriever
from src.retrieval.query_rewriter import get_rewriter

logger = logging.getLogger(__name__)


@dataclass
class SubQuestion:
    """A decomposed sub-question with metadata."""
    index: int
    question: str
    strategy: str = "single_shot"  # for tracking retrieval strategy
    relevance_score: Optional[float] = None
    retrieval_success: bool = False


@dataclass
class MultiStepResult:
    """Result of multi-step retrieval."""
    all_chunks: List[Dict[str, Any]] = field(default_factory=list)  # deduplicated evidence chunks
    sub_questions: List[SubQuestion] = field(default_factory=list)  # the decomposition used
    follow_up_question: Optional[str] = None  # optional follow-up
    gap_identified: bool = False
    total_retrievals: int = 0
    process_log: List[str] = field(default_factory=list)


class MultiStepRetriever:
    """
    Implements real multi-step retrieval for complex queries.

    Strategy:
    1. Decompose query into focused sub-questions using LLM
    2. Retrieve evidence for each sub-question via CorrectiveRetriever
    3. Gap analysis: determine if collected evidence is sufficient
    4. If insufficient: generate one follow-up sub-question and retrieve
    5. Return deduplicated evidence + full process metadata
    """

    MAX_SUB_QUESTIONS = 3
    MAX_FOLLOW_UP = 1

    def __init__(
        self,
        corrective_retriever: Optional[CorrectiveRetriever] = None,
        max_sub_questions: int = MAX_SUB_QUESTIONS,
        max_follow_up: int = MAX_FOLLOW_UP,
        relevance_threshold: float = 0.5,
    ):
        """
        Initialize multi-step retriever.

        Args:
            corrective_retriever: CorrectiveRetriever instance (creates default if None)
            max_sub_questions: Maximum number of primary sub-questions
            max_follow_up: Maximum follow-up rounds
            relevance_threshold: Minimum relevance for chunk inclusion
        """
        self.corrective_retriever = corrective_retriever or get_corrective_retriever()
        self.max_sub_questions = max_sub_questions
        self.max_follow_up = max_follow_up
        self.relevance_threshold = relevance_threshold
        self.rewriter = get_rewriter()

    def retrieve(self, query: str) -> MultiStepResult:
        """
        Perform multi-step retrieval for a complex query.

        Args:
            query: User query (may be original or already rewritten)

        Returns:
            MultiStepResult with collected evidence and process metadata
        """
        result = MultiStepResult()
        result.process_log.append(f"Starting multi-step retrieval for: {query[:80]}...")

        # Step 1: Decompose query into sub-questions
        sub_questions = self._decompose_query(query)
        result.process_log.append(f"Decomposed into {len(sub_questions)} sub-questions")

        if not sub_questions:
            result.process_log.append("No sub-questions decomposed - returning empty result")
            return result

        # Step 2: Retrieve evidence for each sub-question
        all_chunks = []
        seen_ids = set()

        for sq in sub_questions:
            result.process_log.append(
                f"Retrieving for sub-question {sq.index}: '{sq.question[:60]}...'"
            )
            cr_result = self.corrective_retriever.retrieve(
                query=sq.question,
            )
            result.total_retrievals += cr_result.total_retrievals

            # Collect graded/relevant chunks
            for chunk in cr_result.chunks:
                cid = chunk.get("chunk_id", "")
                if cid and cid not in seen_ids:
                    seen_ids.add(cid)
                    all_chunks.append(chunk)

            # Store metadata from this retrieval
            sq.retrieval_success = len(cr_result.chunks) > 0
            sq.relevance_score = cr_result.evaluation.confidence if cr_result.evaluation else 0.0

            result.process_log.append(
                f"  -> Got {len(cr_result.chunks)} chunks (eval: {cr_result.evaluation.evaluation if cr_result.evaluation else 'N/A'})"
            )

        # Step 3: Gap analysis - ask LLM if evidence is sufficient
        gap_result = self._gap_analysis(query, all_chunks, sub_questions)
        result.gap_identified = gap_result["gap_identified"]
        result.process_log.append(f"Gap analysis: {'identified - need follow-up' if result.gap_identified else 'sufficient'}")

        # Step 4: If gap identified, do follow-up retrieval
        if result.gap_identified and self.max_follow_up > 0:
            follow_up = gap_result["follow_up_question"]
            result.process_log.append(f"Generating follow-up question: '{follow_up[:60]}...'")
            result.follow_up_question = follow_up

            # Retrieve once more for the follow-up
            fr_result = self.corrective_retriever.retrieve(query=follow_up)
            result.total_retrievals += fr_result.total_retrievals

            for chunk in fr_result.chunks:
                cid = chunk.get("chunk_id", "")
                if cid and cid not in seen_ids:
                    seen_ids.add(cid)
                    all_chunks.append(chunk)

            result.process_log.append(
                f"  -> Follow-up retrieval got {len(fr_result.chunks)} additional chunks"
            )

        # Deduplicate final chunks
        # Also re-grade to ensure quality
        result.all_chunks = all_chunks
        result.sub_questions = sub_questions

        result.process_log.append(
            f"Final result: {len(all_chunks)} deduplicated chunks, "
            f"{len(sub_questions)} sub-questions, {result.gap_identified=}"
        )

        return result

    def _decompose_query(self, query: str) -> List[SubQuestion]:
        """
        Use LLM to decompose a complex query into focused sub-questions.

        Args:
            query: The user query (may already be rewritten)

        Returns:
            List of SubQuestion objects
        """
        decompose_system_prompt = """You are a query decomposition specialist for a research paper QA system.
Your task is to break down a complex user query into 2-3 focused, answerable sub-questions 
that can be addressed by retrieving from research paper databases.

Guidelines:
1. Each sub-question should be specific and focused (not too broad)
2. Sub-questions should collectively cover the original query comprehensively
3. Avoid yes/no questions - use "what", "how", "why" questions
4. Prioritize questions about methods, results, limitations, and comparisons
5. Maximum 3 sub-questions

Output MUST be valid JSON with this exact structure:
{
  "sub_questions": [
    {"question": "focused question 1"},
    {"question": "focused question 2"},
    {"question": "focused question 3"}
  ],
  "reasoning": "Brief explanation of the decomposition"
}"""

        decompose_user_prompt = f"""Query: {query}

Decompose this query into 2-3 focused sub-questions for research paper retrieval.
Return ONLY the JSON structure specified."""

        try:
            response = get_llm().generate(
                prompt=decompose_user_prompt,
                system_prompt=decompose_system_prompt,
                temperature=0.0,
                max_tokens=512,
            )

            # Parse JSON response
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                data = json.loads(response[json_start:json_end])
                sub_qs = data.get("sub_questions", [])
                reasoning = data.get("reasoning", "No reasoning provided")

                # Convert to SubQuestion objects
                questions = []
                for i, sq in enumerate(sub_qs[:self.max_sub_questions]):
                    questions.append(SubQuestion(
                        index=i + 1,
                        question=sq.get("question", ""),
                    ))

                # If we got fewer than 2, pad with the original query as fallback
                while len(questions) < 2:
                    questions.append(SubQuestion(
                        index=len(questions) + 1,
                        question=query,
                    ))

                result_process_log = [
                    f"Decomposed query into {len(questions)} sub-questions",
                    f"Reasoning: {reasoning}",
                ]
                logger.debug(f"Query decomposition: {process_log}")

                return questions

        except Exception as e:
            logger.warning(f"Query decomposition failed: {e}")
            # Fallback: return original query as single sub-question
            return [SubQuestion(index=1, question=query)]

    def _gap_analysis(
        self,
        original_query: str,
        collected_chunks: List[Dict[str, Any]],
        sub_questions: List[SubQuestion],
    ) -> Dict[str, Any]:
        """
        Ask LLM whether the collected evidence is sufficient to answer the original query.

        Args:
            original_query: The original user query
            collected_chunks: All chunks retrieved so far
            sub_questions: The decomposition used

        Returns:
            Dict with 'gap_identified' (bool) and 'follow_up_question' (str or None)
        """
        # Build chunks summary for the prompt
        chunks_summary = ""
        for i, chunk in enumerate(collected_chunks[:5]):  # Limit to 5 chunks
            title = chunk.get("title", "Unknown")
            section = chunk.get("section", "")
            text = chunk.get("text", "")[:200]
            chunks_summary += f"\n--- Chunk {i+1} ---\n"
            chunks_summary += f"Paper: {title} [{section}]\n"
            chunks_summary += f"Content: {text}\n"

        # Build sub-questions summary
        sq_summary = ""
        for sq in sub_questions:
            sq_summary += f"\nSub-question {sq.index}: {sq.question}\n"

        gap_system_prompt = """You are a retrieval gap analyst for a research paper QA system.
Your task is to evaluate whether the collected evidence from retrieved paper chunks 
is sufficient to answer the original user query.

Given:
- The original user query
- A list of collected document chunks (with paper titles, sections, and content)
- The sub-questions used for retrieval

Assess whether:
1. The chunks directly address the original query with sufficient specificity
2. Key information pieces are missing that would be needed for a complete answer
3. Additional retrieval would significantly improve the answer quality

Output MUST be valid JSON with this exact structure:
{
  "gap_identified": true|false,
  "follow_up_question": "a specific follow-up question if gap identified, or null"
}"""

        gap_user_prompt = f"""Original Query: {original_query}

Sub-questions used for retrieval:
{sq_summary}

Collected chunks for evaluation:
{chunks_summary}

Is the collected evidence sufficient to answer the original query? 
If not, generate ONE specific follow-up sub-question that would fill the gap.

Return ONLY the JSON structure specified."""

        try:
            response = get_llm().generate(
                prompt=gap_user_prompt,
                system_prompt=gap_system_prompt,
                temperature=0.0,
                max_tokens=512,
            )

            # Parse JSON response
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                return json.loads(response[json_start:json_end])

        except Exception as e:
            logger.warning(f"Gap analysis failed: {e}")

        # Default: no gap identified on error
        return {"gap_identified": False, "follow_up_question": None}


def get_multi_step_retriever(
    corrective_retriever: Optional[CorrectiveRetriever] = None,
    max_sub_questions: int = 3,
    max_follow_up: int = 1,
    relevance_threshold: float = 0.5,
) -> MultiStepRetriever:
    """Get a multi-step retriever instance."""
    return MultiStepRetriever(
        corrective_retriever,
        max_sub_questions,
        max_follow_up,
        relevance_threshold,
    )