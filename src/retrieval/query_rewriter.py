"""
Query Rewriter for AI Research Assistant.

Takes the original user query and uses the LLM to produce a clearer,
more retrieval-friendly version. Optionally generates alternative
query formulations for better recall.
"""
import json
import logging
from typing import List, Optional
from dataclasses import dataclass
from src.core.llm import OllamaLLM, get_llm

logger = logging.getLogger(__name__)


@dataclass
class RewrittenQuery:
    """Container for rewritten query results."""
    original: str
    rewritten: str
    alternatives: List[str]
    reasoning: str


class QueryRewriter:
    """
    Rewrites user queries for better retrieval performance.

    Features:
    - Clarifies ambiguous terms
    - Expands abbreviations
    - Adds relevant keywords
    - Generates alternative formulations
    """

    REWRITER_SYSTEM_PROMPT = """You are a query rewriter for a research paper retrieval system.
Your task is to rewrite the user's query to be more effective for semantic search.

Guidelines:
1. Clarify ambiguous terms and pronouns
2. Expand common abbreviations (e.g., "RAG" -> "Retrieval-Augmented Generation")
3. Add relevant technical keywords that might appear in papers
4. Make the query more specific and searchable
5. Keep the original intent intact

Output MUST be valid JSON with this exact structure:
{
  "rewritten": "clearer, more specific query for retrieval",
  "alternatives": ["alternative formulation 1", "alternative formulation 2"],
  "reasoning": "Brief explanation of changes made"
}"""

    USER_PROMPT_TEMPLATE = """Original Query: {query}

Rewrite this query for better semantic search in a research paper database.
Return ONLY the JSON structure specified."""

    def __init__(
        self,
        llm: Optional[OllamaLLM] = None,
        generate_alternatives: bool = True,
        num_alternatives: int = 1,
    ):
        """
        Initialize query rewriter.

        Args:
            llm: LLM client (creates default if None)
            generate_alternatives: Whether to generate alternative queries
            num_alternatives: Number of alternative queries to generate
        """
        self.llm = llm or get_llm()
        self.generate_alternatives = generate_alternatives
        self.num_alternatives = num_alternatives

    def rewrite(self, query: str) -> RewrittenQuery:
        """
        Rewrite a query for better retrieval.

        Args:
            query: Original user query

        Returns:
            RewrittenQuery with rewritten query and alternatives
        """
        logger.debug(f"Rewriting query: {query}")

        prompt = self.USER_PROMPT_TEMPLATE.format(query=query)

        try:
            response = self.llm.generate(
                prompt=prompt,
                system_prompt=self.REWRITER_SYSTEM_PROMPT,
                temperature=0.1,
                max_tokens=512,
            )

            # Parse JSON response
            result = self._parse_response(response, query)

            logger.info(f"Rewritten: '{result.rewritten}' (alternatives: {len(result.alternatives)})")
            return result

        except Exception as e:
            logger.warning(f"Query rewriting failed, using original: {e}")
            return RewrittenQuery(
                original=query,
                rewritten=query,
                alternatives=[],
                reasoning=f"Rewrite failed: {e}",
            )

    def _parse_response(self, response: str, original: str) -> RewrittenQuery:
        """Parse LLM JSON response."""
        try:
            # Extract JSON from response
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                data = json.loads(json_str)
            else:
                raise ValueError("No JSON found in response")

            rewritten = data.get("rewritten", original).strip()
            alternatives = data.get("alternatives", [])
            if not self.generate_alternatives:
                alternatives = []
            alternatives = alternatives[:self.num_alternatives]
            reasoning = data.get("reasoning", "No reasoning provided")

            return RewrittenQuery(
                original=original,
                rewritten=rewritten,
                alternatives=alternatives,
                reasoning=reasoning,
            )

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"Failed to parse rewriter response: {e}")
            return RewrittenQuery(
                original=original,
                rewritten=original,
                alternatives=[],
                reasoning=f"Parse error: {e}",
            )

    def rewrite_batch(self, queries: List[str]) -> List[RewrittenQuery]:
        """Rewrite multiple queries."""
        return [self.rewrite(q) for q in queries]


def get_rewriter(
    llm: Optional[OllamaLLM] = None,
    generate_alternatives: bool = True,
    num_alternatives: int = 1,
) -> QueryRewriter:
    """Get a query rewriter instance."""
    return QueryRewriter(llm, generate_alternatives, num_alternatives)