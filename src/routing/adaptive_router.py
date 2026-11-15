"""
Adaptive Router for AI Research Papers.

Classifies queries and routes them to appropriate retrieval strategies.
Strategies:
- "no_retrieval": Simple questions answerable from model knowledge
- "single_shot": Standard retrieval + generation
- "multi_step": Complex queries requiring multiple retrieval steps
- "crag_raptor": RAPTOR-enhanced hierarchical retrieval for surveys
- "graph_rag": Graph-based retrieval for relational/entity queries
"""
import json
import logging
import re
from typing import Optional, Literal
from dataclasses import dataclass
from src.core.llm import OllamaLLM, get_llm

logger = logging.getLogger(__name__)


Strategy = Literal["no_retrieval", "single_shot", "multi_step", "crag_raptor", "graph_rag"]


@dataclass
class RoutingDecision:
    """Result of query routing."""
    strategy: Strategy
    confidence: float
    reasoning: str
    query_type: str
    use_raptor: bool = False  # Whether RAPTOR-enhanced retrieval should be used
    use_graph: bool = False  # Whether GraphRAG should be used


class AdaptiveRouter:
    """
    Routes queries to appropriate retrieval strategies.

    Uses LLM classification with fallback heuristics.
    Supports five strategies for different query complexities.
    """

    # System prompt for classification
    ROUTER_SYSTEM_PROMPT = """You are a query router for a research paper QA system.
Classify the user's query into one of these strategies:

1. "no_retrieval" - Query can be answered from general knowledge without paper retrieval.
   Examples: "What is a neural network?", "Define overfitting", "Explain backpropagation",
             "What does RAG stand for?"

2. "graph_rag" - Query asks about relationships between entities, methods, or concepts,
   or asks what techniques/methods a paper/introduction/proposal introduces.
   Uses knowledge graph for entity linking and neighborhood traversal.
   Examples: "How is Self-RAG related to CRAG?",
             "What techniques does Self-RAG introduce?",
             "What methods are based on BERT?",
             "What does Self-RAG use?",
             "Which models improve upon Transformer?",
             "What addresses poor retrieval quality?"

3. "single_shot" - Query requires retrieving specific information from papers, answerable in one retrieval step.
   Examples: "What method did the Attention Is All You Need paper use?",
             "What are the results of the BERT paper?",
             "Summarize the contributions of the GPT-3 paper."

4. "crag_raptor" - Query benefits from hierarchical summary retrieval using RAPTOR.
    Provides high-level overview of field/trends, then detailed evidence.
    Examples: "Overview of transformer architectures in recent papers",
              "Trends in attention mechanisms across 2020-2023",
              "Summarize the field of diffusion models",
              "Evolution of RAG techniques in recent years"

5. "multi_step" - Complex queries requiring multiple retrieval steps.
   ...
```

Wait, I need to be careful about the numbering. Let me just add a higher-priority note for graph_rag rather than renumbering everything.
{
  "strategy": "no_retrieval|single_shot|multi_step|crag_raptor|graph_rag",
  "confidence": 0.0-1.0,
  "reasoning": "Brief explanation",
  "query_type": "factual|comparison|specific_paper|methodology|survey|general|relational",
  "use_raptor": true|false,
  "use_graph": true|false
}"""

    USER_PROMPT_TEMPLATE = """Query: {query}

Classify this query and return the JSON structure."""

    # Heuristic patterns for fallback classification
    NO_RETRIEVAL_PATTERNS = [
        r"\b(what is|define|definition of|explain|describe)\b.*\b(neural network|transformer|attention|overfitting|backpropagation|gradient descent|loss function|activation function|regularization|dropout|batch norm|layer norm|embedding|tokenizer|fine-tuning|transfer learning|zero-shot|few-shot|in-context learning|RLHF|RAG|retrieval augmented|diffusion model|variational autoencoder|GAN|CNN|RNN|LSTM|BERT|GPT)\b",
        r"\bhow does\b.*\b(work|function)\b",
        r"\bwhat are\b.*\b(types of|kinds of|categories of)\b",
        r"\bwhat does\b.*\b(stand for|mean)\b",
    ]

    SINGLE_SHOT_PATTERNS = [
        r"\b(paper|study|research|article|publication)\b",
        r"\b(author|authors|proposed|introduced|presented)\b",
        r"\b(results?|findings?|conclusion|limitation|methodology|approach|method|architecture|model)\b",
        r"\b(summarize|summary|overview)\b.*\b(paper|study|work)\b",
        r"\b(arxiv|\d{4}\.\d{4,5})\b",
        r"\bwhat (did|does|was)\b.*\b(paper|study|model|method)\b",
    ]

    MULTI_STEP_PATTERNS = [
        r"\b(compare|comparison|versus|vs\.?|difference between|contrast)\b",
        r"\b(common|shared|across|multiple|various|different)\b.*\b(papers?|studies?|models?|approaches?)\b",
        r"\b(survey|review|evolution|trend|history|timeline|progression)\b",
        r"\b(how do|what are)\b.*\b(different|various|multiple)\b",
        r"\b(common limitations|shared challenges|across papers|in the literature)\b",
    ]

    RAPTOR_PATTERNS = [
        r"\b(overview|trends?|evolution|timeline)\b.*\b(field|area|domain|paradigm)\b",
        r"\b(summarize|survey)\b.*\b(field|area|domain)\b",
        r"\b(key|main|important|significant) .* (developments?|advancements?|trends?)\b",
        r"\b(recent .* year|past .* year)\b.*\b(advancements?|progress|trends)\b",
    ]

    GRAPH_RAG_PATTERNS = [
        # Relational queries: "how is X related to Y"
        r"\b(how is|how does)\b.*\b(related to|connected to|linked to|compared to)\b",
        # Technique/method queries: "what techniques does X introduce"
        r"\b(what|which)\b.*\b(techniques?|methods?|algorithms?)\b.*\b(introduces?|uses?|based on)\b",
        # "what does X use" queries
        r"\b(what does|which)\b.*\b(uses?|employes?)\b",
        # "based on / lineage" queries
        r"\b(lineage|ancestry|family tree|based on|influenced by|derived from)\b",
        # "relate / connect / link" queries (but NOT "what is X")
        r"\b(relat|connect|link|associat)\b.*\b(with|to)\b(?!\s*(is|a|an|the))",
    ]

    def __init__(
        self,
        llm: Optional[OllamaLLM] = None,
        use_llm: bool = True,
    ):
        """
        Initialize adaptive router.

        Args:
            llm: LLM client (creates default if None)
            use_llm: Whether to use LLM for classification (fallback to heuristics if False)
        """
        self.llm = llm or get_llm()
        self.use_llm = use_llm

    def route(self, query: str) -> RoutingDecision:
        """
        Route a query to the appropriate strategy.

        Args:
            query: User query

        Returns:
            RoutingDecision with strategy and metadata
        """
        # Always check heuristics first (they're fast and reliable for common patterns)
        heuristic_decision = self._route_heuristic(query)
        
        if self.use_llm:
            try:
                llm_decision = self._route_llm(query)
                # Use LLM decision only if it has higher confidence than heuristics
                if llm_decision.confidence > heuristic_decision.confidence:
                    return llm_decision
                else:
                    logger.debug(f"LLM routing confidence ({llm_decision.confidence}) lower than heuristics "
                                f"({heuristic_decision.confidence}), using heuristics")
                    return heuristic_decision
            except Exception as e:
                logger.warning(f"LLM routing failed, using heuristics: {e}")
                return heuristic_decision
        else:
            return heuristic_decision

    def _route_llm(self, query: str) -> RoutingDecision:
        """Route using LLM classification."""
        prompt = self.USER_PROMPT_TEMPLATE.format(query=query)

        response = self.llm.generate(
            prompt=prompt,
            system_prompt=self.ROUTER_SYSTEM_PROMPT,
            temperature=0.0,
            max_tokens=512,
        )

        # Parse JSON response
        json_start = response.find("{")
        json_end = response.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            json_str = response[json_start:json_end]
            data = json.loads(json_str)

            strategy = data.get("strategy", "single_shot")
            if strategy not in ("no_retrieval", "single_shot", "multi_step", "crag_raptor", "graph_rag"):
                strategy = "single_shot"

            return RoutingDecision(
                strategy=strategy,
                confidence=float(data.get("confidence", 0.8)),
                reasoning=data.get("reasoning", "LLM classification"),
                query_type=data.get("query_type", "unknown"),
                use_raptor=data.get("use_raptor", False),
                use_graph=data.get("use_graph", False),
            )

        # Fallback if parsing fails
        return self._route_heuristic(query)

    def _route_heuristic(self, query: str) -> RoutingDecision:
        """Route using heuristic pattern matching."""
        query_lower = query.lower()

        # Check for GRAPH_RAG indicators first (relational queries)
        for pattern in self.GRAPH_RAG_PATTERNS:
            if re.search(pattern, query_lower):
                return RoutingDecision(
                    strategy="graph_rag",
                    confidence=0.75,
                    reasoning=f"Heuristic matched graph_rag: {pattern}",
                    query_type="relational",
                    use_raptor=False,
                    use_graph=True,
                )

        # Check for RAPTOR/survey indicators
        for pattern in self.RAPTOR_PATTERNS:
            if re.search(pattern, query_lower):
                return RoutingDecision(
                    strategy="crag_raptor",
                    confidence=0.75,
                    reasoning=f"Heuristic matched RAPTOR: {pattern}",
                    query_type="survey",
                    use_raptor=True,
                    use_graph=False,
                )

        # Check for multi_step indicators (complex comparative/survey queries)
        for pattern in self.MULTI_STEP_PATTERNS:
            if re.search(pattern, query_lower):
                return RoutingDecision(
                    strategy="multi_step",
                    confidence=0.75,
                    reasoning=f"Heuristic matched multi_step: {pattern}",
                    query_type="comparison" if "compare" in pattern or "versus" in pattern else "survey",
                    use_raptor=False,
                    use_graph=False,
                )

        # Check for single_shot indicators (specific paper queries)
        for pattern in self.SINGLE_SHOT_PATTERNS:
            if re.search(pattern, query_lower):
                return RoutingDecision(
                    strategy="single_shot",
                    confidence=0.75,
                    reasoning=f"Heuristic matched single_shot: {pattern}",
                    query_type="specific_paper",
                    use_raptor=False,
                    use_graph=False,
                )

        # Check for no_retrieval indicators (general knowledge)
        for pattern in self.NO_RETRIEVAL_PATTERNS:
            if re.search(pattern, query_lower):
                return RoutingDecision(
                    strategy="no_retrieval",
                    confidence=0.7,
                    reasoning=f"Heuristic matched no_retrieval: {pattern}",
                    query_type="general",
                    use_raptor=False,
                    use_graph=False,
                )

        # Default to single_shot for ambiguous queries
        return RoutingDecision(
            strategy="single_shot",
            confidence=0.5,
            reasoning="Default strategy (no clear heuristic match)",
            query_type="unknown",
            use_raptor=False,
            use_graph=False,
        )

    def route_batch(self, queries: list[str]) -> list[RoutingDecision]:
        """Route multiple queries."""
        return [self.route(q) for q in queries]


def get_router(
    llm: Optional[OllamaLLM] = None,
    use_llm: bool = True,
) -> AdaptiveRouter:
    """Get a router instance."""
    return AdaptiveRouter(llm, use_llm)