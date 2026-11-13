"""
RAPTOR Retriever for AI Research Papers.

Retrieves from both RAPTOR summary nodes and normal child chunks.
For global/survey queries, biases toward summary nodes to provide
high-level overviews before diving into detailed evidence.

Constraints:
- Fully local, no LangChain / LlamaIndex
- Reuse existing embeddings + Qdrant client patterns
- Return mixed evidence (summaries + supporting detailed chunks)
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from src.core.config import config
from src.core.embeddings import OllamaEmbeddings, get_embeddings
from src.retrieval.hybrid_retriever import get_retriever, RetrievalResult
from src.ingestion.raptor_builder import get_raptor_builder
from src.ingestion.raptor_types import RaptorNode
from qdrant_client.models import Filter, FieldCondition, MatchValue
from src.retrieval.qdrant_search import qdrant_search

logger = logging.getLogger(__name__)


@dataclass
class RaptorRetrievalResult:
    """Result from RAPTOR-enhanced retrieval."""
    summary_nodes: List[RaptorNode]  # High-level summary nodes
    detail_chunks: List[Dict[str, Any]]  # Detailed child chunks
    combined_chunks: List[Dict[str, Any]]  # Deduplicated combined evidence
    query_type: str  # "survey", "detail", or "mixed"
    retrieval_stats: Dict[str, Any]  # Statistics about the retrieval


class RaptorRetriever:
    """
    Retrieves from both RAPTOR summary nodes and normal child chunks.

    Strategy:
    1. Determine if query is a "survey/global" type or "detail" query
    2. If survey: retrieve summary nodes first, then detail chunks for supporting evidence
    3. If detail: retrieve child chunks primarily, optionally include summaries
    4. Return mixed evidence with deduplication

    The key insight: summary nodes provide high-level overviews across papers,
    while child chunks provide detailed evidence from specific sections.
    """

    def __init__(
        self,
        embeddings: Optional[OllamaEmbeddings] = None,
        retriever: Optional[Any] = None,
        raptor_builder: Optional[Any] = None,
    ):
        """
        Initialize RAPTOR retriever.

        Args:
            embeddings: Embeddings client (creates default if None)
            retriever: HybridRetriever instance (creates default if None)
            raptor_builder: RaptorBuilder instance (creates default if None)
        """
        self.embeddings = embeddings or get_embeddings()
        self.retriever = retriever or get_retriever()
        self.raptor_builder = raptor_builder or get_raptor_builder()

    def _is_survey_query(self, query: str) -> bool:
        """
        Determine if a query is a survey/global type question.

        Args:
            query: User query

        Returns:
            True if the query suggests a need for high-level overview
        """
        survey_patterns = [
            r"\b(overview|survey|review)\b",
            r"\b(trends|evolution)\b",
            r"\b(summarize|summarise)\b",
            r"\b(evolution of|development of)\b",
            r"\b(field of|area of)\b",
            r"\b(common|shared|across .* papers?)\b",
            r"\b(compare .* architectures?|models?|approaches?)\b",
        ]

        query_lower = query.lower()
        survey_count = sum(1 for pattern in survey_patterns if __import__('re').search(pattern, query_lower))

        # If 2+ survey patterns match, treat as survey query
        return survey_count >= 2

    def retrieve(
        self,
        query: str,
        paper_id: Optional[str] = None,
        k_summary: int = 5,
        k_detail: int = 10,
    ) -> RaptorRetrievalResult:
        """
        Perform RAPTOR-enhanced retrieval.

        Args:
            query: User query
            paper_id: Optional filter for a specific paper
            k_summary: Number of summary nodes to retrieve
            k_detail: Number of child chunks to retrieve

        Returns:
            RaptorRetrievalResult with mixed evidence
        """
        # Step 1: Determine query type
        is_survey = self._is_survey_query(query)
        query_type = "survey" if is_survey else "detail"

        logger.info(f"RAPTOR retrieval: query_type={query_type}, paper_id={paper_id}")

        # Step 2: Retrieve summary nodes
        summary_nodes = self._retrieve_summaries(paper_id, k_summary)

        # Step 3: Retrieve detail chunks
        detail_chunks = self._retrieve_details(query, paper_id, k_detail)

        # Step 4: Combine and deduplicate
        combined_chunks = self._combine_evidence(summary_nodes, detail_chunks, is_survey)

        # Step 5: Build stats
        stats = {
            "query_type": query_type,
            "num_summaries": len(summary_nodes),
            "num_details": len(detail_chunks),
            "num_combined": len(combined_chunks),
            "paper_id": paper_id,
            "is_survey": is_survey,
        }

        result = RaptorRetrievalResult(
            summary_nodes=summary_nodes,
            detail_chunks=detail_chunks,
            combined_chunks=combined_chunks,
            query_type=query_type,
            retrieval_stats=stats,
        )

        logger.info(f"RAPTOR retrieval complete: {len(summary_nodes)} summaries, "
                   f"{len(detail_chunks)} details, {len(combined_chunks)} combined (type={query_type})")

        return result

    def _retrieve_summaries(
        self,
        paper_id: Optional[str],
        k: int,
    ) -> List[RaptorNode]:
        """
        Retrieve RAPTOR summary nodes from Qdrant.

        Args:
            paper_id: Optional filter
            k: Number of summaries to retrieve

        Returns:
            List of RaptorNode summary objects
        """
        builder = self.raptor_builder

        # Get summary nodes
        nodes = builder.get_summary_nodes(
            paper_id=paper_id,
            level=None,  # Get all levels
        )

        # Filter to only summary nodes and sort - in a real implementation,
        # we'd use vector similarity. For now, return what's available.
        # Sort by level (prefer section-level 1, then paper-level 2)
        nodes.sort(key=lambda n: n.level)

        # Return top k
        return nodes[:k]

    def _retrieve_details(
        self,
        query: str,
        paper_id: Optional[str],
        k: int,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve detailed child chunks using the hybrid retriever.

        Args:
            query: User query
            paper_id: Optional filter
            k: Number of chunks to retrieve

        Returns:
            List of chunk payloads (dicts)
        """
        # Use the hybrid retriever to search child chunks
        # Filter by chunk_type="child"
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        must_conditions = [FieldCondition(key="chunk_type", match=MatchValue(value="child"))]

        if paper_id:
            must_conditions.append(FieldCondition(key="paper_id", match=MatchValue(value=paper_id)))

        query_vector = self.embeddings.embed_query(query)

        results = qdrant_search(
            client=self.retriever.client,
            collection_name=config.qdrant.collection,
            query_vector=query_vector,
            query_filter=Filter(must=must_conditions) if must_conditions else None,
            limit=k,
            with_payload=True,
            with_vectors=False,
        )

        chunks = [{"score": hit.score, **hit.payload} for hit in results]
        return chunks

    def _combine_evidence(
        self,
        summary_nodes: List[RaptorNode],
        detail_chunks: List[Dict[str, Any]],
        is_survey: bool,
    ) -> List[Dict[str, Any]]:
        """
        Combine summary nodes and detail chunks into deduplicated evidence.

        For survey queries: prioritize summaries, then add details.
        For detail queries: prioritize details, optionally include summaries.

        Args:
            summary_nodes: RAPTOR summary nodes
            detail_chunks: Detailed child chunks
            is_survey: Whether this is a survey query

        Returns:
            Deduplicated combined chunks with metadata
        """
        combined: List[Dict[str, Any]] = []
        seen_ids: set = set()

        if is_survey:
            # Survey mode: start with summaries, then add detail chunks
            # that reference or extend the summaries

            # Add summary nodes first
            for node in summary_nodes:
                chunk_id = node.chunk_id
                if chunk_id not in seen_ids:
                    seen_ids.add(chunk_id)
                    combined.append({
                        **{k: v for k, v in node.__dict__.items() if k != "_state"},
                        "_raptor_node": True,
                        "summary_level": node.level,
                    })

            # Add detail chunks that complement the summaries
            for chunk in detail_chunks:
                chunk_id = chunk.get("chunk_id", "")
                if chunk_id not in seen_ids:
                    # Check if this chunk belongs to a paper we have summaries for
                    paper_id = chunk.get("paper_id", "")
                    summary_papers = set(n.paper_id for n in summary_nodes)
                    if paper_id in summary_papers or not summary_papers:
                        seen_ids.add(chunk_id)
                        combined.append(chunk)

        else:
            # Detail mode: start with details, add summaries as overviews
            # Add detail chunks first
            for chunk in detail_chunks:
                chunk_id = chunk.get("chunk_id", "")
                if chunk_id not in seen_ids:
                    seen_ids.add(chunk_id)
                    combined.append(chunk)

            # Add summary nodes as supplementary overviews
            for node in summary_nodes:
                chunk_id = node.chunk_id
                if chunk_id not in seen_ids:
                    seen_ids.add(chunk_id)
                    combined.append({
                        **{k: v for k, v in node.__dict__.items() if k != "_state"},
                        "_raptor_node": True,
                        "summary_level": node.level,
                    })

        logger.debug(f"Combined {len(combined)} evidence chunks "
                    f"(summaries: {len(summary_nodes)}, details: {len(detail_chunks)})")

        return combined

    def retrieve_with_rationale(
        self,
        query: str,
        paper_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Retrieve with detailed rationale for each step.

        Args:
            query: User query
            paper_id: Optional paper filter

        Returns:
            Dict with retrieval details
        """
        result = self.retrieve(query, paper_id)

        return {
            "query": query,
            "query_type": result.query_type,
            "summary_count": len(result.summary_nodes),
            "detail_count": len(result.detail_chunks),
            "combined_count": len(result.combined_chunks),
            "summary_nodes": [
                {
                    "chunk_id": n.chunk_id,
                    "paper_id": n.paper_id,
                    "section": n.section,
                    "level": n.level,
                    "summary": n.summary[:100] + "..." if len(n.summary) > 100 else n.summary,
                }
                for n in result.summary_nodes
            ],
            "combined_examples": result.combined_chunks[:3] if result.combined_chunks else [],
        }


def get_raptor_retriever(
    embeddings: Optional[OllamaEmbeddings] = None,
    retriever: Optional[Any] = None,
    raptor_builder: Optional[Any] = None,
) -> RaptorRetriever:
    """Get a RAPTOR retriever instance."""
    return RaptorRetriever(embeddings, retriever, raptor_builder)