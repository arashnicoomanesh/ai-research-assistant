"""
RAPTOR Retriever for AI Research Papers.

Retrieves from both RAPTOR summary nodes and normal child chunks.
For global/survey queries, biases toward summary nodes to provide
high-level overviews before diving into detailed evidence.

Constraints:
- Fully local, no LangChain / LlamaIndex
- Reuse existing embeddings + Qdrant client patterns
- Return mixed evidence (summaries + supporting detailed chunks)
- RAPTOR-inspired: limited hierarchy / summary layers, not full recursive clustering tree
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

# RAPTOR-inspired config constants (not full RAPTOR tree)
DEFAULT_K_SUMMARY = 5
DEFAULT_K_DETAIL_PER_SUMMARY = 3
MAX_DETAIL_CHUNKS_TOTAL = 10


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

        # If 1+ survey patterns match, treat as survey query
        return survey_count >= 1

    def retrieve(
        self,
        query: str,
        paper_id: Optional[str] = None,
        k_summary: int = DEFAULT_K_SUMMARY,
        k_detail: int = MAX_DETAIL_CHUNKS_TOTAL,
    ) -> RaptorRetrievalResult:
        """
        Perform RAPTOR-enhanced retrieval.

        Args:
            query: User query
            paper_id: Optional filter for a specific paper
            k_summary: Number of summary nodes to retrieve (by query similarity)
            k_detail: Maximum total detail chunks to retrieve

        Returns:
            RaptorRetrievalResult with mixed evidence
        """
        # Step 1: Determine query type
        is_survey = self._is_survey_query(query)
        query_type = "survey" if is_survey else "detail"

        logger.info(f"RAPTOR retrieval: query_type={query_type}, paper_id={paper_id}")

        # Step 2: Retrieve summary nodes ranked by query similarity
        summary_nodes = self._retrieve_summaries(query, paper_id, k_summary)

        # Step 3: Retrieve detail chunks
        detail_chunks = self._retrieve_details(query, paper_id, k_detail)

        # Step 4: Combine and deduplicate with detail budget per summary
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
        query: str,
        paper_id: Optional[str],
        k: int,
    ) -> List[RaptorNode]:
        """
        Retrieve RAPTOR summary nodes from Qdrant ranked by query similarity.

        Args:
            query: User query for similarity search
            paper_id: Optional filter
            k: Number of summaries to retrieve

        Returns:
            List of RaptorNode summary objects ranked by embedding similarity
        """
        # Embed the query
        query_vector = self.embeddings.embed_query(query)

        # Build filter for summary nodes
        must_conditions = [
            FieldCondition(key="chunk_type", match=MatchValue(value="summary"))
        ]

        if paper_id:
            must_conditions.append(FieldCondition(key="paper_id", match=MatchValue(value=paper_id)))

        query_filter = Filter(must=must_conditions)

        # Search summary nodes by vector similarity
        results = qdrant_search(
            client=self.retriever.client,
            collection_name=config.qdrant.collection,
            query_vector=query_vector,
            query_filter=query_filter,
            limit=k,
            with_payload=True,
            with_vectors=True,
        )

        # Convert to RaptorNode objects
        nodes = []
        for hit in results:
            payload = hit.payload
            if payload.get("chunk_type") == "summary":
                node = RaptorNode(
                    chunk_id=payload.get("chunk_id", ""),
                    paper_id=payload.get("paper_id", ""),
                    title=payload.get("title", ""),
                    section=payload.get("section", ""),
                    level=payload.get("level", 1),
                    chunk_type=payload.get("chunk_type", ""),
                    summary=payload.get("summary", ""),
                    source_chunk_ids=payload.get("source_chunk_ids", []),
                    metadata=payload.get("metadata", {}),
                    embedding=hit.vector if hit.vector else None,
                )
                nodes.append(node)

        logger.debug(f"Retrieved {len(nodes)} summary nodes by query similarity")
        return nodes

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

        For survey queries: prioritize summaries, then add details linked to each summary.
        For detail queries: prioritize details, optionally include summaries.

        Each selected summary gets at most DEFAULT_K_DETAIL_PER_SUMMARY detail chunks
        from its source_chunk_ids, respecting a total budget of MAX_DETAIL_CHUNKS_TOTAL.

        Args:
            summary_nodes: RAPTOR summary nodes (already ranked by query similarity)
            detail_chunks: Detailed child chunks (ranked by query similarity)
            is_survey: Whether this is a survey query

        Returns:
            Deduplicated combined chunks with metadata
        """
        combined: List[Dict[str, Any]] = []
        seen_ids: set = set()

        # Build lookup for detail chunks by chunk_id for fast access
        detail_by_id = {chunk.get("chunk_id", ""): chunk for chunk in detail_chunks}

        if is_survey:
            # Survey mode: start with summaries, then attach detail chunks per summary
            total_details_added = 0

            for node in summary_nodes:
                chunk_id = node.chunk_id
                if chunk_id not in seen_ids:
                    seen_ids.add(chunk_id)
                    combined.append({
                        **{k: v for k, v in node.__dict__.items() if k not in ("_state", "embedding")},
                        "text": node.summary,  # Map summary to text field for generator
                        "_raptor_node": True,
                        "summary_level": node.level,
                    })

                # Attach detail chunks linked to this summary (via source_chunk_ids)
                # Up to DEFAULT_K_DETAIL_PER_SUMMARY per summary, respecting total budget
                if total_details_added < MAX_DETAIL_CHUNKS_TOTAL:
                    source_ids = node.source_chunk_ids or []
                    added_for_this_summary = 0
                    for sid in source_ids:
                        if total_details_added >= MAX_DETAIL_CHUNKS_TOTAL:
                            break
                        if added_for_this_summary >= DEFAULT_K_DETAIL_PER_SUMMARY:
                            break
                        if sid in detail_by_id and sid not in seen_ids:
                            seen_ids.add(sid)
                            combined.append(detail_by_id[sid])
                            total_details_added += 1
                            added_for_this_summary += 1

            # Note: We do NOT add additional detail chunks beyond the per-summary budget
            # to strictly respect MAX_DETAIL_CHUNKS_TOTAL and keep context bounded.

        else:
            # Detail mode: start with details (up to MAX_DETAIL_CHUNKS_TOTAL), add summaries as overviews
            details_added = 0
            for chunk in detail_chunks:
                if details_added >= MAX_DETAIL_CHUNKS_TOTAL:
                    break
                chunk_id = chunk.get("chunk_id", "")
                if chunk_id not in seen_ids:
                    seen_ids.add(chunk_id)
                    combined.append(chunk)
                    details_added += 1

            # Add summary nodes as supplementary overviews
            for node in summary_nodes:
                chunk_id = node.chunk_id
                if chunk_id not in seen_ids:
                    seen_ids.add(chunk_id)
                    combined.append({
                        **{k: v for k, v in node.__dict__.items() if k not in ("_state", "embedding")},
                        "text": node.summary,  # Map summary to text field for generator
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