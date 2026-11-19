"""
Hybrid Retriever for AI Research Papers.

Combines dense vector search (Qdrant) with optional BM25 keyword search.
Retrieves child chunks for granular retrieval, returns parent_id for context expansion.
Explicit implementation without LangChain/LlamaIndex.
"""
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from src.core.config import config
from src.core.embeddings import OllamaEmbeddings, get_embeddings
from src.retrieval.qdrant_search import qdrant_search

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Container for retrieval results."""
    chunks: List[Dict[str, Any]]
    query: str
    strategy: str = "dense"
    total_found: int = 0


class HybridRetriever:
    """
    Retrieves relevant chunks from Qdrant using dense vector search.

    Features:
    - Searches child chunks (dense vectors) for granular retrieval
    - Supports metadata filtering (paper_id, section, chunk_type)
    - Returns child chunks with parent_id for later context expansion
    - No LangChain dependencies
    """

    def __init__(
        self,
        client: Optional[QdrantClient] = None,
        embeddings: Optional[OllamaEmbeddings] = None,
        collection_name: Optional[str] = None,
        top_k: Optional[int] = None,
    ):
        """
        Initialize hybrid retriever.

        Args:
            client: Qdrant client (creates default if None)
            embeddings: Embeddings client (creates default if None)
            collection_name: Collection name (from config if None)
            top_k: Number of results to retrieve (from config if None)
        """
        self.client = client or QdrantClient(
            host=config.qdrant.host,
            port=config.qdrant.port,
        )
        self.embeddings = embeddings or get_embeddings()
        self.collection_name = collection_name or config.qdrant.collection
        self.top_k = top_k or config.retrieval.top_k_dense

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        paper_id: Optional[str] = None,
        section: Optional[str] = None,
        chunk_type: str = "child",
    ) -> RetrievalResult:
        """
        Retrieve relevant chunks for a query.

        Args:
            query: Search query text
            top_k: Number of results (overrides default)
            paper_id: Optional paper filter
            section: Optional section filter
            chunk_type: Chunk type to search ("child", "parent", "abstract")

        Returns:
            RetrievalResult with chunks and metadata
        """
        k = top_k or self.top_k

        # Generate query embedding
        query_vector = self.embeddings.embed_query(query)

        # Build filter conditions
        must_conditions = [
            FieldCondition(key="chunk_type", match=MatchValue(value=chunk_type))
        ]

        if paper_id:
            must_conditions.append(
                FieldCondition(key="paper_id", match=MatchValue(value=paper_id))
            )

        if section:
            must_conditions.append(
                FieldCondition(key="section", match=MatchValue(value=section))
            )

        query_filter = Filter(must=must_conditions)

        # Search Qdrant
        results = qdrant_search(
            client=self.client,
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter=query_filter,
            limit=k,
            with_payload=True,
            with_vectors=False,
        )

        # Format results
        chunks = []
        for hit in results:
            payload = hit.payload
            chunks.append({
                "score": hit.score,
                "chunk_id": payload.get("chunk_id"),
                "paper_id": payload.get("paper_id"),
                "title": payload.get("title"),
                "section": payload.get("section"),
                "chunk_type": payload.get("chunk_type"),
                "text": payload.get("text"),
                "parent_id": payload.get("parent_id"),
                "chunk_index": payload.get("chunk_index"),
                "token_count": payload.get("token_count"),
            })

        logger.info(f"Retrieved {len(chunks)} {chunk_type} chunks for query: {query[:50]}...")

        return RetrievalResult(
            chunks=chunks,
            query=query,
            strategy="dense",
            total_found=len(chunks),
        )

    def retrieve_multi_type(
        self,
        query: str,
        top_k: Optional[int] = None,
        chunk_types: Optional[List[str]] = None,
        paper_id: Optional[str] = None,
    ) -> RetrievalResult:
        """
        Retrieve across multiple chunk types.

        Args:
            query: Search query text
            top_k: Number of results per type
            chunk_types: List of chunk types to search
            paper_id: Optional paper filter

        Returns:
            RetrievalResult with combined chunks
        """
        if chunk_types is None:
            chunk_types = ["child", "abstract"]

        k = top_k or self.top_k
        query_vector = self.embeddings.embed_query(query)

        must_conditions = []
        if paper_id:
            must_conditions.append(
                FieldCondition(key="paper_id", match=MatchValue(value=paper_id))
            )

        if chunk_types:
            should_conditions = [
                FieldCondition(key="chunk_type", match=MatchValue(value=ct))
                for ct in chunk_types
            ]
            must_conditions.append(Filter(should=should_conditions))

        query_filter = Filter(must=must_conditions) if must_conditions else None

        results = qdrant_search(
            client=self.client,
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter=query_filter,
            limit=k,
            with_payload=True,
            with_vectors=False,
        )

        chunks = []
        for hit in results:
            payload = hit.payload
            chunks.append({
                "score": hit.score,
                "chunk_id": payload.get("chunk_id"),
                "paper_id": payload.get("paper_id"),
                "title": payload.get("title"),
                "section": payload.get("section"),
                "chunk_type": payload.get("chunk_type"),
                "text": payload.get("text"),
                "parent_id": payload.get("parent_id"),
                "chunk_index": payload.get("chunk_index"),
                "token_count": payload.get("token_count"),
            })

        logger.info(f"Retrieved {len(chunks)} chunks ({', '.join(chunk_types)}) for query: {query[:50]}...")

        return RetrievalResult(
            chunks=chunks,
            query=query,
            strategy="dense_multi",
            total_found=len(chunks),
        )

    def get_parent_chunk(self, parent_chunk_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a parent chunk by its chunk_id for context expansion.

        Args:
            parent_chunk_id: Parent chunk's chunk_id (human-readable, stored in payload)

        Returns:
            Parent chunk payload or None if not found
        """
        try:
            # Search by payload chunk_id filter instead of point ID
            # Parent chunks are stored with chunk_id in payload, point ID is UUID
            must_conditions = [
                FieldCondition(key="chunk_id", match=MatchValue(value=parent_chunk_id)),
                FieldCondition(key="chunk_type", match=MatchValue(value="parent")),
            ]

            results = qdrant_search(
                client=self.client,
                collection_name=self.collection_name,
                query_vector=[0.0] * 768,  # dummy vector - filter does the work
                query_filter=Filter(must=must_conditions),
                limit=1,
                with_payload=True,
                with_vectors=False,
            )

            if results:
                payload = results[0].payload
                return {
                    "chunk_id": payload.get("chunk_id"),
                    "paper_id": payload.get("paper_id"),
                    "title": payload.get("title"),
                    "section": payload.get("section"),
                    "chunk_type": payload.get("chunk_type"),
                    "text": payload.get("text"),
                    "parent_id": payload.get("parent_id"),
                    "chunk_index": payload.get("chunk_index"),
                    "token_count": payload.get("token_count"),
                }
        except Exception as e:
            logger.warning(f"Failed to retrieve parent chunk {parent_chunk_id}: {e}")

        return None

    def expand_to_parents(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Expand child chunks to include their parent chunks for better context.

        When a parent is added because of a child hit, the parent inherits
        the maximum score of all children that expanded to it, so parents
        are not eliminated solely due to missing/low retrieval score.

        Args:
            chunks: List of child chunk dicts with parent_id

        Returns:
            List of chunks with parent chunks included (deduplicated)
        """
        # Map parent_id -> list of child scores that reference it
        parent_to_child_scores: Dict[str, List[float]] = {}
        parent_ids = set()

        for chunk in chunks:
            pid = chunk.get("parent_id")
            score = chunk.get("score", 0.0)
            if pid:
                parent_ids.add(pid)
                if pid not in parent_to_child_scores:
                    parent_to_child_scores[pid] = []
                parent_to_child_scores[pid].append(score)

        # Fetch all unique parents
        parents = []
        for pid in parent_ids:
            parent = self.get_parent_chunk(pid)
            if parent:
                # Assign parent the max score of triggering children
                child_scores = parent_to_child_scores.get(pid, [0.0])
                parent["score"] = max(child_scores)
                parents.append(parent)

        # Combine children and parents, deduplicate by chunk_id
        seen = set()
        combined = []

        for chunk in chunks + parents:
            cid = chunk.get("chunk_id")
            if cid and cid not in seen:
                seen.add(cid)
                combined.append(chunk)

        # Sort by score (children have scores, parents now inherit max child score)
        combined.sort(key=lambda x: (x.get("score", 0), x.get("chunk_type") != "parent"), reverse=True)

        logger.info(f"Expanded {len(chunks)} children to {len(combined)} total chunks with {len(parents)} parents "
                    f"(parents inherit child scores)")

        return combined


def get_retriever(
    client: Optional[QdrantClient] = None,
    embeddings: Optional[OllamaEmbeddings] = None,
    collection_name: Optional[str] = None,
    top_k: Optional[int] = None,
) -> HybridRetriever:
    """Get a retriever instance."""
    return HybridRetriever(client, embeddings, collection_name, top_k)