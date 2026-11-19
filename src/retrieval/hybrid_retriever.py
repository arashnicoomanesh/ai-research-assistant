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
from src.retrieval.bm25_index import get_bm25_indexer, BM25Indexer

logger = logging.getLogger(__name__)


# RRF constant
RRF_K = 60


@dataclass
class RetrievalResult:
    """Container for retrieval results."""
    chunks: List[Dict[str, Any]]
    query: str
    strategy: str = "dense"
    total_found: int = 0


def _rrf_fuse(
    dense_results: List[Dict[str, Any]],
    bm25_results: List[Dict[str, Any]],
    top_k_final: int,
    rrf_k: int = RRF_K,
) -> List[Dict[str, Any]]:
    """
    Fuse dense and BM25 results using Reciprocal Rank Fusion (RRF).

    Args:
        dense_results: Dense results with 'score' and 'chunk_id'
        bm25_results: BM25 results with 'score' and 'chunk_id'
        top_k_final: Final number of results to return
        rrf_k: RRF constant (default 60)

    Returns:
        Fused and ranked results with 'rrf_score' field
    """
    # Compute RRF scores
    rrf_scores: Dict[str, float] = {}
    chunk_map: Dict[str, Dict[str, Any]] = {}

    # Process dense results
    for rank, chunk in enumerate(dense_results, 1):
        cid = chunk.get("chunk_id")
        if not cid:
            continue
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)
        chunk_map[cid] = chunk

    # Process BM25 results
    for rank, chunk in enumerate(bm25_results, 1):
        cid = chunk.get("chunk_id")
        if not cid:
            continue
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)
        if cid not in chunk_map:
            chunk_map[cid] = chunk

    # Sort by RRF score descending
    fused = sorted(
        [(cid, rrf_scores[cid], chunk_map[cid]) for cid in rrf_scores],
        key=lambda x: x[1],
        reverse=True,
    )

    # Build final results
    final = []
    for cid, rrf_score, chunk in fused[:top_k_final]:
        result = chunk.copy()
        result["rrf_score"] = rrf_score
        final.append(result)

    return final


class HybridRetriever:
    """
    Retrieves relevant chunks from Qdrant using dense vector search
    and optionally BM25 keyword search with RRF fusion.

    Features:
    - Searches child chunks (dense vectors) for granular retrieval
    - Supports metadata filtering (paper_id, section, chunk_type)
    - Returns child chunks with parent_id for later context expansion
    - BM25 + RRF fusion when HYBRID_ENABLED and index available
    - No LangChain dependencies
    """

    def __init__(
        self,
        client: Optional[QdrantClient] = None,
        embeddings: Optional[OllamaEmbeddings] = None,
        collection_name: Optional[str] = None,
        top_k: Optional[int] = None,
        bm25_indexer: Optional[BM25Indexer] = None,
        hybrid_enabled: Optional[bool] = None,
    ):
        """
        Initialize hybrid retriever.

        Args:
            client: Qdrant client (creates default if None)
            embeddings: Embeddings client (creates default if None)
            collection_name: Collection name (from config if None)
            top_k: Number of results to retrieve (from config if None)
            bm25_indexer: BM25 indexer instance (creates default if None)
            hybrid_enabled: Whether to enable BM25+RRF fusion (default from config)
        """
        self.client = client or QdrantClient(
            host=config.qdrant.host,
            port=config.qdrant.port,
        )
        self.embeddings = embeddings or get_embeddings()
        self.collection_name = collection_name or config.qdrant.collection
        self.top_k = top_k or config.retrieval.top_k_dense

        # BM25 indexer (lazy init)
        self._bm25_indexer = bm25_indexer
        self.hybrid_enabled = hybrid_enabled if hybrid_enabled is not None else config.retrieval.hybrid_enabled

    @property
    def bm25_indexer(self) -> Optional[BM25Indexer]:
        """Lazy-initialize BM25 indexer."""
        if self._bm25_indexer is None and self.hybrid_enabled:
            try:
                self._bm25_indexer = get_bm25_indexer()
                logger.info(f"BM25 indexer initialized: {self._bm25_indexer.get_stats()}")
            except Exception as e:
                logger.warning(f"Failed to initialize BM25 indexer: {e}")
                self._bm25_indexer = None
        return self._bm25_indexer

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
        top_k_bm25 = config.retrieval.top_k_bm25
        top_k_final = config.retrieval.top_k_final

        # Dense retrieval
        query_vector = self.embeddings.embed_query(query)

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

        dense_results = qdrant_search(
            client=self.client,
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter=query_filter,
            limit=k,
            with_payload=True,
            with_vectors=False,
        )

        dense_chunks = []
        for hit in dense_results:
            payload = hit.payload
            dense_chunks.append({
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

        # BM25 retrieval (if enabled and index available)
        bm25_chunks = []
        if self.hybrid_enabled and self.bm25_indexer:
            try:
                bm25_results = self.bm25_indexer.search(query, top_k=top_k_bm25)
                for r in bm25_results:
                    bm25_chunks.append({
                        "score": r.get("score", 0.0),
                        "chunk_id": r.get("chunk_id"),
                        "paper_id": r.get("paper_id"),
                        "title": r.get("title"),
                        "section": r.get("section"),
                        "chunk_type": r.get("chunk_type"),
                        "text": "",  # BM25 index doesn't store full text
                        "parent_id": r.get("parent_id"),
                        "chunk_index": r.get("chunk_index"),
                        "token_count": r.get("token_count"),
                    })
            except Exception as e:
                logger.warning(f"BM25 search failed: {e}")

        # Fuse results
        if bm25_chunks and dense_chunks:
            fused_chunks = _rrf_fuse(dense_chunks, bm25_chunks, top_k_final)
            strategy = "hybrid_rrf"
        elif dense_chunks:
            fused_chunks = dense_chunks[:top_k_final]
            strategy = "dense"
        elif bm25_chunks:
            fused_chunks = bm25_chunks[:top_k_final]
            strategy = "bm25"
        else:
            fused_chunks = []
            strategy = "empty"

        # If BM25 chunks were used, we may need to hydrate text from dense results
        # or fetch from Qdrant for BM25-only results
        for chunk in fused_chunks:
            if not chunk.get("text") and chunk.get("chunk_id"):
                # Try to find in dense chunks
                dense_match = next((c for c in dense_chunks if c.get("chunk_id") == chunk["chunk_id"]), None)
                if dense_match:
                    chunk["text"] = dense_match.get("text", "")

        logger.info(f"Retrieved {len(fused_chunks)} chunks ({strategy}) for query: {query[:50]}...")

        return RetrievalResult(
            chunks=fused_chunks,
            query=query,
            strategy=strategy,
            total_found=len(fused_chunks),
        )

    def retrieve_multi_type(
        self,
        query: str,
        top_k: Optional[int] = None,
        chunk_types: Optional[List[str]] = None,
        paper_id: Optional[str] = None,
    ) -> RetrievalResult:
        """
        Retrieve across multiple chunk types with BM25+RRF fusion.

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
        top_k_bm25 = config.retrieval.top_k_bm25
        top_k_final = config.retrieval.top_k_final

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

        dense_results = qdrant_search(
            client=self.client,
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter=query_filter,
            limit=k,
            with_payload=True,
            with_vectors=False,
        )

        dense_chunks = []
        for hit in dense_results:
            payload = hit.payload
            dense_chunks.append({
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

        # BM25 retrieval (if enabled and index available)
        bm25_chunks = []
        if self.hybrid_enabled and self.bm25_indexer:
            try:
                bm25_results = self.bm25_indexer.search(query, top_k=top_k_bm25)
                for r in bm25_results:
                    bm25_chunks.append({
                        "score": r.get("score", 0.0),
                        "chunk_id": r.get("chunk_id"),
                        "paper_id": r.get("paper_id"),
                        "title": r.get("title"),
                        "section": r.get("section"),
                        "chunk_type": r.get("chunk_type"),
                        "text": "",
                        "parent_id": r.get("parent_id"),
                        "chunk_index": r.get("chunk_index"),
                        "token_count": r.get("token_count"),
                    })
            except Exception as e:
                logger.warning(f"BM25 search failed: {e}")

        # Fuse results
        if bm25_chunks and dense_chunks:
            fused_chunks = _rrf_fuse(dense_chunks, bm25_chunks, top_k_final)
            strategy = "hybrid_rrf"
        elif dense_chunks:
            fused_chunks = dense_chunks[:top_k_final]
            strategy = "dense"
        elif bm25_chunks:
            fused_chunks = bm25_chunks[:top_k_final]
            strategy = "bm25"
        else:
            fused_chunks = []
            strategy = "empty"

        # Hydrate text for BM25-only results
        for chunk in fused_chunks:
            if not chunk.get("text") and chunk.get("chunk_id"):
                dense_match = next((c for c in dense_chunks if c.get("chunk_id") == chunk["chunk_id"]), None)
                if dense_match:
                    chunk["text"] = dense_match.get("text", "")

        logger.info(f"Retrieved {len(fused_chunks)} chunks ({strategy}) for query: {query[:50]}...")

        return RetrievalResult(
            chunks=fused_chunks,
            query=query,
            strategy=strategy,
            total_found=len(fused_chunks),
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
            must_conditions = [
                FieldCondition(key="chunk_id", match=MatchValue(value=parent_chunk_id)),
                FieldCondition(key="chunk_type", match=MatchValue(value="parent")),
            ]

            results = qdrant_search(
                client=self.client,
                collection_name=self.collection_name,
                query_vector=[0.0] * 768,
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

        parents = []
        for pid in parent_ids:
            parent = self.get_parent_chunk(pid)
            if parent:
                child_scores = parent_to_child_scores.get(pid, [0.0])
                parent["score"] = max(child_scores)
                parents.append(parent)

        seen = set()
        combined = []

        for chunk in chunks + parents:
            cid = chunk.get("chunk_id")
            if cid and cid not in seen:
                seen.add(cid)
                combined.append(chunk)

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