"""
BM25 Index for AI Research Papers.

Builds and maintains a BM25 keyword search index over chunk texts.
Uses rank_bm25 for efficient CPU-based keyword search.
Persists index to disk for fast startup.
"""
import logging
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from rank_bm25 import BM25Okapi
from src.core.config import config

logger = logging.getLogger(__name__)


@dataclass
class BM25Index:
    """BM25 index container with metadata."""
    corpus: List[str] = field(default_factory=list)  # Tokenized documents
    doc_ids: List[str] = field(default_factory=list)  # chunk_id for each document
    metadata: List[Dict[str, Any]] = field(default_factory=list)  # Full metadata for each document
    bm25: Optional[BM25Okapi] = None
    total_docs: int = 0


class BM25Indexer:
    """
    Builds and manages BM25 index over paper chunks.

    Features:
    - Incremental index building from Qdrant chunks
    - Disk persistence (pickle) for fast startup
    - Automatic rebuild when corpus changes
    - Query-time BM25 scoring with RRF fusion support
    """

    def __init__(
        self,
        index_dir: Optional[Path] = None,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
    ):
        """
        Initialize BM25 indexer.

        Args:
            index_dir: Directory to persist index (default: data/processed/bm25 from config)
            bm25_k1: BM25 k1 parameter (term frequency saturation)
            bm25_b: BM25 b parameter (length normalization)
        """
        from pathlib import Path
        self.index_dir = Path(index_dir) if index_dir else config.paths.bm25_index_dir
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.index_dir / "bm25_index.pkl"
        self.meta_file = self.index_dir / "bm25_meta.json"

        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b

        self.index: Optional[BM25Index] = None
        self._load_or_build()

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for BM25."""
        # Lowercase, split on non-alphanumeric, filter short tokens
        import re
        tokens = re.findall(r'\b\w+\b', text.lower())
        return [t for t in tokens if len(t) > 1]

    def _load_or_build(self) -> None:
        """Load existing index or build new one."""
        if self.index_file.exists() and self.meta_file.exists():
            try:
                self._load()
                logger.info(f"Loaded BM25 index: {self.index.total_docs} documents")
                return
            except Exception as e:
                logger.warning(f"Failed to load BM25 index: {e}, will rebuild")
        else:
            logger.info("No existing BM25 index found, will build on first use")

    def _load(self) -> None:
        """Load index from disk."""
        with open(self.index_file, "rb") as f:
            self.index = pickle.load(f)

        # Verify index integrity
        if self.index is None or self.index.bm25 is None:
            raise ValueError("Invalid index file")

    def _save(self) -> None:
        """Save index to disk."""
        if self.index is None:
            return

        # Save BM25 index (pickle handles BM25Okapi object)
        with open(self.index_file, "wb") as f:
            pickle.dump(self.index, f)

        # Save metadata separately for inspection
        meta = {
            "total_docs": self.index.total_docs,
            "doc_ids": self.index.doc_ids,
            "metadata": self.index.metadata,
        }
        with open(self.meta_file, "w") as f:
            json.dump(meta, f, indent=2)

        logger.info(f"Saved BM25 index: {self.index.total_docs} documents")

    def build_from_qdrant(
        self,
        chunk_types: Optional[List[str]] = None,
        paper_id: Optional[str] = None,
    ) -> int:
        """
        Build BM25 index from Qdrant chunks.

        Args:
            chunk_types: Chunk types to index (default: child, abstract)
            paper_id: Optional paper filter

        Returns:
            Number of documents indexed
        """
        from src.ingestion.indexer import get_indexer
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        if chunk_types is None:
            chunk_types = ["child", "abstract"]

        indexer = get_indexer()

        # Retrieve all chunks for indexing
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

        # Scroll through all matching points
        chunks = []
        offset = 0
        limit = 1000
        while True:
            results = indexer.client.scroll(
                collection_name=config.qdrant.collection,
                scroll_filter=query_filter,
                limit=limit,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            points = results[0]
            if not points:
                break
            chunks.extend([p.payload for p in points])
            offset += limit
            if len(points) < limit:
                break

        logger.info(f"Retrieved {len(chunks)} chunks from Qdrant for BM25 indexing")

        if not chunks:
            logger.warning("No chunks found for BM25 indexing")
            return 0

        return self.build_from_chunks(chunks)

    def build_from_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """
        Build BM25 index from chunk payloads.

        Args:
            chunks: List of chunk payload dicts from Qdrant

        Returns:
            Number of documents indexed
        """
        corpus = []
        doc_ids = []
        metadata = []

        for chunk in chunks:
            text = chunk.get("text", "")
            chunk_id = chunk.get("chunk_id")
            if not text or not chunk_id:
                continue

            tokens = self._tokenize(text)
            if not tokens:
                continue

            corpus.append(tokens)
            doc_ids.append(chunk_id)
            metadata.append({
                "chunk_id": chunk_id,
                "paper_id": chunk.get("paper_id"),
                "title": chunk.get("title"),
                "section": chunk.get("section"),
                "chunk_type": chunk.get("chunk_type"),
                "parent_id": chunk.get("parent_id"),
                "chunk_index": chunk.get("chunk_index"),
                "token_count": chunk.get("token_count"),
            })

        if not corpus:
            logger.warning("No valid documents for BM25 indexing")
            return 0

        # Build BM25 index
        bm25 = BM25Okapi(corpus, k1=self.bm25_k1, b=self.bm25_b)

        self.index = BM25Index(
            corpus=corpus,
            doc_ids=doc_ids,
            metadata=metadata,
            bm25=bm25,
            total_docs=len(corpus),
        )

        self._save()
        logger.info(f"Built BM25 index with {len(corpus)} documents")
        return len(corpus)

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Search BM25 index.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of results with chunk_id, score, and metadata
        """
        if self.index is None or self.index.bm25 is None:
            logger.warning("BM25 index not built, returning empty results")
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        # Get BM25 scores
        scores = self.index.bm25.get_scores(query_tokens)

        # Get top-k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            meta = self.index.metadata[idx].copy()
            meta["score"] = float(scores[idx])
            meta["rank"] = len(results) + 1  # 1-based rank for RRF
            results.append(meta)

        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        if self.index is None:
            return {"status": "not_built", "total_docs": 0}
        return {
            "status": "ready",
            "total_docs": self.index.total_docs,
            "index_file": str(self.index_file),
            "meta_file": str(self.meta_file),
        }

    def clear(self) -> None:
        """Clear the index and delete persisted files."""
        self.index = None
        if self.index_file.exists():
            self.index_file.unlink()
        if self.meta_file.exists():
            self.meta_file.unlink()
        logger.info("Cleared BM25 index")


def get_bm25_indexer(
    index_dir: Optional[Path] = None,
    bm25_k1: float = 1.5,
    bm25_b: float = 0.75,
) -> BM25Indexer:
    """Get a BM25 indexer instance."""
    return BM25Indexer(index_dir, bm25_k1, bm25_b)