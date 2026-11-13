"""
Semantic Cache for AI Research Assistant.

Implements a simple semantic cache using embeddings to avoid
re-computing answers for similar queries.
"""
import json
import logging
import time
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, asdict
import numpy as np
from src.core.embeddings import OllamaEmbeddings, get_embeddings

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Single cache entry."""
    query: str
    query_embedding: List[float]
    answer: str
    sources: List[Dict[str, Any]]
    strategy: str
    timestamp: float
    hit_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CacheEntry":
        """Create from dictionary."""
        return cls(**data)


class SemanticCache:
    """
    Semantic cache for query-answer pairs.

    Uses embedding similarity to find cached answers for similar queries.
    Stores cache on disk as JSON for persistence.
    """

    def __init__(
        self,
        embeddings: Optional[OllamaEmbeddings] = None,
        similarity_threshold: float = 0.92,
        max_entries: int = 1000,
        cache_path: Optional[str] = None,
    ):
        """
        Initialize semantic cache.

        Args:
            embeddings: Embeddings client (creates default if None)
            similarity_threshold: Minimum cosine similarity for cache hit (0.0-1.0)
            max_entries: Maximum number of entries to keep
            cache_path: Path to cache file (None = in-memory only)
        """
        self.embeddings = embeddings or get_embeddings()
        self.similarity_threshold = similarity_threshold
        self.max_entries = max_entries
        self.cache_path = Path(cache_path) if cache_path else None
        self.entries: List[CacheEntry] = []

        # Load existing cache if path provided
        if self.cache_path:
            self._load()

    def _load(self) -> None:
        """Load cache from disk."""
        if not self.cache_path or not self.cache_path.exists():
            logger.info("No existing cache file, starting fresh")
            return

        try:
            with open(self.cache_path, "r") as f:
                data = json.load(f)

            self.entries = [CacheEntry.from_dict(item) for item in data]
            logger.info(f"Loaded {len(self.entries)} cache entries from {self.cache_path}")

        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            self.entries = []

    def _save(self) -> None:
        """Save cache to disk."""
        if not self.cache_path:
            return

        try:
            # Ensure parent directory exists
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)

            with open(self.cache_path, "w") as f:
                json.dump([e.to_dict() for e in self.entries], f, indent=2)

            logger.debug(f"Saved {len(self.entries)} cache entries to {self.cache_path}")

        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

    def get(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Check cache for similar query.

        Args:
            query: User query

        Returns:
            Cached answer dict if hit, None if miss
        """
        if not self.entries:
            return None

        # Embed the query
        try:
            query_embedding = self.embeddings.embed_query(query)
        except Exception as e:
            logger.warning(f"Failed to embed query for cache lookup: {e}")
            return None

        # Find best match
        best_entry = None
        best_similarity = 0.0

        for entry in self.entries:
            similarity = self._cosine_similarity(query_embedding, entry.query_embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_entry = entry

        if best_entry and best_similarity >= self.similarity_threshold:
            # Cache hit
            best_entry.hit_count += 1
            self._save()

            logger.info(f"Cache HIT: similarity={best_similarity:.4f}, query='{query[:50]}...'")
            return {
                "answer": best_entry.answer,
                "sources": best_entry.sources,
                "strategy": best_entry.strategy,
                "cached": True,
                "similarity": best_similarity,
                "original_query": best_entry.query,
                "hit_count": best_entry.hit_count,
            }

        logger.info(f"Cache MISS: best_similarity={best_similarity:.4f}, query='{query[:50]}...'")
        return None

    def put(
        self,
        query: str,
        answer: str,
        sources: List[Dict[str, Any]],
        strategy: str,
    ) -> None:
        """
        Store a query-answer pair in cache.

        Args:
            query: Original query
            answer: Generated answer
            sources: Source citations
            strategy: Retrieval strategy used
        """
        if not query or not answer:
            return

        try:
            query_embedding = self.embeddings.embed_query(query)
        except Exception as e:
            logger.warning(f"Failed to embed query for cache storage: {e}")
            return

        entry = CacheEntry(
            query=query,
            query_embedding=query_embedding,
            answer=answer,
            sources=sources,
            strategy=strategy,
            timestamp=time.time(),
            hit_count=0,
        )

        # Check if similar entry already exists (update instead of duplicate)
        for existing in self.entries:
            similarity = self._cosine_similarity(query_embedding, existing.query_embedding)
            if similarity >= 0.98:  # Very high similarity = likely same query
                existing.answer = answer
                existing.sources = sources
                existing.strategy = strategy
                existing.timestamp = time.time()
                self._save()
                logger.debug(f"Updated existing cache entry (similarity={similarity:.4f})")
                return

        # Add new entry
        self.entries.append(entry)

        # Trim if over max_entries (remove oldest)
        if len(self.entries) > self.max_entries:
            self.entries.sort(key=lambda e: e.timestamp)
            self.entries = self.entries[-self.max_entries:]

        self._save()
        logger.debug(f"Cached new entry, total: {len(self.entries)}")

    def clear(self) -> None:
        """Clear all cache entries."""
        self.entries = []
        self._save()
        logger.info("Cache cleared")

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self.entries:
            return {"entries": 0, "total_hits": 0, "cache_path": str(self.cache_path) if self.cache_path else "memory"}

        total_hits = sum(e.hit_count for e in self.entries)
        return {
            "entries": len(self.entries),
            "total_hits": total_hits,
            "cache_path": str(self.cache_path) if self.cache_path else "memory",
            "similarity_threshold": self.similarity_threshold,
            "oldest_entry": min(e.timestamp for e in self.entries) if self.entries else None,
            "newest_entry": max(e.timestamp for e in self.entries) if self.entries else None,
        }


def get_cache(
    embeddings: Optional[OllamaEmbeddings] = None,
    similarity_threshold: float = 0.92,
    max_entries: int = 1000,
    cache_path: Optional[str] = None,
) -> SemanticCache:
    """Get a cache instance."""
    return SemanticCache(embeddings, similarity_threshold, max_entries, cache_path)