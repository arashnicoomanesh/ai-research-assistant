"""
Embeddings client for generating vector representations via Ollama.

Provides embedding generation for text chunks using local embedding models.
Supports batch processing with retry logic for CPU-safe operation.
"""
import logging
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Union
import requests
import numpy as np
from requests.exceptions import RequestException, Timeout
from src.core.config import get_config

logger = logging.getLogger(__name__)


class OllamaEmbeddings:
    """Client for Ollama embeddings API with retry logic."""

    _session: Optional[requests.Session] = None
    _session_lock = threading.Lock()

    # Query embedding cache: key = (model, normalized_query), value = embedding vector
    # Uses OrderedDict for FIFO eviction when max size reached
    _query_cache: "OrderedDict[tuple[str, str], List[float]]" = OrderedDict()
    _cache_lock = threading.Lock()

    @classmethod
    def _get_max_cache_size(cls) -> int:
        """Get max cache size from config."""
        return get_config().embedding.query_cache_max_size

    def __init__(
        self,
        host: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
        batch_size: Optional[int] = None,
        max_retries: Optional[int] = None,
    ):
        """
        Initialize Ollama embeddings client.

        Args:
            host: Ollama server URL (default from config)
            model: Embedding model name (default from config)
            timeout: Request timeout in seconds (default from config)
            batch_size: Batch size for embed_batch (default from config)
            max_retries: Maximum retry attempts on failure (default from config)
        """
        embedding_config = get_config().embedding
        self.host = host or get_config().ollama.host
        self.model = model or get_config().ollama.embedding_model
        self.timeout = timeout or embedding_config.timeout_seconds
        self.batch_size = batch_size or embedding_config.batch_size
        self.max_retries = max_retries if max_retries is not None else embedding_config.max_retries
        self._base_url = f"{self.host}/api"
        self._embedding_dim: Optional[int] = None

    @classmethod
    def _get_session(cls) -> requests.Session:
        """Get or create the shared requests.Session (thread-safe)."""
        if cls._session is None:
            with cls._session_lock:
                if cls._session is None:
                    cls._session = requests.Session()
                    logger.debug("Created shared requests.Session for OllamaEmbeddings")
        return cls._session

    @classmethod
    def close_session(cls) -> None:
        """Close the shared session (for cleanup/testing)."""
        with cls._session_lock:
            if cls._session is not None:
                cls._session.close()
                cls._session = None
                logger.debug("Closed shared requests.Session for OllamaEmbeddings")

    @classmethod
    def _normalize_query(cls, query: str) -> str:
        """Normalize query for cache key."""
        return query.strip().lower()

    @classmethod
    def _get_cache_key(cls, model: str, query: str) -> tuple[str, str]:
        """Generate cache key from model and normalized query."""
        return (model, cls._normalize_query(query))

    @classmethod
    def _get_cached_embedding(cls, model: str, query: str) -> Optional[List[float]]:
        """Get embedding from cache if present."""
        key = cls._get_cache_key(model, query)
        with cls._cache_lock:
            if key in cls._query_cache:
                # Move to end (most recently used)
                embedding = cls._query_cache.pop(key)
                cls._query_cache[key] = embedding
                logger.debug(f"Query embedding cache hit for model={model}")
                return embedding
        return None

    @classmethod
    def _cache_embedding(cls, model: str, query: str, embedding: List[float]) -> None:
        """Store embedding in cache with FIFO eviction."""
        key = cls._get_cache_key(model, query)
        max_cache_size = cls._get_max_cache_size()
        with cls._cache_lock:
            if key in cls._query_cache:
                # Update existing
                cls._query_cache.pop(key)
            elif len(cls._query_cache) >= max_cache_size:
                # FIFO eviction
                cls._query_cache.popitem(last=False)
            cls._query_cache[key] = embedding

    @classmethod
    def clear_query_cache(cls) -> None:
        """Clear the query embedding cache (for testing)."""
        with cls._cache_lock:
            cls._query_cache.clear()
            logger.debug("Cleared query embedding cache")

    def embed_one(self, text: str) -> List[float]:
        """
        Generate embedding for a single text (convenience method).

        Args:
            text: Input text to embed

        Returns:
            Embedding vector as list of floats
        """
        return self.embed_batch([text])[0]

    def embed(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Input text to embed

        Returns:
            Embedding vector as list of floats
        """
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts with retry logic.

        Args:
            texts: List of input texts

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        # Process in sub-batches of batch_size for better timeout handling
        all_embeddings = []
        total = len(texts)
        
        for i in range(0, total, self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_num = (i // self.batch_size) + 1
            total_batches = (total + self.batch_size - 1) // self.batch_size
            
            logger.info(f"Embedding batch {batch_num}/{total_batches} (size={len(batch)})")
            
            embeddings = self._embed_batch_with_retry(batch, batch_num)
            all_embeddings.extend(embeddings)
        
        return all_embeddings

    def _embed_batch_with_retry(self, texts: List[str], batch_num: int) -> List[List[float]]:
        """
        Embed a single batch with exponential backoff retry.

        Args:
            texts: List of texts to embed
            batch_num: Batch number for logging

        Returns:
            List of embedding vectors
        """
        # For single-text queries, check cache first
        if len(texts) == 1:
            cached = self._get_cached_embedding(self.model, texts[0])
            if cached is not None:
                logger.debug(f"Batch {batch_num}: cache hit for single query")
                return [cached]

        last_exception: Optional[BaseException] = None
        
        for attempt in range(1, self.max_retries + 1):
            try:
                payload: Dict[str, Any] = {
                    "model": self.model,
                    "input": texts
                }

                logger.debug(f"Batch {batch_num}, attempt {attempt}/{self.max_retries}: calling Ollama embed API")
                response = self._get_session().post(
                    f"{self._base_url}/embed",
                    json=payload,
                    timeout=self.timeout
                )
                response.raise_for_status()

                data: Dict[str, Any] = response.json()
                embeddings: List[List[float]] = data.get("embeddings", [])
                
                if len(embeddings) != len(texts):
                    raise ValueError(f"Expected {len(texts)} embeddings, got {len(embeddings)}")

                # Cache embedding dimension on first successful call
                if self._embedding_dim is None and embeddings:
                    self._embedding_dim = len(embeddings[0])

                # Cache single-query results
                if len(texts) == 1:
                    self._cache_embedding(self.model, texts[0], embeddings[0])

                logger.info(f"Batch {batch_num} completed: {len(embeddings)} embeddings generated")
                return embeddings

            except Timeout as e:
                last_exception = e
                wait_time = 2 ** attempt  # Exponential backoff: 2, 4, 8...
                logger.warning(f"Batch {batch_num} attempt {attempt} timed out after {self.timeout}s. Retrying in {wait_time}s...")
                time.sleep(wait_time)
                
            except RequestException as e:
                last_exception = e
                wait_time = 2 ** attempt
                logger.warning(f"Batch {batch_num} attempt {attempt} failed: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
                
            except Exception as e:
                last_exception = e
                wait_time = 2 ** attempt
                logger.warning(f"Batch {batch_num} attempt {attempt} failed with error: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)

        # All retries exhausted
        logger.error(f"Batch {batch_num} failed after {self.max_retries} attempts. Last error: {last_exception}")
        if last_exception:
            raise last_exception
        raise RuntimeError("All retries exhausted with no exception captured")

    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a search query.
        Uses in-process cache for repeated identical queries.

        Args:
            query: Search query text

        Returns:
            Query embedding vector
        """
        # Check cache first
        cached = self._get_cached_embedding(self.model, query)
        if cached is not None:
            return cached
        
        # Not in cache, embed and cache
        embedding = self.embed(query)
        self._cache_embedding(self.model, query, embedding)
        return embedding

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of documents.

        Args:
            documents: List of document texts

        Returns:
            List of document embedding vectors
        """
        return self.embed_batch(documents)

    @property
    def dimension(self) -> int:
        """Get the embedding dimension (requires at least one embedding call first)."""
        if self._embedding_dim is None:
            # Make a dummy call to get dimension
            self.embed("test")
        return self._embedding_dim or 0

    def is_available(self) -> bool:
        """Check if Ollama server and embedding model are available."""
        try:
            response = self._get_session().get(f"{self.host}/api/tags", timeout=5)
            if response.status_code != 200:
                return False
            data: Dict[str, Any] = response.json()
            models = [m["name"] for m in data.get("models", [])]
            return any(self.model in m for m in models)
        except requests.RequestException:
            return False

    def similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Compute cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity score (-1 to 1)
        """
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


# Convenience function for quick usage
def get_embeddings() -> OllamaEmbeddings:
    """Get a default embeddings client instance."""
    return OllamaEmbeddings()