"""
Smoke test for semantic cache with TTL and model invalidation.

Verifies:
- TTL expiration
- TTL=0 means no expiry
- Model identity mismatch invalidates cache
- Cache invalidate method works
"""
import time
import tempfile
import os
from unittest.mock import patch, MagicMock

from src.retrieval.semantic_cache import SemanticCache, CacheEntry


class TestSemanticCache:
    """Tests for semantic cache with TTL and model invalidation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_embeddings = MagicMock()
        self.mock_embeddings.embed_query.return_value = [0.1] * 384

    def test_ttl_zero_means_no_expiry(self):
        """TTL=0 should mean entries never expire by time."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "cache.json")
            cache = SemanticCache(
                embeddings=self.mock_embeddings,
                cache_path=cache_path,
                ttl_seconds=0,
                llm_model="test-model",
                embedding_model="test-embed",
            )

            # Add an entry
            cache.put("test query", "test answer", [], "test_strategy")

            # Manually set timestamp to very old
            cache.entries[0].timestamp = time.time() - 86400  # 1 day ago

            # Should still hit (TTL=0 means no expiry)
            result = cache.get("test query")
            assert result is not None
            assert result["answer"] == "test answer"

    def test_expired_ttl_returns_miss(self):
        """Expired entries should be treated as miss and removed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "cache.json")
            cache = SemanticCache(
                embeddings=self.mock_embeddings,
                cache_path=cache_path,
                ttl_seconds=1,  # 1 second TTL
                llm_model="test-model",
                embedding_model="test-embed",
            )

            # Add an entry
            cache.put("test query", "test answer", [], "test_strategy")

            # Wait for expiry
            time.sleep(1.5)

            # Should miss (expired)
            result = cache.get("test query")
            assert result is None
            assert len(cache.entries) == 0  # Expired entry removed

    def test_ttl_not_expired_returns_hit(self):
        """Non-expired entries should return hit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "cache.json")
            cache = SemanticCache(
                embeddings=self.mock_embeddings,
                cache_path=cache_path,
                ttl_seconds=10,  # 10 second TTL
                llm_model="test-model",
                embedding_model="test-embed",
            )

            # Add an entry
            cache.put("test query", "test answer", [], "test_strategy")

            # Should hit immediately
            result = cache.get("test query")
            assert result is not None
            assert result["answer"] == "test answer"

    def test_model_mismatch_invalidates_on_load(self):
        """Cache entries with different model should be ignored on load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "cache.json")

            # Create cache with one model
            cache1 = SemanticCache(
                embeddings=self.mock_embeddings,
                cache_path=cache_path,
                ttl_seconds=0,
                llm_model="model-v1",
                embedding_model="embed-v1",
            )
            cache1.put("test query", "test answer", [], "test_strategy")

            # Load with different model
            cache2 = SemanticCache(
                embeddings=self.mock_embeddings,
                cache_path=cache_path,
                ttl_seconds=0,
                llm_model="model-v2",  # Different model
                embedding_model="embed-v1",
            )

            # Should miss because model changed
            result = cache2.get("test query")
            assert result is None
            assert len(cache2.entries) == 0

    def test_embedding_model_mismatch_invalidates_on_load(self):
        """Cache entries with different embedding model should be ignored on load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "cache.json")

            # Create cache with one embedding model
            cache1 = SemanticCache(
                embeddings=self.mock_embeddings,
                cache_path=cache_path,
                ttl_seconds=0,
                llm_model="model-v1",
                embedding_model="embed-v1",
            )
            cache1.put("test query", "test answer", [], "test_strategy")

            # Load with different embedding model
            cache2 = SemanticCache(
                embeddings=self.mock_embeddings,
                cache_path=cache_path,
                ttl_seconds=0,
                llm_model="model-v1",
                embedding_model="embed-v2",  # Different embedding model
            )

            # Should miss because embedding model changed
            result = cache2.get("test query")
            assert result is None
            assert len(cache2.entries) == 0

    def test_same_models_loads_correctly(self):
        """Cache with matching models should load correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "cache.json")

            cache1 = SemanticCache(
                embeddings=self.mock_embeddings,
                cache_path=cache_path,
                ttl_seconds=0,
                llm_model="model-v1",
                embedding_model="embed-v1",
            )
            cache1.put("test query", "test answer", [], "test_strategy")

            # Load with same models
            cache2 = SemanticCache(
                embeddings=self.mock_embeddings,
                cache_path=cache_path,
                ttl_seconds=0,
                llm_model="model-v1",
                embedding_model="embed-v1",
            )

            # Should hit
            result = cache2.get("test query")
            assert result is not None
            assert result["answer"] == "test answer"

    def test_invalidate_clears_entries(self):
        """invalidate() should clear all entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "cache.json")
            
            # Create fresh mock for this test
            mock_emb = MagicMock()
            mock_emb.embed_query.return_value = [0.1] * 384
            
            cache = SemanticCache(
                embeddings=mock_emb,
                cache_path=cache_path,
                ttl_seconds=0,
                llm_model="test-model",
                embedding_model="test-embed",
            )

            # Manually add two entries with different embeddings to avoid deduplication
            cache.entries.append(CacheEntry(
                query="query1",
                query_embedding=[0.1] * 384,
                answer="answer1",
                sources=[],
                strategy="strat1",
                timestamp=time.time(),
                hit_count=0,
                llm_model="test-model",
                embedding_model="test-embed",
            ))
            cache.entries.append(CacheEntry(
                query="query2",
                query_embedding=[0.2] * 384,
                answer="answer2",
                sources=[],
                strategy="strat2",
                timestamp=time.time(),
                hit_count=0,
                llm_model="test-model",
                embedding_model="test-embed",
            ))
            cache._save()
            assert len(cache.entries) == 2

            cleared = cache.invalidate(reason="test")
            assert cleared == 2
            assert len(cache.entries) == 0

            # Should miss after invalidate
            mock_emb.embed_query.return_value = [0.1] * 384
            result = cache.get("query1")
            assert result is None

    def test_invalidate_persists(self):
        """Invalidate should persist to disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "cache.json")
            cache1 = SemanticCache(
                embeddings=self.mock_embeddings,
                cache_path=cache_path,
                ttl_seconds=0,
                llm_model="test-model",
                embedding_model="test-embed",
            )
            cache1.put("test query", "test answer", [], "test_strategy")
            cache1.invalidate(reason="test")

            # Reload and verify empty
            cache2 = SemanticCache(
                embeddings=self.mock_embeddings,
                cache_path=cache_path,
                ttl_seconds=0,
                llm_model="test-model",
                embedding_model="test-embed",
            )
            assert len(cache2.entries) == 0

    def test_stats_includes_model_and_ttl(self):
        """stats() should include model info and TTL."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "cache.json")
            cache = SemanticCache(
                embeddings=self.mock_embeddings,
                cache_path=cache_path,
                ttl_seconds=3600,
                llm_model="test-model",
                embedding_model="test-embed",
            )
            cache.put("test", "answer", [], "strat")

            stats = cache.stats()
            assert stats["ttl_seconds"] == 3600
            assert stats["llm_model"] == "test-model"
            assert stats["embedding_model"] == "test-embed"
            assert stats["entries"] == 1

    def test_empty_cache_on_model_change_new_entry(self):
        """New entries with different model should start fresh cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "cache.json")

            cache1 = SemanticCache(
                embeddings=self.mock_embeddings,
                cache_path=cache_path,
                ttl_seconds=0,
                llm_model="model-v1",
                embedding_model="embed-v1",
            )
            cache1.put("query1", "answer1", [], "strat1")

            # Create new cache with different model
            cache2 = SemanticCache(
                embeddings=self.mock_embeddings,
                cache_path=cache_path,
                ttl_seconds=0,
                llm_model="model-v2",
                embedding_model="embed-v1",
            )
            # Old entries should be gone
            assert len(cache2.entries) == 0

            # Add new entry with new model
            cache2.put("query2", "answer2", [], "strat2")
            assert len(cache2.entries) == 1
            assert cache2.entries[0].llm_model == "model-v2"