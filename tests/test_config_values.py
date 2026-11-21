"""
Tests for config values - verifies that all magic numbers moved to config
have correct defaults matching the original hardcoded values.
"""
from src.core.config import get_config


class TestConfigValues:
    """Verify config defaults match original hardcoded values."""

    def test_retrieval_rrf_k(self):
        """RRF k should default to 60 (was RRF_K = 60)."""
        cfg = get_config().retrieval
        assert cfg.rrf_k == 60

    def test_retrieval_sludge_thresholds(self):
        """Sludge filter thresholds should match original module-level constants."""
        cfg = get_config().retrieval
        assert cfg.sludge_min_letter_ratio == 0.30
        assert cfg.sludge_max_digit_ratio == 0.50
        assert cfg.sludge_max_punct_ratio == 0.40
        assert cfg.sludge_min_words == 5
        assert cfg.sludge_min_alpha_words == 2
        assert cfg.sludge_max_glued_ratio == 0.05

    def test_embedding_query_cache_max_size(self):
        """Embedding query cache max size should default to 256 (was _max_cache_size = 256)."""
        cfg = get_config().embedding
        assert cfg.query_cache_max_size == 256

    def test_cache_similarity_threshold(self):
        """Cache similarity threshold should default to 0.92 (was 0.92 in semantic_cache)."""
        cfg = get_config().cache
        assert cfg.similarity_threshold == 0.92

    def test_cache_duplicate_threshold(self):
        """Cache duplicate threshold should default to 0.98 (was hardcoded 0.98 in semantic_cache.put)."""
        cfg = get_config().cache
        assert cfg.duplicate_threshold == 0.98

    def test_cache_max_entries(self):
        """Cache max entries should default to 1000."""
        cfg = get_config().cache
        assert cfg.max_entries == 1000

    def test_raptor_k_summary(self):
        """RAPTOR k_summary should default to 5 (was DEFAULT_K_SUMMARY = 5)."""
        cfg = get_config().raptor
        assert cfg.k_summary == 5

    def test_raptor_k_detail_per_summary(self):
        """RAPTOR k_detail_per_summary should default to 3 (was DEFAULT_K_DETAIL_PER_SUMMARY = 3)."""
        cfg = get_config().raptor
        assert cfg.k_detail_per_summary == 3

    def test_raptor_max_detail_chunks_total(self):
        """RAPTOR max_detail_chunks_total should default to 10 (was MAX_DETAIL_CHUNKS_TOTAL = 10)."""
        cfg = get_config().raptor
        assert cfg.max_detail_chunks_total == 10

    def test_raptor_builder_max_tokens(self):
        """RAPTOR builder max_tokens should default to 256 (was max_tokens=256 in raptor_builder)."""
        cfg = get_config().raptor
        assert cfg.builder_max_tokens == 256

    def test_raptor_builder_batch_size(self):
        """RAPTOR builder batch_size should default to 16 (was batch_size=16 in raptor_builder)."""
        cfg = get_config().raptor
        assert cfg.builder_batch_size == 16

    def test_raptor_builder_max_chunks_per_group(self):
        """RAPTOR builder max_chunks_per_group should default to 20 (was 20 in raptor_builder)."""
        cfg = get_config().raptor
        assert cfg.builder_max_chunks_per_group == 20

    def test_multi_step_max_sub_questions(self):
        """Multi-step max_sub_questions should default to 3 (was MAX_SUB_QUESTIONS = 3)."""
        cfg = get_config().multi_step
        assert cfg.max_sub_questions == 3

    def test_multi_step_max_follow_up(self):
        """Multi-step max_follow_up should default to 1 (was MAX_FOLLOW_UP = 1)."""
        cfg = get_config().multi_step
        assert cfg.max_follow_up == 1

    def test_graph_entity_extract_max_chars(self):
        """Graph entity extract max_chars should default to 900 (was 900 in entity_extractor)."""
        cfg = get_config().graph
        assert cfg.entity_extract_max_chars == 900

    def test_graph_entity_extract_fallback_chars(self):
        """Graph entity extract fallback_chars should default to 600 (was 600 in entity_extractor)."""
        cfg = get_config().graph
        assert cfg.entity_extract_fallback_chars == 600


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])