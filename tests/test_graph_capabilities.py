"""
Tests for GraphRAG capability manifest.

Verifies that Stage A flags correctly indicate which Microsoft-style
GraphRAG features are NOT implemented in the current implementation.
"""
from src.graph.capabilities import (
    GRAPHRAG_STAGE,
    HAS_COMMUNITY_DETECTION,
    HAS_GLOBAL_SUMMARY,
    HAS_MAP_REDUCE,
    HAS_COMMUNITY_REPORTS,
    STRATEGY_LOCAL,
    describe,
)


class TestGraphCapabilities:
    """Tests for GraphRAG Stage A capability manifest."""

    def test_stage_is_a(self):
        """GraphRAG stage should be A (local graph only)."""
        assert GRAPHRAG_STAGE == "A"

    def test_strategy_local(self):
        """Strategy identifier should be graph_rag for local graph retrieval."""
        assert STRATEGY_LOCAL == "graph_rag"

    def test_community_detection_false(self):
        """Community detection is NOT implemented in Stage A."""
        assert HAS_COMMUNITY_DETECTION is False

    def test_global_summary_false(self):
        """Global community summaries are NOT implemented in Stage A."""
        assert HAS_GLOBAL_SUMMARY is False

    def test_map_reduce_false(self):
        """Map-reduce over communities is NOT implemented in Stage A."""
        assert HAS_MAP_REDUCE is False

    def test_community_reports_false(self):
        """Community report generation is NOT implemented in Stage A."""
        assert HAS_COMMUNITY_REPORTS is False

    def test_describe_returns_string(self):
        """describe() should return a non-empty string."""
        desc = describe()
        assert isinstance(desc, str)
        assert len(desc) > 0
        assert "Stage A" in desc
        assert "local entity graph" in desc.lower()
        assert "community detection" in desc.lower()
        assert "global summaries" in desc.lower()

    def test_no_stage_b_features_true(self):
        """All Stage B feature flags should be False."""
        stage_b_flags = [
            HAS_COMMUNITY_DETECTION,
            HAS_GLOBAL_SUMMARY,
            HAS_MAP_REDUCE,
            HAS_COMMUNITY_REPORTS,
        ]
        assert all(flag is False for flag in stage_b_flags)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])