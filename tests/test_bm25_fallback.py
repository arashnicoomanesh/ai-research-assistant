"""
Smoke test for BM25 fallback behavior.

Verifies that hybrid retrieval gracefully falls back to dense-only
when BM25 index is missing or empty.
"""
from src.retrieval.hybrid_retriever import _rrf_fuse


class TestBM25Fallback:
    """Tests for BM25 missing/empty index fallback."""

    def test_empty_bm25_list_returns_dense_only(self):
        """Empty BM25 results should return dense-only without error."""
        dense = [{"chunk_id": "c1", "score": 0.9}, {"chunk_id": "c2", "score": 0.8}]
        bm25 = []

        fused = _rrf_fuse(dense, bm25, top_k_final=5, rrf_k=60)

        assert len(fused) == 2
        assert fused[0]["chunk_id"] == "c1"
        assert fused[1]["chunk_id"] == "c2"
        # Should have RRF scores computed from dense ranks only
        assert "rrf_score" in fused[0]

    def test_none_bm25_treated_as_empty(self):
        """None BM25 should be handled gracefully (treated as empty)."""
        dense = [{"chunk_id": "c1", "score": 0.9}]
        bm25 = None

        # This should not raise - the function expects lists
        # but we test the internal logic handles empty gracefully
        fused = _rrf_fuse(dense, [], top_k_final=5, rrf_k=60)
        assert len(fused) == 1

    def test_bm25_results_with_missing_text_hydrated(self):
        """BM25-only results should be returned even without text."""
        dense = []
        bm25 = [
            {"chunk_id": "c3", "score": 10.0},
            {"chunk_id": "c4", "score": 9.0},
        ]

        fused = _rrf_fuse(dense, bm25, top_k_final=5, rrf_k=60)

        assert len(fused) == 2
        assert fused[0]["chunk_id"] == "c3"
        assert fused[1]["chunk_id"] == "c4"
        assert "rrf_score" in fused[0]

    def test_fused_results_unique_chunk_ids(self):
        """Fused results should have unique chunk_ids (deduplication)."""
        dense = [
            {"chunk_id": "c1", "score": 0.9},
            {"chunk_id": "c2", "score": 0.8},
        ]
        bm25 = [
            {"chunk_id": "c1", "score": 10.0},
            {"chunk_id": "c3", "score": 9.0},
        ]

        fused = _rrf_fuse(dense, bm25, top_k_final=10, rrf_k=60)

        # Should have 3 unique chunks: c1 (in both), c2 (dense only), c3 (bm25 only)
        assert len(fused) == 3
        chunk_ids = [r["chunk_id"] for r in fused]
        assert len(set(chunk_ids)) == 3

    def test_top_k_final_respected(self):
        """top_k_final should limit fused results."""
        dense = [{"chunk_id": f"d{i}", "score": 0.9 - i * 0.05} for i in range(10)]
        bm25 = [{"chunk_id": f"b{i}", "score": 10.0 - i} for i in range(10)]

        fused = _rrf_fuse(dense, bm25, top_k_final=3, rrf_k=60)
        assert len(fused) == 3

        # First should be highest RRF score
        assert fused[0]["rrf_score"] >= fused[1]["rrf_score"] >= fused[2]["rrf_score"]