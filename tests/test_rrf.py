"""
Smoke test for Reciprocal Rank Fusion (RRF).

Verifies that the RRF fusion correctly combines ranked lists
from dense and BM25 retrievers.
"""
from src.retrieval.hybrid_retriever import _rrf_fuse


class TestRRF:
    """Tests for Reciprocal Rank Fusion."""

    def test_rrf_basic_fusion(self):
        """Two ranked lists should fuse with correct RRF scores (k=60)."""
        dense_results = [
            {"chunk_id": "c1", "score": 0.9},
            {"chunk_id": "c2", "score": 0.8},
            {"chunk_id": "c3", "score": 0.7},
        ]
        bm25_results = [
            {"chunk_id": "c2", "score": 10.0},
            {"chunk_id": "c4", "score": 9.0},
            {"chunk_id": "c1", "score": 8.0},
        ]

        fused = _rrf_fuse(dense_results, bm25_results, top_k_final=10, rrf_k=60)

        # Verify order: c2 (rank 1 BM25 + rank 2 dense) > c1 (rank 1 dense + rank 3 BM25)
        # c2: 1/61 + 1/62 = 0.01639 + 0.01613 = 0.03252
        # c1: 1/61 + 1/63 = 0.01639 + 0.01587 = 0.03226
        # c4: 1/62 = 0.01613
        # c3: 1/63 = 0.01587
        assert fused[0]["chunk_id"] == "c2"
        assert fused[1]["chunk_id"] == "c1"
        assert fused[2]["chunk_id"] == "c4"
        assert fused[3]["chunk_id"] == "c3"

    def test_rrf_top_k_limit(self):
        """top_k_final should limit results."""
        dense = [{"chunk_id": f"d{i}", "score": 0.9 - i * 0.1} for i in range(10)]
        bm25 = [{"chunk_id": f"b{i}", "score": 10.0 - i} for i in range(10)]

        fused = _rrf_fuse(dense, bm25, top_k_final=5, rrf_k=60)
        assert len(fused) == 5

    def test_rrf_dense_only(self):
        """Empty BM25 list should return dense-only with RRF scores."""
        dense = [{"chunk_id": "c1", "score": 0.9}, {"chunk_id": "c2", "score": 0.8}]
        bm25 = []

        fused = _rrf_fuse(dense, bm25, top_k_final=5, rrf_k=60)
        assert len(fused) == 2
        assert fused[0]["chunk_id"] == "c1"
        assert "rrf_score" in fused[0]

    def test_rrf_bm25_only(self):
        """Empty dense list should return BM25-only with RRF scores."""
        dense = []
        bm25 = [{"chunk_id": "c3", "score": 10.0}, {"chunk_id": "c4", "score": 9.0}]

        fused = _rrf_fuse(dense, bm25, top_k_final=5, rrf_k=60)
        assert len(fused) == 2
        assert fused[0]["chunk_id"] == "c3"
        assert "rrf_score" in fused[0]

    def test_rrf_deduplication(self):
        """Duplicate chunk_ids should be merged with combined RRF score."""
        dense = [
            {"chunk_id": "c1", "score": 0.9},
            {"chunk_id": "c2", "score": 0.8},
        ]
        bm25 = [
            {"chunk_id": "c1", "score": 10.0},
            {"chunk_id": "c3", "score": 9.0},
        ]

        fused = _rrf_fuse(dense, bm25, top_k_final=10, rrf_k=60)
        # c1 appears in both lists, should have higher RRF score
        c1 = next(r for r in fused if r["chunk_id"] == "c1")
        c2 = next(r for r in fused if r["chunk_id"] == "c2")
        c3 = next(r for r in fused if r["chunk_id"] == "c3")

        assert c1["rrf_score"] > c2["rrf_score"]
        assert c1["rrf_score"] > c3["rrf_score"]

    def test_rrf_k_constant(self):
        """Custom RRF k constant should affect scores."""
        dense = [{"chunk_id": "c1", "score": 0.9}]
        bm25 = [{"chunk_id": "c1", "score": 10.0}]

        # k=10: 1/11 + 1/11 = 0.1818
        fused_10 = _rrf_fuse(dense, bm25, top_k_final=5, rrf_k=10)
        # k=60: 1/61 + 1/61 = 0.0328
        fused_60 = _rrf_fuse(dense, bm25, top_k_final=5, rrf_k=60)

        assert fused_10[0]["rrf_score"] > fused_60[0]["rrf_score"]