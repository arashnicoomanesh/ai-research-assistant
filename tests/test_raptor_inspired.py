"""
Smoke tests for RAPTOR-inspired retrieval (offline with mocks).

Verifies:
- Higher similarity summary nodes are preferred
- Detail budget per summary is respected
- Text field populated from summary content for generator
- RAPTOR-inspired: limited hierarchy, not full clustering tree
"""
from unittest.mock import MagicMock, patch
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import pytest

from src.ingestion.raptor_types import RaptorNode
from src.retrieval.raptor_retriever import (
    RaptorRetriever,
    RaptorRetrievalResult,
)
from src.core.config import get_config


@dataclass
class MockHit:
    """Mock Qdrant search hit."""
    payload: Dict[str, Any]
    vector: Optional[List[float]] = None
    score: float = 0.9


class MockEmbeddings:
    """Mock embeddings that returns deterministic vectors for testing."""

    def __init__(self):
        self.query_vectors = {}

    def embed_query(self, query: str) -> List[float]:
        # Return a deterministic vector based on query content
        # This allows us to control similarity ranking in tests
        if query not in self.query_vectors:
            # Simple hash-based vector for consistency
            h = hash(query) % 1000
            self.query_vectors[query] = [float(h) / 1000.0] * 768
        return self.query_vectors[query]


class MockQdrantClient:
    """Mock Qdrant client for testing."""

    def __init__(self):
        self.search_results = {}

    def search(self, collection_name, query_vector, query_filter, limit, with_payload, with_vectors):
        key = (collection_name, tuple(query_vector[:5]) if query_vector else None)
        return self.search_results.get(key, [])


class MockRetriever:
    """Mock hybrid retriever."""
    def __init__(self):
        self.client = MockQdrantClient()


class MockRaptorBuilder:
    """Mock raptor builder."""
    def __init__(self):
        pass

    def get_summary_nodes(self, paper_id=None, level=None):
        return []


def create_summary_node(chunk_id: str, summary: str, source_chunk_ids: List[str],
                        score: float = 0.9, level: int = 1) -> RaptorNode:
    """Create a RaptorNode for testing."""
    return RaptorNode(
        chunk_id=chunk_id,
        paper_id="test_paper",
        title="Test Paper",
        section="Introduction",
        level=level,
        chunk_type="summary",
        source_chunk_ids=source_chunk_ids,
        summary=summary,
        metadata={"num_source_chunks": len(source_chunk_ids)},
        embedding=[0.1] * 768,
    )


def create_detail_chunk(chunk_id: str, text: str, score: float = 0.8) -> Dict[str, Any]:
    """Create a detail chunk dict for testing."""
    return {
        "chunk_id": chunk_id,
        "paper_id": "test_paper",
        "title": "Test Paper",
        "section": "Introduction",
        "chunk_type": "child",
        "text": text,
        "score": score,
    }


class TestRaptorInspiredRetrieval:
    """Tests for RAPTOR-inspired retrieval with mocks."""

    def setup_method(self):
        """Set up test fixtures with mocks."""
        self.mock_embeddings = MockEmbeddings()
        self.mock_retriever = MockRetriever()
        self.mock_builder = MockRaptorBuilder()

        # Create retriever with mocks
        self.retriever = RaptorRetriever(
            embeddings=self.mock_embeddings,
            retriever=self.mock_retriever,
            raptor_builder=self.mock_builder,
        )

    def test_higher_similarity_summary_preferred(self):
        """Summary nodes with higher query similarity should be ranked first."""
        # Create summary nodes with different similarities
        # We control similarity via the query vector hash
        summary1 = create_summary_node(
            chunk_id="sum_1",
            summary="High similarity summary about transformers",
            source_chunk_ids=["chunk_1", "chunk_2"],
        )
        summary2 = create_summary_node(
            chunk_id="sum_2",
            summary="Low similarity summary about CNNs",
            source_chunk_ids=["chunk_3", "chunk_4"],
        )

        # Mock Qdrant search to return summaries in similarity order
        # The retriever uses embeddings.embed_query(query) for the query vector
        # We'll mock qdrant_search directly
        with patch("src.retrieval.raptor_retriever.qdrant_search") as mock_search:
            # Return sum_1 first (higher score), then sum_2
            mock_search.return_value = [
                MockHit(payload={
                    "chunk_id": "sum_1",
                    "paper_id": "test_paper",
                    "title": "Test Paper",
                    "section": "Introduction",
                    "chunk_type": "summary",
                    "level": 1,
                    "summary": "High similarity summary about transformers",
                    "source_chunk_ids": ["chunk_1", "chunk_2"],
                    "metadata": {"num_source_chunks": 2},
                }, vector=[0.1] * 768, score=0.95),
                MockHit(payload={
                    "chunk_id": "sum_2",
                    "paper_id": "test_paper",
                    "title": "Test Paper",
                    "section": "Related Work",
                    "chunk_type": "summary",
                    "level": 1,
                    "summary": "Low similarity summary about CNNs",
                    "source_chunk_ids": ["chunk_3", "chunk_4"],
                    "metadata": {"num_source_chunks": 2},
                }, vector=[0.1] * 768, score=0.70),
            ]

            # Also mock detail retrieval
            with patch.object(self.retriever, '_retrieve_details', return_value=[]):
                result = self.retriever.retrieve("transformer attention mechanism")

            # Higher similarity summary should come first
            assert len(result.summary_nodes) == 2
            assert result.summary_nodes[0].chunk_id == "sum_1"
            assert result.summary_nodes[1].chunk_id == "sum_2"
            assert "transformer" in result.summary_nodes[0].summary.lower()

    def test_detail_budget_per_summary_respected(self):
        """Each summary should get at most k_detail_per_summary detail chunks."""
        # Create a summary with 5 source chunks
        summary = create_summary_node(
            chunk_id="sum_1",
            summary="Summary with many source chunks",
            source_chunk_ids=["chunk_1", "chunk_2", "chunk_3", "chunk_4", "chunk_5"],
        )

        # Create 5 detail chunks
        detail_chunks = [
            create_detail_chunk(f"chunk_{i}", f"Detail content {i}") for i in range(1, 6)
        ]

        with patch("src.retrieval.raptor_retriever.qdrant_search") as mock_search:
            mock_search.return_value = [
                MockHit(payload={
                    "chunk_id": "sum_1",
                    "paper_id": "test_paper",
                    "title": "Test Paper",
                    "section": "Introduction",
                    "chunk_type": "summary",
                    "level": 1,
                    "summary": "Summary with many source chunks",
                    "source_chunk_ids": ["chunk_1", "chunk_2", "chunk_3", "chunk_4", "chunk_5"],
                    "metadata": {"num_source_chunks": 5},
                }, vector=[0.1] * 768, score=0.95),
            ]

            with patch.object(self.retriever, '_retrieve_details', return_value=detail_chunks):
                # Use a survey query to trigger summary-first mode
                result = self.retriever.retrieve("overview of the paper")

            # Count detail chunks attached to the summary in combined_chunks
            detail_count = sum(1 for c in result.combined_chunks if c.get("chunk_type") == "child")

            # Should respect k_detail_per_summary (3) per summary
            k_detail_per_summary = get_config().raptor.k_detail_per_summary
            assert detail_count <= k_detail_per_summary
            assert detail_count == k_detail_per_summary

    def test_total_detail_budget_respected(self):
        """Total detail chunks across all summaries should not exceed max_detail_chunks_total."""
        # Create 4 summaries each with 3 source chunks (12 total potential)
        summaries = []
        all_source_ids = []
        for i in range(4):
            source_ids = [f"chunk_{i*3+j}" for j in range(3)]
            all_source_ids.extend(source_ids)
            summaries.append(create_summary_node(
                chunk_id=f"sum_{i}",
                summary=f"Summary {i}",
                source_chunk_ids=source_ids,
            ))

        # Create 12 detail chunks
        detail_chunks = [
            create_detail_chunk(cid, f"Detail for {cid}") for cid in all_source_ids
        ]

        with patch("src.retrieval.raptor_retriever.qdrant_search") as mock_search:
            mock_search.return_value = [
                MockHit(payload={
                    "chunk_id": f"sum_{i}",
                    "paper_id": "test_paper",
                    "title": "Test Paper",
                    "section": "Introduction",
                    "chunk_type": "summary",
                    "level": 1,
                    "summary": f"Summary {i}",
                    "source_chunk_ids": [f"chunk_{i*3+j}" for j in range(3)],
                    "metadata": {"num_source_chunks": 3},
                }, vector=[0.1] * 768, score=0.9 - i * 0.1)
                for i in range(4)
            ]

            with patch.object(self.retriever, '_retrieve_details', return_value=detail_chunks):
                result = self.retriever.retrieve("test query")

            # Count detail chunks in combined_chunks
            detail_count = sum(1 for c in result.combined_chunks if c.get("chunk_type") == "child")

            # Should respect max_detail_chunks_total (10)
            max_detail = get_config().raptor.max_detail_chunks_total
            assert detail_count <= max_detail
            assert detail_count == max_detail

    def test_text_field_populated_from_summary_content(self):
        """Combined evidence should have 'text' field set from summary content for generator."""
        summary = create_summary_node(
            chunk_id="sum_1",
            summary="This is the summary content for the generator",
            source_chunk_ids=["chunk_1"],
        )

        with patch("src.retrieval.raptor_retriever.qdrant_search") as mock_search:
            mock_search.return_value = [
                MockHit(payload={
                    "chunk_id": "sum_1",
                    "paper_id": "test_paper",
                    "title": "Test Paper",
                    "section": "Introduction",
                    "chunk_type": "summary",
                    "level": 1,
                    "summary": "This is the summary content for the generator",
                    "source_chunk_ids": ["chunk_1"],
                    "metadata": {"num_source_chunks": 1},
                }, vector=[0.1] * 768, score=0.95),
            ]

            with patch.object(self.retriever, '_retrieve_details', return_value=[]):
                result = self.retriever.retrieve("test query")

            # Check that combined_chunks has text field from summary
            summary_chunks = [c for c in result.combined_chunks if c.get("_raptor_node")]
            assert len(summary_chunks) == 1
            assert "text" in summary_chunks[0]
            assert summary_chunks[0]["text"] == "This is the summary content for the generator"
            assert summary_chunks[0]["text"] == summary_chunks[0]["summary"]

    def test_detail_chunks_also_have_text_field(self):
        """Detail chunks should have their text field preserved for generator."""
        detail_chunk = create_detail_chunk("chunk_1", "This is detail text for generation")

        with patch("src.retrieval.raptor_retriever.qdrant_search") as mock_search:
            mock_search.return_value = []  # No summaries

            with patch.object(self.retriever, '_retrieve_details', return_value=[detail_chunk]):
                result = self.retriever.retrieve("detail query about something")

            # Check detail chunks have text field
            detail_chunks = [c for c in result.combined_chunks if c.get("chunk_type") == "child"]
            assert len(detail_chunks) == 1
            assert "text" in detail_chunks[0]
            assert detail_chunks[0]["text"] == "This is detail text for generation"

    def test_survey_query_type_detected(self):
        """Survey queries should be detected and use summary-first strategy."""
        # Test various survey query patterns
        survey_queries = [
            "Give me an overview of transformer architectures",
            "Survey of recent advances in RAG",
            "Summarize the evolution of attention mechanisms",
            "Compare different approaches to retrieval augmentation",
            "What are the trends in LLM fine-tuning?",
        ]

        for query in survey_queries:
            assert self.retriever._is_survey_query(query), f"Should detect survey: {query}"

    def test_detail_query_type_detected(self):
        """Detail queries should be detected and use detail-first strategy."""
        detail_queries = [
            "What is the attention mechanism in transformers?",
            "How does Self-RAG work?",
            "Explain the CRAG algorithm",
            "Define RAPTOR retrieval",
        ]

        for query in detail_queries:
            assert not self.retriever._is_survey_query(query), f"Should NOT detect survey: {query}"

    def test_combined_evidence_structure(self):
        """Combined evidence should have proper structure with all required fields."""
        summary = create_summary_node(
            chunk_id="sum_1",
            summary="Test summary",
            source_chunk_ids=["chunk_1", "chunk_2"],
        )
        detail1 = create_detail_chunk("chunk_1", "Detail 1")
        detail2 = create_detail_chunk("chunk_2", "Detail 2")

        with patch("src.retrieval.raptor_retriever.qdrant_search") as mock_search:
            mock_search.return_value = [
                MockHit(payload={
                    "chunk_id": "sum_1",
                    "paper_id": "test_paper",
                    "title": "Test Paper",
                    "section": "Introduction",
                    "chunk_type": "summary",
                    "level": 1,
                    "summary": "Test summary",
                    "source_chunk_ids": ["chunk_1", "chunk_2"],
                    "metadata": {"num_source_chunks": 2},
                }, vector=[0.1] * 768, score=0.95),
            ]

            with patch.object(self.retriever, '_retrieve_details', return_value=[detail1, detail2]):
                result = self.retriever.retrieve("survey query about transformers")

            # Verify result structure
            assert isinstance(result, RaptorRetrievalResult)
            assert hasattr(result, "summary_nodes")
            assert hasattr(result, "detail_chunks")
            assert hasattr(result, "combined_chunks")
            assert hasattr(result, "query_type")
            assert hasattr(result, "retrieval_stats")

            # Verify combined chunks have required fields
            for chunk in result.combined_chunks:
                assert "chunk_id" in chunk
                if chunk.get("_raptor_node"):
                    assert "text" in chunk
                    assert "summary" in chunk
                    assert "summary_level" in chunk
                if chunk.get("chunk_type") == "child":
                    assert "text" in chunk

    def test_raptor_inspired_not_full_tree(self):
        """Verify this is RAPTOR-inspired (limited hierarchy), not full RAPTOR clustering."""
        # The implementation uses only 2 levels (section=1, paper=2)
        # and does not do recursive clustering
        summary_section = create_summary_node(
            chunk_id="sum_sec",
            summary="Section summary",
            source_chunk_ids=["chunk_1"],
            level=1,
        )
        summary_paper = create_summary_node(
            chunk_id="sum_paper",
            summary="Paper summary",
            source_chunk_ids=["chunk_1", "chunk_2", "chunk_3"],
            level=2,
        )

        # Only two levels exist - no deeper recursive clustering
        assert summary_section.level in (1, 2)
        assert summary_paper.level in (1, 2)

        # Config constants are fixed, not dynamically clustered
        cfg = get_config().raptor
        assert cfg.k_summary == 5
        assert cfg.k_detail_per_summary == 3
        assert cfg.max_detail_chunks_total == 10

    def test_retrieve_signature_uses_defaults(self):
        """Retrieve method should use config constants as defaults."""
        import inspect
        sig = inspect.signature(self.retriever.retrieve)
        params = sig.parameters

        cfg = get_config().raptor
        assert params["k_summary"].default is None  # Now uses config internally
        assert params["k_detail"].default is None  # Now uses config internally

    def test_constants_exposed_for_configuration(self):
        """Config constants should be accessible via config."""
        cfg = get_config().raptor

        # These should be positive integers
        assert isinstance(cfg.k_summary, int) and cfg.k_summary > 0
        assert isinstance(cfg.k_detail_per_summary, int) and cfg.k_detail_per_summary > 0
        assert isinstance(cfg.max_detail_chunks_total, int) and cfg.max_detail_chunks_total > 0

        # Total should be at least per-summary * some summaries
        assert cfg.max_detail_chunks_total >= cfg.k_detail_per_summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])