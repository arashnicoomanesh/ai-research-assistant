"""
Smoke test for sludge filter.

Verifies that table dumps, OCR noise, and numeric blocks are filtered
while normal prose is kept.
"""
from src.retrieval.sludge_filter import is_sludge, filter_sludge


class TestSludgeFilter:
    """Tests for sludge detection and filtering."""

    def test_table_dump_filtered(self):
        """Table-like content with pipes and numbers should be filtered."""
        sludge = "123 45 67 | --- | 89 10 11 | 12 13 14"
        assert is_sludge(sludge) is True

    def test_mostly_numeric_filtered(self):
        """Content with high digit ratio should be filtered."""
        sludge = "12345 67890 11111 22222 33333"
        assert is_sludge(sludge) is True

    def test_mostly_punctuation_filtered(self):
        """Content with high punctuation ratio should be filtered."""
        sludge = "| --- | --- | --- | --- | --- |"
        assert is_sludge(sludge) is True

    def test_glued_tokens_filtered(self):
        """Glued camelCase tokens without spaces should be filtered."""
        sludge = "ThisIsAGluedTokenWithNoSpacesAndMixedCase123"
        assert is_sludge(sludge) is True

    def test_normal_prose_kept(self):
        """Normal readable prose should NOT be filtered."""
        prose = "Self-RAG is a retrieval-augmented generation method that uses reflection tokens."
        assert is_sludge(prose) is False

    def test_short_meaningful_prose_kept(self):
        """Short but meaningful prose should NOT be filtered."""
        prose = "BERT uses masked language modeling."
        assert is_sludge(prose) is False

    def test_empty_string_filtered(self):
        """Empty or whitespace-only strings should be filtered."""
        assert is_sludge("") is True
        assert is_sludge("   ") is True
        assert is_sludge("\n\t") is True

    def test_filter_sludge_function(self):
        """filter_sludge should remove sludge chunks and keep good ones."""
        chunks = [
            {"chunk_id": "c1", "text": "123 45 67 | --- | 89", "score": 0.9},
            {"chunk_id": "c2", "text": "Self-RAG is a retrieval method.", "score": 0.8},
            {"chunk_id": "c3", "text": "ThisIsAGluedToken", "score": 0.7},
            {"chunk_id": "c4", "text": "BERT uses masked language modeling.", "score": 0.6},
        ]

        filtered = filter_sludge(chunks)
        assert len(filtered) == 2
        assert filtered[0]["chunk_id"] == "c2"
        assert filtered[1]["chunk_id"] == "c4"