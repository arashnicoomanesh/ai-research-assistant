"""
Smoke test for query rewriter entity preservation.

Verifies that named entities like "Self-RAG", "CRAG", "BERT", "GPT-3",
"FlashAttention", "LoRA" are preserved in rewritten queries.
"""
import re
from src.retrieval.query_rewriter import QueryRewriter


class TestQueryRewriterEntities:
    """Tests for entity preservation in query rewriting."""

    def test_self_rag_preserved(self):
        """Self-RAG entity should be preserved when dropped by rewriter."""
        rewriter = QueryRewriter()
        original = "What is Self-RAG?"
        rewritten = "What is retrieval augmented generation?"
        result = rewriter._preserve_entities(original, rewritten)
        assert "Self-RAG" in result

    def test_crag_preserved(self):
        """CRAG entity should be preserved."""
        rewriter = QueryRewriter()
        original = "How does CRAG work?"
        rewritten = "How does corrective retrieval augmented generation work?"
        result = rewriter._preserve_entities(original, rewritten)
        assert "CRAG" in result

    def test_bert_gpt3_preserved(self):
        """BERT and GPT-3 entities should be preserved."""
        rewriter = QueryRewriter()
        original = "Compare BERT and GPT-3"
        rewritten = "Compare transformer models"
        result = rewriter._preserve_entities(original, rewritten)
        assert "BERT" in result
        assert "GPT-3" in result

    def test_flash_attention_preserved(self):
        """FlashAttention entity should be preserved."""
        rewriter = QueryRewriter()
        original = "What techniques does FlashAttention introduce?"
        rewritten = "What techniques improve transformer efficiency?"
        result = rewriter._preserve_entities(original, rewritten)
        assert "FlashAttention" in result

    def test_lora_preserved(self):
        """LoRA entity should be preserved."""
        rewriter = QueryRewriter()
        original = "Explain LoRA fine-tuning"
        rewritten = "Explain low rank adaptation technique"
        result = rewriter._preserve_entities(original, rewritten)
        assert "LoRA" in result

    def test_no_entities_no_change(self):
        """Queries without special entities should not be modified."""
        rewriter = QueryRewriter()
        original = "What is machine learning?"
        rewritten = "Define machine learning concepts"
        result = rewriter._preserve_entities(original, rewritten)
        assert result == rewritten

    def test_entity_already_present_no_duplicate(self):
        """If entity already in rewrite, should not duplicate."""
        rewriter = QueryRewriter()
        original = "How does CRAG work?"
        rewritten = "How does CRAG improve retrieval quality?"
        result = rewriter._preserve_entities(original, rewritten)
        assert result == rewritten
        assert result.count("CRAG") == 1

    def test_multiple_entities(self):
        """Multiple entities should all be preserved."""
        rewriter = QueryRewriter()
        original = "Compare BERT, GPT-3, and Self-RAG"
        rewritten = "Compare language models"
        result = rewriter._preserve_entities(original, rewritten)
        assert "BERT" in result
        assert "GPT-3" in result
        assert "Self-RAG" in result

    def test_entity_extraction_patterns(self):
        """Test the entity extraction regex patterns directly."""
        # These are the patterns used in _preserve_entities
        acronym_pattern = re.compile(r'^[A-Z]{2,}(?:-\d+)?$')
        hyphenated_pattern = re.compile(r'^[A-Z][a-z]+(?:-[A-Z][A-Za-z]*)+$')
        camel_pattern = re.compile(r'^[A-Z][a-z]*(?:[A-Z][a-z]*)+$')

        assert acronym_pattern.match("CRAG")
        assert acronym_pattern.match("BERT")
        assert acronym_pattern.match("GPT-3")
        assert hyphenated_pattern.match("Self-RAG")
        assert hyphenated_pattern.match("Flash-Attention")
        assert camel_pattern.match("FlashAttention")
        assert camel_pattern.match("LoRA")
        assert not camel_pattern.match("What")
        assert not camel_pattern.match("How")