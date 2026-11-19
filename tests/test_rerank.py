"""
Smoke test for optional cross-encoder reranker.

Verifies:
- Disabled by default (no model load)
- When disabled: returns chunks unchanged, no model load
- When enabled: mock scorer can reorder results
- Failure falls back gracefully
"""
import os
import sys
from unittest.mock import MagicMock, patch

# Mock sentence_transformers BEFORE importing reranker
mock_st = MagicMock()
mock_st.CrossEncoder = MagicMock()
sys.modules['sentence_transformers'] = mock_st

from src.retrieval.reranker import Reranker


class TestReranker:
    """Tests for optional cross-encoder reranker."""

    def setup_method(self):
        """Set up test fixtures."""
        self.test_chunks = [
            {"chunk_id": "c1", "text": "BERT uses masked language modeling", "score": 0.9},
            {"chunk_id": "c2", "text": "GPT-3 uses autoregressive training", "score": 0.8},
            {"chunk_id": "c3", "text": "Transformers use self-attention", "score": 0.7},
        ]

    def test_disabled_by_default(self):
        """Reranker should be disabled by default (RERANK_ENABLED=false)."""
        with patch.dict(os.environ, {"RERANK_ENABLED": "false"}, clear=True):
            from importlib import reload
            import src.core.config as config_module
            reload(config_module)
            from src.core.config import config
            
            reranker = Reranker()
            assert reranker.enabled is False
            assert reranker._model is None

    def test_disabled_returns_chunks_unchanged(self):
        """Disabled reranker should return chunks unchanged without calling model."""
        with patch.dict(os.environ, {"RERANK_ENABLED": "false"}, clear=True):
            from importlib import reload
            import src.core.config as config_module
            reload(config_module)
            from src.core.config import config
            
            reranker = Reranker()
            result = reranker.rerank("test query", self.test_chunks)
            
            # Should return original chunks unchanged (up to top_k)
            assert len(result) == len(self.test_chunks)
            assert result[0]["chunk_id"] == "c1"
            assert result[1]["chunk_id"] == "c2"
            assert result[2]["chunk_id"] == "c3"

    def test_enabled_requires_model_load(self):
        """Enabled reranker should attempt to load model."""
        with patch.dict(os.environ, {"RERANK_ENABLED": "true"}, clear=True):
            from importlib import reload
            import src.core.config as config_module
            reload(config_module)
            from src.core.config import config
            assert config.retrieval.rerank_enabled is True
            
            # Now create reranker - it should read the updated config
            reranker = Reranker()
            assert reranker.enabled is True

    def test_rerank_reorders_chunks(self):
        """Enabled reranker with mocked scorer should reorder chunks by score."""
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.5, 0.3, 0.8]
        
        with patch.dict(os.environ, {"RERANK_ENABLED": "true"}, clear=True):
            from importlib import reload
            import src.core.config as config_module
            reload(config_module)
            from src.core.config import config
            
            reranker = Reranker()
            reranker._model = MagicMock(predict=lambda x: [0.5, 0.3, 0.8])
            reranker._initialized = True
            reranker.enabled = True
            
            result = reranker.rerank("test query", self.test_chunks)
            
            # Should be reordered: c3 (0.8), c1 (0.5), c2 (0.3)
            assert len(result) == 3
            assert result[0]["chunk_id"] == "c3"
            assert result[1]["chunk_id"] == "c1"
            assert result[2]["chunk_id"] == "c2"
            assert "rerank_score" in result[0]
            assert result[0]["rerank_score"] == 0.8

    def test_top_k_limit(self):
        """Reranked results should respect top_k limit."""
        with patch.dict(os.environ, {"RERANK_ENABLED": "true"}, clear=True):
            from importlib import reload
            import src.core.config as config_module
            reload(config_module)
            from src.core.config import config
            
            reranker = Reranker()
            reranker._model = MagicMock(predict=lambda x: [0.5, 0.3, 0.8, 0.9])
            reranker._initialized = True
            reranker.enabled = True
            
            chunks = [
                {"chunk_id": "c1", "text": "text1"},
                {"chunk_id": "c2", "text": "text2"},
                {"chunk_id": "c3", "text": "text3"},
                {"chunk_id": "c4", "text": "text4"},
            ]
            
            result = reranker.rerank("test query", chunks, top_k=2)
            
            assert len(result) == 2

    def test_failure_falls_back_gracefully(self):
        """Reranker failure should fall back to original order without crashing."""
        with patch.dict(os.environ, {"RERANK_ENABLED": "true"}, clear=True):
            from importlib import reload
            import src.core.config as config_module
            reload(config_module)
            from src.core.config import config
            
            reranker = Reranker()
            reranker._model = MagicMock(predict=MagicMock(side_effect=Exception("Model error")))
            reranker._initialized = True
            reranker.enabled = True
            
            result = reranker.rerank("test query", self.test_chunks)
            
            # Should fall back to original order
            assert len(result) == len(self.test_chunks)
            assert result[0]["chunk_id"] == "c1"

    def test_empty_chunks_returns_empty(self):
        """Empty chunk list should return empty list."""
        with patch.dict(os.environ, {"RERANK_ENABLED": "true"}, clear=True):
            from importlib import reload
            import src.core.config as config_module
            reload(config_module)
            from src.core.config import config
            
            reranker = Reranker()
            reranker._initialized = True
            reranker.enabled = True
            
            result = reranker.rerank("test query", [])
            assert result == []

    def test_missing_text_handled(self):
        """Chunks without text field should be handled gracefully."""
        with patch.dict(os.environ, {"RERANK_ENABLED": "true"}, clear=True):
            from importlib import reload
            import src.core.config as config_module
            reload(config_module)
            from src.core.config import config
            
            reranker = Reranker()
            reranker._model = MagicMock(predict=lambda x: [0.5, 0.3])
            reranker._initialized = True
            reranker.enabled = True
            
            chunks = [
                {"chunk_id": "c1", "text": "has text"},
                {"chunk_id": "c2"},  # missing text
            ]
            
            result = reranker.rerank("test query", chunks)
            
            assert len(result) == 2


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])