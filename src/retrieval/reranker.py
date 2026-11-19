"""
Optional Cross-Encoder Reranker for AI Research Papers.

Provides optional reranking of retrieved chunks using a cross-encoder model.
Disabled by default (RERANK_ENABLED=false) for CPU-safe operation.
"""
import logging
from typing import List, Dict, Any, Optional
from src.core.config import get_config

logger = logging.getLogger(__name__)


class Reranker:
    """
    Cross-encoder reranker for re-ranking retrieved chunks.
    
    Uses sentence-transformers CrossEncoder to score (query, chunk) pairs.
    Disabled by default - only initializes when explicitly enabled.
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        enabled: Optional[bool] = None,
    ):
        """
        Initialize reranker.
        
        Args:
            model_name: Cross-encoder model name (from config if None)
            enabled: Whether reranking is enabled (from config if None)
        """
        cfg = get_config()
        self.model_name = model_name or cfg.retrieval.rerank_model
        self.enabled = enabled if enabled is not None else cfg.retrieval.rerank_enabled
        self._model = None
        self._initialized = False
        
        if self.enabled:
            self._initialize()
    
    def _initialize(self) -> None:
        """Lazy initialize the cross-encoder model."""
        if self._initialized:
            return
        
        try:
            from sentence_transformers import CrossEncoder
            logger.info(f"Loading reranker model: {self.model_name}")
            self._model = CrossEncoder(self.model_name)
            self._initialized = True
            logger.info("Reranker model loaded successfully")
        except ImportError:
            logger.warning("sentence-transformers not installed, reranking disabled")
            self.enabled = False
        except Exception as e:
            logger.warning(f"Failed to load reranker model '{self.model_name}': {e}")
            self.enabled = False
    
    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Rerank chunks using cross-encoder scores.
        
        Args:
            query: The search query
            chunks: List of chunk dicts with 'text' field
            top_k: Number of top results to return (uses config if None)
            
        Returns:
            Reranked chunks with 'rerank_score' field added
        """
        if not self.enabled or not self._initialized or not chunks:
            return chunks[:top_k] if top_k else chunks
        
        try:
            cfg = get_config()
            top_k = top_k or cfg.retrieval.top_k_final
            
            # Prepare query-chunk pairs
            pairs = []
            for chunk in chunks:
                text = chunk.get("text", "")
                if text:
                    pairs.append([query, text])
                else:
                    pairs.append([query, ""])
            
            # Score pairs
            scores = self._model.predict(pairs)
            
            # Add rerank scores to chunks
            for i, chunk in enumerate(chunks):
                chunk["rerank_score"] = float(scores[i])
            
            # Sort by rerank score descending
            reranked = sorted(chunks, key=lambda x: x.get("rerank_score", 0.0), reverse=True)
            
            # Return top_k
            result = reranked[:top_k]
            
            logger.debug(f"Reranked {len(chunks)} chunks, returning top {len(result)}")
            return result
            
        except Exception as e:
            logger.warning(f"Reranking failed: {e}, returning original order")
            return chunks[:top_k] if top_k else chunks


def get_reranker(
    model_name: Optional[str] = None,
    enabled: Optional[bool] = None,
) -> Reranker:
    """Get a reranker instance."""
    return Reranker(model_name, enabled)