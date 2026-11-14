"""
Retrieval package for AI Research Assistant.
"""
from src.retrieval.hybrid_retriever import HybridRetriever, RetrievalResult, get_retriever
from src.retrieval.grader import DocumentGrader, GradedChunk, get_grader
from src.retrieval.query_rewriter import QueryRewriter, RewrittenQuery, get_rewriter
from src.retrieval.semantic_cache import SemanticCache, CacheEntry, get_cache
from src.retrieval.retrieval_evaluator import RetrievalEvaluator, RetrievalEvaluation, get_evaluator
from src.retrieval.corrective_retriever import CorrectiveRetriever, CorrectiveRetrievalResult, get_corrective_retriever
from src.retrieval.multi_step_retriever import MultiStepRetriever, get_multi_step_retriever

# Lazy import for raptor_retriever to avoid circular import
_RaptorRetriever = None
_get_raptor_retriever_func = None

def _load_raptor_retriever():
    global _RaptorRetriever, _get_raptor_retriever_func
    if _RaptorRetriever is None:
        from src.retrieval.raptor_retriever import RaptorRetriever, get_raptor_retriever
        _RaptorRetriever = RaptorRetriever
        _get_raptor_retriever_func = get_raptor_retriever
    return _RaptorRetriever, _get_raptor_retriever_func

class _LazyRaptorModule:
    def __getattr__(self, name):
        if name in ("RaptorRetriever", "get_raptor_retriever"):
            _RaptorRetriever, _get_raptor_retriever_func = _load_raptor_retriever()
            return _RaptorRetriever if name == "RaptorRetriever" else _get_raptor_retriever_func
        raise AttributeError(f"module 'src.retrieval' has no attribute '{name}'")

import sys as _sys
_module = _sys.modules[__name__]
# Replace module's __getattr__ for lazy loading
_lazy = _LazyRaptorModule()
for _attr in ("RaptorRetriever", "get_raptor_retriever"):
    setattr(_module, _attr, property(lambda self, _attr=_attr: getattr(_lazy, _attr)))

__all__ = [
    "HybridRetriever",
    "RetrievalResult",
    "get_retriever",
    "DocumentGrader",
    "GradedChunk",
    "get_grader",
    "QueryRewriter",
    "RewrittenQuery",
    "get_rewriter",
    "SemanticCache",
    "CacheEntry",
    "get_cache",
    "RetrievalEvaluator",
    "RetrievalEvaluation",
    "get_evaluator",
    "CorrectiveRetriever",
    "CorrectiveRetrievalResult",
    "get_corrective_retriever",
    "MultiStepRetriever",
    "get_multi_step_retriever",
    "RaptorRetriever",
    "get_raptor_retriever",
]