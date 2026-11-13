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
def _get_raptor_retriever():
    from src.retrieval.raptor_retriever import RaptorRetriever, get_raptor_retriever
    return RaptorRetriever, get_raptor_retriever

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

# Make RaptorRetriever and get_raptor_retriever available as attributes
import sys as _sys
_module = _sys.modules[__name__]
_RaptorRetriever, _get_raptor_retriever = _get_raptor_retriever()
setattr(_module, "RaptorRetriever", _RaptorRetriever)
setattr(_module, "get_raptor_retriever", _get_raptor_retriever)