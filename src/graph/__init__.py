"""
GraphRAG package for AI Research Assistant.

Stage A Scope (current):
- Entity extraction and linking from paper abstracts/chunks
- Local entity-relation graph stored as NetworkX JSON
- 1-hop neighborhood traversal for query-time retrieval
- Grounded answers with explicit entity/relation evidence

Stage B Non-Goals (not implemented):
- Community detection (Leiden/clustering)
- Community summary generation
- Global map-reduce query answering over communities
- GraphRAG global search / thematic queries

This is a LOCAL GRAPH RAG only. It does not implement Microsoft-style
GraphRAG global reasoning via community hierarchies.
"""
from src.graph.schema import Entity, Relation, GraphEvidence, EntityType, RelationType
from src.graph.entity_extractor import EntityExtractor, get_entity_extractor
from src.graph.graph_store import GraphStore, get_graph_store
from src.graph.graph_retriever import GraphRetriever, get_graph_retriever
from src.graph.capabilities import (
    GRAPHRAG_STAGE,
    HAS_COMMUNITY_DETECTION,
    HAS_GLOBAL_SUMMARY,
    HAS_MAP_REDUCE,
    HAS_COMMUNITY_REPORTS,
    STRATEGY_LOCAL,
    describe,
    log_capabilities,
)

__all__ = [
    "Entity",
    "Relation",
    "GraphEvidence",
    "EntityType",
    "RelationType",
    "EntityExtractor",
    "get_entity_extractor",
    "GraphStore",
    "get_graph_store",
    "GraphRetriever",
    "get_graph_retriever",
    # Capability manifest
    "GRAPHRAG_STAGE",
    "HAS_COMMUNITY_DETECTION",
    "HAS_GLOBAL_SUMMARY",
    "HAS_MAP_REDUCE",
    "HAS_COMMUNITY_REPORTS",
    "STRATEGY_LOCAL",
    "describe",
    "log_capabilities",
]