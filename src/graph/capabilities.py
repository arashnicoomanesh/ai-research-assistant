"""
GraphRAG Capability Manifest.

This module declares the current GraphRAG implementation stage and
which Microsoft-style GraphRAG features are NOT implemented.

Stage A (current): Local entity graph only
- Entity extraction and linking
- 1-hop neighborhood traversal
- Grounded answers with supporting chunks

Stage B (not implemented): Community detection + global summaries
- Leiden community detection
- Community report generation
- Global map-reduce query answering
"""

# GraphRAG implementation stage
GRAPHRAG_STAGE = "A"

# Stage B features - explicitly False for Stage A
HAS_COMMUNITY_DETECTION = False
HAS_GLOBAL_SUMMARY = False
HAS_MAP_REDUCE = False
HAS_COMMUNITY_REPORTS = False

# Strategy identifier used in router
STRATEGY_LOCAL = "graph_rag"

# Human-readable description for logs
_STAGE_A_DESCRIPTION = (
    "GraphRAG Stage A (local entity graph): entity linking + 1-hop neighborhood traversal. "
    "No community detection, no global summaries, no map-reduce over communities."
)


def describe() -> str:
    """Return a one-line description of current GraphRAG capabilities for logging."""
    return _STAGE_A_DESCRIPTION


def log_capabilities() -> None:
    """Log the capability manifest at INFO level."""
    import logging
    logger = logging.getLogger(__name__)
    logger.info("GraphRAG Capability Manifest:")
    logger.info("  Stage: %s", GRAPHRAG_STAGE)
    logger.info("  Strategy: %s", STRATEGY_LOCAL)
    logger.info("  Community Detection: %s", HAS_COMMUNITY_DETECTION)
    logger.info("  Global Summaries: %s", HAS_GLOBAL_SUMMARY)
    logger.info("  Map-Reduce: %s", HAS_MAP_REDUCE)
    logger.info("  Community Reports: %s", HAS_COMMUNITY_REPORTS)
    logger.warning(
        "Stage A experimental; no community detection / global summaries; "
        "global theme queries may underperform."
    )