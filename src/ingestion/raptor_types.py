"""
Shared RAPTOR types for AI Research Papers.

This module contains types shared between ingestion and retrieval modules
to avoid circular imports.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class RaptorNode:
    """A RAPTOR summary node."""
    chunk_id: str
    paper_id: str
    title: str
    section: str  # may be "Abstract" or section heading
    level: int  # 1 = section-level, 2 = paper-level
    chunk_type: str = "summary"  # always "summary"
    source_chunk_ids: List[str] = field(default_factory=list)
    summary: str = ""
    embedding: Optional[List[float]] = None
    # Metadata for retrieval filtering
    metadata: Dict[str, Any] = field(default_factory=dict)