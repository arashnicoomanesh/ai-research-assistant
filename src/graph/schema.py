"""
GraphRAG Schema for AI Research Papers.

Defines core data models for entities, relations, and graph evidence.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


class EntityType(str, Enum):
    """Entity types for research paper knowledge graph."""
    METHOD = "Method"
    MODEL = "Model"
    DATASET = "Dataset"
    METRIC = "Metric"
    TASK = "Task"
    TECHNIQUE = "Technique"
    PAPER = "Paper"
    OTHER = "Other"


class RelationType(str, Enum):
    """Allowed relation types in the graph."""
    INTRODUCES = "introduces"
    IMPROVES = "improves"
    BASED_ON = "based_on"
    COMPARES_WITH = "compares_with"
    EVALUATES_ON = "evaluates_on"
    REPORTS = "reports"
    USES = "uses"
    ADDRESSES = "addresses"
    RELATED_TO = "related_to"


@dataclass
class Entity:
    """An entity in the knowledge graph."""
    id: str
    name: str
    type: EntityType
    aliases: List[str] = field(default_factory=list)
    paper_ids: List[str] = field(default_factory=list)

    def __post_init__(self):
        if isinstance(self.type, str):
            try:
                self.type = EntityType(self.type)
            except ValueError:
                self.type = EntityType.OTHER

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type.value,
            "aliases": self.aliases,
            "paper_ids": self.paper_ids,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Entity":
        return cls(
            id=data["id"],
            name=data["name"],
            type=data["type"],
            aliases=data.get("aliases", []),
            paper_ids=data.get("paper_ids", []),
        )


@dataclass
class Relation:
    """A relation between two entities."""
    id: str
    source_id: str
    relation_type: RelationType
    target_id: str
    evidence: str
    chunk_ids: List[str] = field(default_factory=list)
    paper_id: str = ""
    weight: float = 1.0

    def __post_init__(self):
        if isinstance(self.relation_type, str):
            try:
                self.relation_type = RelationType(self.relation_type)
            except ValueError:
                self.relation_type = RelationType.RELATED_TO

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "source_id": self.source_id,
            "relation_type": self.relation_type.value,
            "target_id": self.target_id,
            "evidence": self.evidence,
            "chunk_ids": self.chunk_ids,
            "paper_id": self.paper_id,
            "weight": self.weight,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Relation":
        return cls(
            id=data["id"],
            source_id=data["source_id"],
            relation_type=data["relation_type"],
            target_id=data["target_id"],
            evidence=data["evidence"],
            chunk_ids=data.get("chunk_ids", []),
            paper_id=data.get("paper_id", ""),
            weight=data.get("weight", 1.0),
        )


@dataclass
class GraphEvidence:
    """Evidence collected from graph retrieval."""
    entities: List[Entity] = field(default_factory=list)
    relations: List[Relation] = field(default_factory=list)
    supporting_chunks: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_context_string(self) -> str:
        """Render compact graph context for LLM."""
        if not self.entities and not self.relations:
            return "[Graph Evidence]\nNo graph evidence found."

        lines = ["[Graph Evidence]"]

        # Group relations by source for cleaner rendering
        for rel in self.relations:
            source_name = rel.source_id
            target_name = rel.target_id
            # Try to find display names
            for e in self.entities:
                if e.id == rel.source_id:
                    source_name = e.name
                if e.id == rel.target_id:
                    target_name = e.name
            lines.append(
                f"- {source_name} -[{rel.relation_type.value}]-> {target_name}: {rel.evidence}"
            )

        if self.supporting_chunks:
            lines.append("\n[Supporting Chunks]")
            for i, chunk in enumerate(self.supporting_chunks, 1):
                title = chunk.get("title", "Unknown")
                section = chunk.get("section", "")
                text = chunk.get("text", "")[:150]
                lines.append(f"  {i}. [{title}] {section}: {text}...")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entities": [e.to_dict() for e in self.entities],
            "relations": [r.to_dict() for r in self.relations],
            "supporting_chunks": self.supporting_chunks,
            "metadata": self.metadata,
        }