"""
Graph Store for GraphRAG.

Local knowledge graph persistence using NetworkX with JSON/CSV backup.
"""
import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from dataclasses import asdict

try:
    import networkx as nx
except ImportError:
    nx = None
    logging.warning("NetworkX not installed. Graph store will use fallback mode.")

from src.graph.schema import Entity, Relation, EntityType, RelationType
from src.core.config import config

logger = logging.getLogger(__name__)


class GraphStore:
    """
    Local knowledge graph store.

    Uses NetworkX for in-memory graph operations with JSON persistence.
    """

    def __init__(self, graph_dir: Optional[str] = None):
        """
        Initialize graph store.

        Args:
            graph_dir: Directory for graph persistence (default from config)
        """
        if graph_dir is None:
            graph_dir = config.graph.graph_dir
        self.graph_dir = Path(graph_dir)
        self.graph_dir.mkdir(parents=True, exist_ok=True)

        self.graph_file = self.graph_dir / "graph.json"
        self.processed_file = self.graph_dir / "processed_chunks.json"

        # Initialize graph
        if nx is not None:
            self._graph = nx.MultiDiGraph()
        else:
            self._graph = None

        # Track processed chunks for incremental builds
        self._processed_chunks: Set[str] = set()
        # Track failed chunks for retryable builds
        self._failed_chunks: Set[str] = set()

        # Load existing graph
        self.load()

    def _ensure_graph(self):
        """Ensure graph is initialized."""
        if self._graph is None and nx is not None:
            self._graph = nx.MultiDiGraph()

    def add_entities(self, entities: List[Entity]) -> int:
        """
        Add or update entities in the graph.

        Args:
            entities: List of Entity objects

        Returns:
            Number of entities added/updated
        """
        self._ensure_graph()
        if self._graph is None:
            return 0

        count = 0
        for entity in entities:
            node_data = {
                "name": entity.name,
                "type": entity.type.value if isinstance(entity.type, EntityType) else entity.type,
                "aliases": entity.aliases,
                "paper_ids": entity.paper_ids,
            }
            self._graph.add_node(entity.id, **node_data)
            count += 1

        logger.debug(f"Added/updated {count} entities")
        return count

    def add_relations(self, relations: List[Relation]) -> int:
        """
        Add or update relations in the graph.

        Args:
            relations: List of Relation objects

        Returns:
            Number of relations added/updated
        """
        self._ensure_graph()
        if self._graph is None:
            return 0

        count = 0
        for relation in relations:
            edge_data = {
                "relation_type": relation.relation_type.value
                if isinstance(relation.relation_type, RelationType)
                else relation.relation_type,
                "evidence": relation.evidence,
                "chunk_ids": relation.chunk_ids,
                "paper_id": relation.paper_id,
                "weight": relation.weight,
            }
            self._graph.add_edge(
                relation.source_id,
                relation.target_id,
                key=relation.id,
                **edge_data,
            )
            count += 1

        logger.debug(f"Added/updated {count} relations")
        return count

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get entity by ID."""
        self._ensure_graph()
        if self._graph is None or entity_id not in self._graph:
            return None

        data = self._graph.nodes[entity_id]
        return Entity(
            id=entity_id,
            name=data.get("name", entity_id),
            type=data.get("type", "Other"),
            aliases=data.get("aliases", []),
            paper_ids=data.get("paper_ids", []),
        )

    def search_entities_by_name(
        self, query: str, limit: int = 10
    ) -> List[Entity]:
        """
        Search entities by name (exact or partial match).

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            List of matching entities
        """
        self._ensure_graph()
        if self._graph is None:
            return []

        query_lower = query.lower().strip()
        results = []

        for node_id, data in self._graph.nodes(data=True):
            name = data.get("name", "").lower()
            aliases = [a.lower() for a in data.get("aliases", [])]

            # Exact match on name or aliases
            if query_lower == name or query_lower in aliases:
                results.append(
                    Entity(
                        id=node_id,
                        name=data.get("name", node_id),
                        type=data.get("type", "Other"),
                        aliases=data.get("aliases", []),
                        paper_ids=data.get("paper_ids", []),
                    )
                )
            # Partial match
            elif query_lower in name or any(query_lower in a for a in aliases):
                results.append(
                    Entity(
                        id=node_id,
                        name=data.get("name", node_id),
                        type=data.get("type", "Other"),
                        aliases=data.get("aliases", []),
                        paper_ids=data.get("paper_ids", []),
                    )
                )

            if len(results) >= limit:
                break

        return results

    def neighbors(self, entity_id: str, hops: int = 1) -> Set[str]:
        """
        Get neighboring entity IDs within hops.

        Args:
            entity_id: Starting entity ID
            hops: Number of hops (default 1)

        Returns:
            Set of neighboring entity IDs
        """
        self._ensure_graph()
        if self._graph is None or entity_id not in self._graph:
            return set()

        if hops == 1:
            return set(self._graph.successors(entity_id)) | set(
                self._graph.predecessors(entity_id)
            )

        # Multi-hop BFS
        visited = {entity_id}
        current_level = {entity_id}

        for _ in range(hops):
            next_level = set()
            for node in current_level:
                next_level.update(self._graph.successors(node))
                next_level.update(self._graph.predecessors(node))
            next_level -= visited
            visited.update(next_level)
            current_level = next_level
            if not current_level:
                break

        visited.discard(entity_id)
        return visited

    def relations_of(self, entity_ids: List[str]) -> List[Relation]:
        """
        Get all relations between the given entity IDs.

        Args:
            entity_ids: List of entity IDs

        Returns:
            List of Relation objects
        """
        self._ensure_graph()
        if self._graph is None:
            return []

        entity_set = set(entity_ids)
        relations = []

        for u, v, key, data in self._graph.edges(keys=True, data=True):
            if u in entity_set and v in entity_set:
                relations.append(
                    Relation(
                        id=key,
                        source_id=u,
                        relation_type=data.get("relation_type", "related_to"),
                        target_id=v,
                        evidence=data.get("evidence", ""),
                        chunk_ids=data.get("chunk_ids", []),
                        paper_id=data.get("paper_id", ""),
                        weight=data.get("weight", 1.0),
                    )
                )

        return relations

    def all_entities(self) -> List[Entity]:
        """Get all entities in the graph."""
        self._ensure_graph()
        if self._graph is None:
            return []

        return [
            Entity(
                id=node_id,
                name=data.get("name", node_id),
                type=data.get("type", "Other"),
                aliases=data.get("aliases", []),
                paper_ids=data.get("paper_ids", []),
            )
            for node_id, data in self._graph.nodes(data=True)
        ]

    def all_relations(self) -> List[Relation]:
        """Get all relations in the graph."""
        self._ensure_graph()
        if self._graph is None:
            return []

        relations = []
        for u, v, key, data in self._graph.edges(keys=True, data=True):
            relations.append(
                Relation(
                    id=key,
                    source_id=u,
                    relation_type=data.get("relation_type", "related_to"),
                    target_id=v,
                    evidence=data.get("evidence", ""),
                    chunk_ids=data.get("chunk_ids", []),
                    paper_id=data.get("paper_id", ""),
                    weight=data.get("weight", 1.0),
                )
            )
        return relations

    def stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        self._ensure_graph()
        if self._graph is None:
            return {
                "node_count": 0,
                "edge_count": 0,
                "types": {},
                "relation_types": {},
            }

        node_count = self._graph.number_of_nodes()
        edge_count = self._graph.number_of_edges()

        # Count types
        types = {}
        for _, data in self._graph.nodes(data=True):
            t = data.get("type", "Other")
            types[t] = types.get(t, 0) + 1

        relation_types = {}
        for _, _, data in self._graph.edges(data=True):
            rt = data.get("relation_type", "related_to")
            relation_types[rt] = relation_types.get(rt, 0) + 1

        return {
            "node_count": node_count,
            "edge_count": edge_count,
            "types": types,
            "relation_types": relation_types,
        }

    def is_chunk_processed(self, chunk_id: str) -> bool:
        """Check if a chunk has been processed."""
        return chunk_id in self._processed_chunks

    def mark_chunk_processed(self, chunk_id: str) -> None:
        """Mark a chunk as processed."""
        self._processed_chunks.add(chunk_id)

    def is_chunk_failed(self, chunk_id: str) -> bool:
        """Check if a chunk has failed (retryable)."""
        return chunk_id in self._failed_chunks

    def mark_chunk_failed(self, chunk_id: str) -> None:
        """Mark a chunk as failed (retryable)."""
        self._failed_chunks.add(chunk_id)

    def save(self) -> None:
        """Save graph to disk."""
        self._ensure_graph()

        # Save graph as JSON
        if self._graph is not None and self._graph.number_of_nodes() > 0:
            data = {
                "nodes": [
                    {"id": node_id, **node_data}
                    for node_id, node_data in self._graph.nodes(data=True)
                ],
                "edges": [
                    {
                        "source": u,
                        "target": v,
                        "key": key,
                        **edge_data,
                    }
                    for u, v, key, edge_data in self._graph.edges(keys=True, data=True)
                ],
            }
            with open(self.graph_file, "w") as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved graph to {self.graph_file}: {self._graph.number_of_nodes()} nodes, {self._graph.number_of_edges()} edges")

        # Save processed chunks
        with open(self.processed_file, "w") as f:
            json.dump(list(self._processed_chunks), f, indent=2)
        logger.debug(f"Saved {len(self._processed_chunks)} processed chunk IDs")

    def load(self) -> None:
        """Load graph from disk."""
        self._ensure_graph()

        # Load graph
        if self.graph_file.exists():
            try:
                with open(self.graph_file, "r") as f:
                    data = json.load(f)

                if self._graph is not None:
                    for node in data.get("nodes", []):
                        node_id = node.pop("id")
                        self._graph.add_node(node_id, **node)

                    for edge in data.get("edges", []):
                        source = edge.pop("source")
                        target = edge.pop("target")
                        key = edge.pop("key")
                        self._graph.add_edge(source, target, key=key, **edge)

                logger.info(
                    f"Loaded graph from {self.graph_file}: "
                    f"{self._graph.number_of_nodes()} nodes, {self._graph.number_of_edges()} edges"
                )
            except Exception as e:
                logger.error(f"Failed to load graph: {e}")

        # Load processed chunks
        if self.processed_file.exists():
            try:
                with open(self.processed_file, "r") as f:
                    self._processed_chunks = set(json.load(f))
                logger.debug(f"Loaded {len(self._processed_chunks)} processed chunk IDs")
            except Exception as e:
                logger.warning(f"Failed to load processed chunks: {e}")


def get_graph_store(graph_dir: Optional[str] = None) -> GraphStore:
    """Get a graph store instance."""
    return GraphStore(graph_dir)