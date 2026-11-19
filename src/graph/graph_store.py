"""Rewrite graph_store.py with proper class structure."""
import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
from src.core.config import config
try:
    import networkx as nx
except ImportError:
    nx = None
from src.graph.schema import Entity, Relation, EntityType, RelationType

# Entity ID normalization (mirrors entity_extractor._normalize_entity_id)
# Strongly normalizes entity name to a canonical ID.
_steps = [
    lambda n: n.lower().strip(),
    lambda n: re.sub(r'[\s\-]+', '-', n),
    lambda n: re.sub(r'[^a-z0-9-]', '', n),
    lambda n: re.sub(r'-+', '-', n),
    lambda n: n.strip('-'),
    lambda n: n[:64].strip('-') if len(n) > 64 else n,
    lambda n: re.sub(r'[^a-z0-9]', '', n)[:32].strip('-') if not n else n,
]
_CANONICAL_NAME_MAP = {
    # Self-RAG variants → self-rag
    "self-reflective-retrieval-augmented-generation": "self-rag",
    "self-reflectiveretrievalaugmentedgeneration": "self-rag",
    # reflection tokens variants → reflection-tokens
    "reflectiontokens": "reflection-tokens",
    # CRAG variants → crag
    "corrective-rag": "crag",
    # RAPTOR variants → raptor
    "raptor-model": "raptor",
    "raptor_base": "raptor",
    "raptor-large": "raptor",
}


def _normalize_entity_id(name: str) -> str:
    """Normalize entity name to canonical ID."""
    normalized = name
    for step in _steps:
        normalized = step(normalized)
    return normalized


def _is_canonical_match(entity_name: str, existing_name: str) -> bool:
    """Check if two entity names match via normalized ID or canonical map."""
    entity_norm = _normalize_entity_id(entity_name)
    existing_norm = _normalize_entity_id(existing_name)
    if entity_norm == existing_norm:
        return True
    # Check canonical name map (normalize both sides for comparison)
    # Normalize the entity names and check against map keys/values
    entity_lower_norm = _normalize_entity_id(entity_name.lower())
    existing_lower_norm = _normalize_entity_id(existing_name.lower())
    for map_key, map_value in _CANONICAL_NAME_MAP.items():
        map_key_norm = _normalize_entity_id(map_key.lower())
        map_value_norm = _normalize_entity_id(map_value.lower())
        # entity variant -> canonical
        if entity_lower_norm == map_key_norm:
            return map_value_norm == existing_norm
        # existing -> canonical via map
        if existing_lower_norm == map_key_norm:
            return entity_norm == map_value_norm
    return False

logger = logging.getLogger(__name__)


class GraphStore:
    """Local knowledge graph store.

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
            logger.info(
                f"Saved graph to {self.graph_file}: "
                f"{self._graph.number_of_nodes()} nodes, {self._graph.number_of_edges()} edges"
            )

        # Save processed chunks
        with open(self.processed_file, "w") as f:
            json.dump(list(self._processed_chunks), f, indent=2)
        logger.debug(f"Saved {len(self._processed_chunks)} processed chunk IDs")

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

    def add_entities(self, entities: List[Entity]) -> int:
        """
        Add or update entities in the graph.

        If an entity with the same normalized ID already exists,
        merge aliases and paper_ids instead of creating a duplicate.

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
            normalized_id = entity.id

            # Check if entity already exists by looking for a node with matching alias or name
            existing_id = None
            for node_id in self._graph.nodes():
                node_data = self._graph.nodes[node_id]
                # Check if this node's name or aliases match
                if node_data.get("name") == entity.name:
                    existing_id = node_id
                    break
                # Check aliases (case-insensitive)
                if entity.name in [a.lower() for a in node_data.get("aliases", [])]:
                    existing_id = node_id
                    break
                # Check normalized ID match or canonical name map match
                if existing_id is None:
                    if _is_canonical_match(entity.name, node_data.get("name", "")):
                        existing_id = node_id
                        break

            if existing_id:
                # Merge: update aliases and paper_ids
                existing_data = self._graph.nodes[existing_id]
                existing_aliases = existing_data.get("aliases", [])
                existing_paper_ids = set(existing_data.get("paper_ids", []))

                # Add new aliases (case-insensitive dedup)
                for alias in entity.aliases:
                    alias_lower = alias.lower()
                    if alias_lower not in [a.lower() for a in existing_aliases]:
                        existing_aliases.append(alias)

                # Union paper_ids
                for pid in entity.paper_ids:
                    existing_paper_ids.add(pid)

                # Update the node using add_node (NodeView does not support item assignment)
                self._graph.add_node(
                    existing_id,
                    name=entity.name,
                    type=entity.type.value if isinstance(entity.type, EntityType) else entity.type,
                    aliases=existing_aliases,
                    paper_ids=list(existing_paper_ids),
                )
                count += 1
            else:
                # New entity - add as normal
                node_data = {
                    "name": entity.name,
                    "type": entity.type.value if isinstance(entity.type, EntityType) else entity.type,
                    "aliases": entity.aliases,
                    "paper_ids": entity.paper_ids,
                }
                self._graph.add_node(entity.id, **node_data)
                count += 1

        logger.debug(f"Added/updated {count} entities (with merge)")
        return count

    def dedupe(self) -> None:
        """
        Deduplicate entities in the graph by canonical alias map.

        Merges duplicate entities by:
        - Grouping entities by normalized canonical ID
        - Merging aliases and paper_ids across duplicates
        - Rewiring edges to use canonical entity IDs
        - Removing duplicate nodes

        No LLM calls or Qdrant extraction. Works on existing graph.json only.
        """
        self._ensure_graph()
        if self._graph is None:
            return

        import time as _time
        t0 = _time.time()

        # Step 1: Collect all node data
        node_ids = list(self._graph.nodes())
        node_data = {}
        for nid in node_ids:
            node_data[nid] = self._graph.nodes[nid]

        # Step 2: Find merge pairs using canonical match
        # Track which node absorbs which other nodes
        canonical_to_absorbed: Dict[str, Set[str]] = {}
        absorbed_to_canonical: Dict[str, str] = {}
        nodes_merged = 0

        for i_id in node_ids:
            if i_id in absorbed_to_canonical:
                continue
            i_name = node_data[i_id].get("name", "")
            for j_id in node_ids:
                if j_id in absorbed_to_canonical or j_id == i_id:
                    continue
                j_name = node_data[j_id].get("name", "")

                # Check if these two nodes should be merged
                if _is_canonical_match(i_name, j_name):
                    # Merge j_id into i_id (i_id becomes canonical)
                    absorbed_to_canonical[j_id] = i_id
                    if i_id not in canonical_to_absorbed:
                        canonical_to_absorbed[i_id] = set()
                    canonical_to_absorbed[i_id].add(j_id)
                    nodes_merged += 1

        # Step 3: Merge absorbed nodes into their canonical nodes
        for canonical_id, absorbed_set in canonical_to_absorbed.items():
            if not absorbed_set:
                continue

            canonical_data = node_data[canonical_id]
            merged_aliases = set(canonical_data.get("aliases", []))
            merged_paper_ids = set(canonical_data.get("paper_ids", []))

            # Merge data from all absorbed nodes
            for absorbed_id in absorbed_set:
                absorbed_data = node_data[absorbed_id]
                merged_aliases.update(absorbed_data.get("aliases", []))
                merged_paper_ids.update(absorbed_data.get("paper_ids", []))

                # Rewire edges from absorbed_id to canonical_id
                self._rewire_edges(absorbed_id, canonical_id)

                # Remove the absorbed node
                self._graph.remove_node(absorbed_id)

            # Update the canonical node with merged data
            self._graph.add_node(
                canonical_id,
                name=canonical_data.get("name", ""),
                type=canonical_data.get("type", "Other"),
                aliases=sorted(merged_aliases),
                paper_ids=sorted(merged_paper_ids),
            )

        self.save()

        t1 = _time.time()
        logger.info(f"Deduplication complete in {t1-t0:.1f}s: {nodes_merged} nodes merged")

    def _rewire_edges(self, old_id: str, new_id: str) -> None:
        """
        Rewire all edges incident to old_id to use new_id instead.

        For each edge where old_id is source or target,
        remove the old edge and add a new edge with new_id.
        """
        # Edges where old_id is source: (old_id, v, key, data)
        for u, v, key, data in list(self._graph.edges(old_id, data=True, keys=True)):
            if u == old_id:
                self._graph.remove_edge(u, v, key)
                self._graph.add_edge(new_id, v, key, **data)

        # Edges where old_id is target: iterate all edges and check target
        for u, v, key, data in list(self._graph.edges(data=True, keys=True)):
            if v == old_id:
                self._graph.remove_edge(u, v, key)
                self._graph.add_edge(u, new_id, key, **data)

    def add_relations(self, relations: List[Relation]) -> int:
        """
        Add or update relations in the graph.

        Both source and target entities must exist in the graph.
        Relations with missing endpoints are rejected (not stored).

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
            # Check both endpoints exist
            if relation.source_id not in self._graph:
                logger.warning(f"Skipping relation {relation.id}: source '{relation.source_id}' not in graph")
                continue
            if relation.target_id not in self._graph:
                logger.warning(f"Skipping relation {relation.id}: target '{relation.target_id}' not in graph")
                continue

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
def get_graph_store(graph_dir: Optional[str] = None) -> GraphStore:
    """Get a graph store instance."""
    return GraphStore(graph_dir)

def get_graph_store(graph_dir: Optional[str] = None) -> GraphStore:
    """Get a graph store instance."""
    return GraphStore(graph_dir)
