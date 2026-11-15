"""
Graph Retriever for GraphRAG.

Performs local graph retrieval: entity linking + neighborhood traversal.
"""
import logging
import re
from typing import List, Dict, Any, Optional, Set, Tuple
from src.core.config import config
from src.core.embeddings import OllamaEmbeddings, get_embeddings
from src.retrieval.hybrid_retriever import get_retriever
from src.retrieval.qdrant_search import qdrant_search
from qdrant_client.models import Filter, FieldCondition, MatchValue
from src.graph.schema import Entity, Relation, GraphEvidence
from src.graph.graph_store import get_graph_store, GraphStore
from src.graph.entity_extractor import get_entity_extractor, EntityExtractor

logger = logging.getLogger(__name__)


class GraphRetriever:
    """
    Local GraphRAG retrieval.

    Flow:
    1. Entity linking from query (exact + fuzzy match ALL entities)
    2. Expand neighborhood (hops=1 or 2 for each linked entity)
    3. Collect relations among selected nodes (direct + shared-neighbor)
    4. Optionally fetch supporting vector chunks
    5. Return GraphEvidence with explicit grounding
    """

    def __init__(
        self,
        graph_store: Optional[GraphStore] = None,
        entity_extractor: Optional[EntityExtractor] = None,
        embeddings: Optional[OllamaEmbeddings] = None,
        retriever: Optional[Any] = None,
        hops: int = 1,
        supporting_chunks: int = 3,
    ):
        """
        Initialize graph retriever.

        Args:
            graph_store: GraphStore instance
            entity_extractor: EntityExtractor instance
            embeddings: Embeddings client
            retriever: HybridRetriever for vector search
            hops: Neighborhood expansion hops (1 or 2)
            supporting_chunks: Number of supporting vector chunks to fetch
        """
        self.graph_store = graph_store or get_graph_store()
        self.entity_extractor = entity_extractor or get_entity_extractor()
        self.embeddings = embeddings or get_embeddings()
        self.retriever = retriever or get_retriever()
        self.hops = hops
        self.supporting_chunks = supporting_chunks

    def retrieve(self, query: str, paper_id: Optional[str] = None) -> GraphEvidence:
        """
        Perform graph retrieval for a query.

        Args:
            query: User query
            paper_id: Optional paper filter

        Returns:
            GraphEvidence with entities, relations, and supporting chunks
        """
        logger.info(f"Graph retrieval: query='{query}', paper_id={paper_id}, hops={self.hops}")

        # Step 1: Entity linking from query - link ALL named entities
        linked_entities = self._link_entities(query)
        logger.info(f"Linked entities: {[e.name for e in linked_entities]}")

        if not linked_entities:
            logger.warning("No entities linked from query")
            return GraphEvidence(metadata={"fallback": True, "reason": "no_entities_linked"})

        # Step 2: Expand neighborhood for EACH linked entity
        all_entity_ids = set()
        for e in linked_entities:
            neighbors = self.graph_store.neighbors(e.id, hops=self.hops)
            all_entity_ids.add(e.id)
            all_entity_ids.update(neighbors)
        logger.info(f"Expanded to {len(all_entity_ids)} entities (hops={self.hops})")

        # Step 3: Get entities and relations
        entities = []
        for eid in all_entity_ids:
            entity = self.graph_store.get_entity(eid)
            if entity:
                entities.append(entity)

        relations = self.graph_store.relations_of(list(all_entity_ids))
        logger.info(f"Found {len(relations)} relations among selected entities")

        # Step 4: Fetch supporting vector chunks
        supporting_chunks = self._fetch_supporting_chunks(query, linked_entities, paper_id)

        # Step 5: Identify direct edges between originally linked entities
        direct_edges = self._find_direct_edges(linked_entities, relations)

        evidence = GraphEvidence(
            entities=entities,
            relations=relations,
            supporting_chunks=supporting_chunks,
            metadata={
                "linked_entity_ids": [e.id for e in linked_entities],
                "expanded_entity_count": len(all_entity_ids),
                "relation_count": len(relations),
                "direct_edges": len(direct_edges),
                "supporting_chunk_count": len(supporting_chunks),
                "hops": self.hops,
                "paper_id": paper_id,
            },
        )

        logger.info(
            f"Graph retrieval complete: {len(entities)} entities, "
            f"{len(relations)} relations, {len(supporting_chunks)} supporting chunks, "
            f"{len(direct_edges)} direct edges"
        )

        return evidence

    def _link_entities(self, query: str) -> List[Entity]:
        """
        Link entities from query using exact and fuzzy matching.
        Links ALL named entities found, not just the first one.

        Args:
            query: User query

        Returns:
            List of linked Entity objects (may be 1 or more)
        """
        linked = []
        query_lower = query.lower()

        # Get all entities from graph
        all_entities = self.graph_store.all_entities()

        # Build name -> entity mapping for exact match
        name_to_entity = {}
        for entity in all_entities:
            name_to_entity[entity.name.lower()] = entity
            for alias in entity.aliases:
                name_to_entity[alias.lower()] = entity

        # 1. Exact match on full entity names/aliases
        for name, entity in name_to_entity.items():
            if name in query_lower and entity not in linked:
                linked.append(entity)

        # 2. Fuzzy match: check if query contains entity name or vice versa
        if not linked:
            for entity in all_entities:
                entity_name_lower = entity.name.lower()
                # Check if entity name is substring of query or query is substring of entity
                if entity_name_lower in query_lower or query_lower in entity_name_lower:
                    if entity not in linked:
                        linked.append(entity)
                        continue

                # Check aliases
                for alias in entity.aliases:
                    alias_lower = alias.lower()
                    if alias_lower in query_lower or query_lower in alias_lower:
                        if entity not in linked:
                            linked.append(entity)
                            break

        # 3. Token-based matching for compound names
        if not linked:
            query_tokens = set(re.findall(r'\b\w+\b', query_lower))
            for entity in all_entities:
                entity_tokens = set(re.findall(r'\b\w+\b', entity.name.lower()))
                # Check overlap
                if query_tokens & entity_tokens:
                    overlap = len(query_tokens & entity_tokens)
                    if overlap >= 2 or (overlap == 1 and len(entity_tokens) <= 3):
                        if entity not in linked:
                            linked.append(entity)

        # Limit but don't cap too aggressively - try to get all relevant entities
        return linked[:8]

    def _expand_neighborhood(self, seed_entity_ids: List[str]) -> Set[str]:
        """
        Expand entity set by graph neighborhood.

        Args:
            seed_entity_ids: Initial entity IDs

        Returns:
            Expanded set of entity IDs
        """
        expanded = set()

        for entity_id in seed_entity_ids:
            neighbors = self.graph_store.neighbors(entity_id, hops=self.hops)
            expanded.update(neighbors)

        return expanded

    def _find_direct_edges(
        self, linked_entities: List[Entity], relations: List[Relation]
    ) -> List[Dict[str, Any]]:
        """
        Find direct edges between originally linked entities.

        Args:
            linked_entities: The original entities linked from the query
            relations: All relations in the expanded subgraph

        Returns:
            List of direct edge descriptions
        """
        linked_ids = {e.id for e in linked_entities}
        direct = []

        for r in relations:
            if r.source_id in linked_ids and r.target_id in linked_ids:
                direct.append(
                    {
                        "source": r.source_id,
                        "target": r.target_id,
                        "relation_type": r.relation_type.value,
                        "evidence": r.evidence,
                    }
                )

        return direct

    def _fetch_supporting_chunks(
        self,
        query: str,
        linked_entities: List[Entity],
        paper_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Fetch supporting vector chunks for context.

        Args:
            query: Original query
            linked_entities: Linked entities
            paper_id: Optional paper filter

        Returns:
            List of chunk payloads
        """
        try:
            # Build filter for vector search
            must_conditions = []

            # Use MatchValue for single paper_id (compatible with this qdrant-client version)
            if paper_id:
                must_conditions.append(
                    FieldCondition(key="paper_id", match=MatchValue(value=paper_id))
                )
            # If multiple entity paper_ids but no single paper_id, skip paper filter
            # and rely on vector search + entity names in results

            query_filter = Filter(must=must_conditions) if must_conditions else None

            query_vector = self.embeddings.embed_query(query)

            results = qdrant_search(
                client=self.retriever.client,
                collection_name=config.qdrant.collection,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=self.supporting_chunks,
                with_payload=True,
                with_vectors=False,
            )

            chunks = [{"score": hit.score, **hit.payload} for hit in results]
            return chunks

        except Exception as e:
            logger.warning(f"Failed to fetch supporting chunks: {e}")
            return []


def get_graph_retriever(
    graph_store: Optional[GraphStore] = None,
    entity_extractor: Optional[EntityExtractor] = None,
    embeddings: Optional[OllamaEmbeddings] = None,
    retriever: Optional[Any] = None,
    hops: Optional[int] = None,
    supporting_chunks: Optional[int] = None,
) -> GraphRetriever:
    """Get a graph retriever instance."""
    if hops is None:
        hops = config.graph.hops
    if supporting_chunks is None:
        supporting_chunks = config.graph.supporting_chunks

    return GraphRetriever(
        graph_store=graph_store,
        entity_extractor=entity_extractor,
        embeddings=embeddings,
        retriever=retriever,
        hops=hops,
        supporting_chunks=supporting_chunks,
    )