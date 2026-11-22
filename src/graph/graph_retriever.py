"""
Graph Retriever for GraphRAG.

Performs local graph retrieval: entity linking + neighborhood traversal.

Stage A Scope: Local entity graph only. No community detection, no global summaries,
no map-reduce over communities. Entity linking is heuristic (exact/fuzzy/token overlap).
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
from src.graph.graph_store import get_graph_store, GraphStore, _normalize_entity_id
from src.graph.entity_extractor import get_entity_extractor, EntityExtractor
from src.graph.capabilities import describe, log_capabilities

logger = logging.getLogger(__name__)

# Track whether we've logged the Stage A warning for this session
_GRAPHRAG_STAGE_WARNING_LOGGED = False

# Canonical entity ID map for strong alias resolution
_CANONICAL_ALIASES = {
    "self-rag": ["self-rag", "self rag", "self-reflective rag", "self reflective retrieval augmented generation"],
    "crag": ["crag", "corrective rag", "corrective retrieval augmented generation"],
    "raptor": ["raptor", "raptor model", "recursive abstractive processing"],
    "reflection-tokens": ["reflection tokens", "reflection token", "reflective tokens"],
}

# Generic/weak tokens that should be rejected unless exact canonical match
_GENERIC_TOKENS = {"llm", "llms", "model", "models", "method", "methods", "paper", "papers", "study", "work", "approach", "technique", "techniques", "algorithm", "algorithms", "system", "systems", "framework", "frameworks", "architecture", "architectures", "transformer", "transformers", "attention", "bert", "gpt", "t5", "bart"}

# Minimum length for a token to be considered for fuzzy matching
_MIN_TOKEN_LEN = 3

# Maximum number of linked seed entities to return
_MAX_LINKED_SEEDS = 3


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
        global _GRAPHRAG_STAGE_WARNING_LOGGED
        if not _GRAPHRAG_STAGE_WARNING_LOGGED:
            logger.warning(
                "GraphRAG Stage A experimental; no community detection / global summaries; "
                "global theme queries may underperform. %s",
                describe(),
            )
            _GRAPHRAG_STAGE_WARNING_LOGGED = True

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
        Link entities from query using exact and fuzzy matching with precision controls.

        Returns:
            List of linked Entity objects (capped to top 1-3 strongest matches)
        """
        linked = []
        query_lower = query.lower()

        # Step 0: Strong canonical alias force-match (before other matching)
        # These are high-priority known entities that must be linked if present
        canonical_linked: Set[str] = set()
        for canonical_id, aliases in _CANONICAL_ALIASES.items():
            for alias in aliases:
                if alias in query_lower:
                    # Find the entity with this canonical ID from the graph
                    all_entities = self.graph_store.all_entities()
                    for entity in all_entities:
                        if entity.id == canonical_id and canonical_id not in canonical_linked:
                            linked.append(entity)
                            canonical_linked.add(canonical_id)
                            logger.debug(f"Canonical alias force-match: {entity.name} (id={entity.id})")
                            break

        # 1. Exact match on full entity names/aliases (HIGHEST PRECISION)
        exact_matches = []
        for entity in self.graph_store.all_entities():
            if entity.id in canonical_linked:
                continue
            name_lower = entity.name.lower()
            # Check direct name match
            if name_lower in query_lower:
                exact_matches.append(entity)
            # Check alias match
            for alias in entity.aliases:
                if alias.lower() in query_lower and entity not in exact_matches:
                    exact_matches.append(entity)
                    break

        # 2. Strong alias match from canonical alias map (supplementary)
        alias_matches = []
        for canonical_id, aliases in _CANONICAL_ALIASES.items():
            if canonical_id in canonical_linked:
                continue
            for alias in aliases:
                if alias in query_lower:
                    # Find the entity with this canonical ID
                    all_entities = self.graph_store.all_entities()
                    for entity in all_entities:
                        if entity.id == canonical_id and entity not in alias_matches and entity not in exact_matches:
                            alias_matches.append(entity)
                            break

        # 3. Fuzzy match: token overlap with strict thresholds
        fuzzy_matches = []
        query_tokens = set(re.findall(r'\b\w+\b', query_lower))
        # Filter out generic/weak tokens (note: "self-rag" tokens are preserved
        # since "rag" was removed from _GENERIC_TOKENS above)
        query_tokens = {t for t in query_tokens if len(t) >= _MIN_TOKEN_LEN and t not in _GENERIC_TOKENS}

        if query_tokens:
            all_entities = self.graph_store.all_entities()
            for entity in all_entities:
                if entity in exact_matches or entity in alias_matches or entity in fuzzy_matches:
                    continue
                entity_name_lower = entity.name.lower()
                entity_tokens = set(re.findall(r'\b\w+\b', entity_name_lower))
                entity_tokens = {t for t in entity_tokens if len(t) >= _MIN_TOKEN_LEN and t not in _GENERIC_TOKENS}

                # Check aliases too
                alias_tokens = set()
                for alias in entity.aliases:
                    alias_tokens.update(re.findall(r'\b\w+\b', alias.lower()))
                alias_tokens = {t for t in alias_tokens if len(t) >= _MIN_TOKEN_LEN and t not in _GENERIC_TOKENS}
                entity_tokens.update(alias_tokens)

                if not entity_tokens:
                    continue

                overlap = query_tokens & entity_tokens
                if not overlap:
                    continue

                # Strict threshold: require >= 2 token overlap OR single strong token
                overlap_count = len(overlap)
                entity_token_count = len(entity_tokens)

                # Score: overlap ratio * entity specificity
                if entity_token_count > 0:
                    score = overlap_count / entity_token_count
                    # Require strong match
                    if score >= 0.6 or (overlap_count >= 2 and entity_token_count <= 4):
                        fuzzy_matches.append((entity, score))

        # Sort fuzzy by score descending
        fuzzy_matches.sort(key=lambda x: x[1], reverse=True)
        fuzzy_entities = [e for e, s in fuzzy_matches]

        # Combine in priority order: exact -> alias -> fuzzy
        for entity in exact_matches:
            linked.append(entity)
            logger.debug(f"Exact match: {entity.name} (id={entity.id})")
        for entity in alias_matches:
            linked.append(entity)
            logger.debug(f"Alias match: {entity.name} (id={entity.id})")
        for entity in fuzzy_entities:
            linked.append(entity)
            logger.debug(f"Fuzzy match: {entity.name} (id={entity.id})")

        # Cap to max seeds
        if len(linked) > _MAX_LINKED_SEEDS:
            logger.info(f"Capping linked entities from {len(linked)} to {_MAX_LINKED_SEEDS}")
            linked = linked[:_MAX_LINKED_SEEDS]

        return linked

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