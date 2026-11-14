"""
RAPTOR Builder for AI Research Papers.

Builds a lightweight hierarchical summary layer from existing child chunks.
Groups chunks by paper_id + section, generates concise summaries via LLM,
and upserts summary nodes into Qdrant for enhanced global/survey retrieval.

Constraints:
- Fully local, CPU-friendly
- Incremental (skip already-indexed groups)
- Small batch sizes
- No LangChain / LlamaIndex
"""
import json
import logging
import uuid
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, field
import numpy as np
from src.core.config import config
from src.core.llm import OllamaLLM, get_llm
from src.core.embeddings import OllamaEmbeddings, get_embeddings
from src.ingestion.indexer import get_indexer, IndexStats
from src.ingestion.raptor_types import RaptorNode
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue

logger = logging.getLogger(__name__)


class RaptorBuilder:
    """
    Builds a lightweight RAPTOR summary layer from existing child chunks.

    Strategy:
    1. Group child chunks by (paper_id, section)
    2. For each group, generate a concise summary via LLM
    3. Embed the summary and upsert to Qdrant as a summary node
    4. Support incremental building - skip groups already having summaries
    """

    def __init__(
        self,
        client: Optional[Any] = None,
        embeddings: Optional[OllamaEmbeddings] = None,
        llm: Optional[OllamaLLM] = None,
        batch_size: int = 16,  # Small batch for CPU
        max_chunks_per_group: int = 20,  # Max chunks to summarize per group
    ):
        """
        Initialize RAPTOR builder.

        Args:
            client: Qdrant client (creates default if None)
            embeddings: Embeddings client (creates default if None)
            llm: LLM client (creates default if None)
            batch_size: Number of nodes to upsert per batch
            max_chunks_per_group: Max child chunks to include per summary group
        """
        self.client = client or get_indexer().client
        self.embeddings = embeddings or get_embeddings()
        self.llm = llm or get_llm()
        self.batch_size = batch_size
        self.max_chunks_per_group = max_chunks_per_group

        # Qdrant payload field schemas (must match what indexer uses)
        self._payload_fields = {
            "chunk_id": "keyword",
            "paper_id": "keyword",
            "title": "keyword",
            "section": "keyword",
            "chunk_type": "keyword",
            "level": "integer",
            "source_chunk_ids": "keyword",
            "summary": "text",
        }

    def _group_chunks_by_section(self, chunks: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Group child chunks by (paper_id, section).

        Args:
            chunks: List of chunk payloads from Qdrant

        Returns:
            Dict mapping (paper_id, section) -> list of chunks
        """
        groups: Dict[str, List[Dict[str, Any]]] = {}

        for chunk in chunks:
            paper_id = chunk.get("paper_id", "")
            section = chunk.get("section", "Abstract")

            # Create group key: paper_id|section
            key = f"{paper_id}|{section}" if paper_id else section

            if key not in groups:
                groups[key] = []
            groups[key].append(chunk)

        logger.debug(f"Grouped {len(chunks)} chunks into {len(groups)} section groups")
        return groups

    def _needs_summary(self, group_key: str, existing_summary_ids: set) -> bool:
        """
        Check if a group already has a summary upserted.

        Args:
            group_key: The group key (paper_id|section)
            existing_summary_ids: Set of chunk_ids that are summaries

        Returns:
            True if this group needs a new summary
        """
        # Extract paper_id from group key
        parts = group_key.split("|", 1)
        paper_id = parts[0] if len(parts) > 1 else None
        section = parts[-1]

        # Look for existing summary chunk_ids in this group
        # Summary chunk_ids follow pattern: {paper_id}_summary_{section_key}
        if paper_id:
            summary_id_pattern = f"{paper_id}_summary"
            # Check if any chunk in this group is already a summary
            # We'll check the existing_summary_ids set
            for sid in existing_summary_ids:
                if sid.startswith(summary_id_pattern) and paper_id in sid:
                    return False
        return True

    def _generate_summary(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Generate a concise summary of child chunks using the LLM.

        Args:
            chunks: Child chunks to summarize (limited to max_chunks_per_group)

        Returns:
            Concise summary text
        """
        # Limit chunks for prompt
        limited = chunks[: self.max_chunks_per_group]

        # Build context summary
        chunks_text = ""
        for i, chunk in enumerate(limited):
            text = chunk.get("text", "")[:500]  # Limit per chunk
            title = chunk.get("title", "Unknown Paper")
            section = chunk.get("section", "")
            chunks_text += f"\n--- Chunk {i+1} ({section}) ---\n{text}\n"

        prompt = f"""Summarize the following text from a research paper into a concise paragraph (2-3 sentences).
Focus on the main contribution, method, or key finding. Do not include references to page numbers or chunk IDs.

Query context:
{chunks_text}

Summary:"""

        try:
            response = self.llm.generate(
                prompt=prompt,
                system_prompt="You are a research assistant creating concise paper summaries.",
                temperature=0.0,
                max_tokens=256,
            )
            summary = response.strip()
            # Ensure it's not too long
            if len(summary) > 500:
                summary = summary[:500] + "..."
            return summary
        except Exception as e:
            logger.warning(f"Summary generation failed: {e}")
            # Fallback: create a simple summary from first chunk
            first_text = limited[0].get("text", "")[:200] if limited else ""
            return first_text or "No summary available"

    def _create_summary_node(
        self,
        group_key: str,
        chunks: List[Dict[str, Any]],
        existing_summaries: List[RaptorNode] = None,
    ) -> Optional[RaptorNode]:
        """
        Create a RaptorNode (summary) for a group of child chunks.

        Args:
            group_key: The group key (paper_id|section)
            chunks: Child chunks in this group
            existing_summaries: Existing summaries to avoid duplicates

        Returns:
            RaptorNode or None if group already has a summary
        """
        # Parse group key
        parts = group_key.split("|", 1)
        paper_id = parts[0] if len(parts) > 1 else None
        section = parts[-1] if len(parts) > 1 else group_key

        # Check if summary already exists for this group
        if existing_summaries:
            for existing in existing_summaries:
                if existing.paper_id == paper_id and existing.section == section:
                    logger.debug(f"Summary already exists for {paper_id} [{section}]")
                    return None

        # Generate summary
        summary_text = self._generate_summary(chunks)

        # Determine level:
        # - If section is "Abstract", level = 2 (paper-level)
        # - Otherwise, level = 1 (section-level)
        level = 2 if section.lower() == "abstract" else 1

        # Create source_chunk_ids from the child chunks
        source_chunk_ids = [c.get("chunk_id", "") for c in chunks[:self.max_chunks_per_group]]

        # Create the summary node
        node_id = f"{paper_id}_summary_{section[:30].replace(' ', '_')}" if paper_id else f"summary_{section[:30].replace(' ', '_')}"

        node = RaptorNode(
            chunk_id=node_id,
            paper_id=paper_id or "unknown",
            title=chunks[0].get("title", "Unknown") if chunks else "Unknown",
            section=section,
            level=level,
            chunk_type="summary",
            source_chunk_ids=source_chunk_ids,
            summary=summary_text,
            metadata={
                "num_source_chunks": min(len(chunks), self.max_chunks_per_group),
                "group_key": group_key,
                "created_by": "raptor_builder",
            },
        )

        return node

    def _embed_node(self, node: RaptorNode) -> RaptorNode:
        """
        Generate embedding for the summary node.

        Args:
            node: RaptorNode to embed

        Returns:
            RaptorNode with embedding set
        """
        try:
            # Embed the summary text
            embedding = self.embeddings.embed_query(node.summary)
            node.embedding = embedding
        except Exception as e:
            logger.warning(f"Embedding generation failed for {node.chunk_id}: {e}")
            # Generate a dummy embedding of the expected dimension
            node.embedding = [0.0] * 768  # nomic-embed-text dimension

        return node

    def _upsert_batch(self, nodes: List[RaptorNode]) -> None:
        """
        Upsert a batch of summary nodes into Qdrant.

        Args:
            nodes: List of RaptorNode to upsert
        """
        if not nodes:
            return

        points = []

        for node in nodes:
            if node.embedding is None:
                self._embed_node(node)

            # Generate deterministic UUID for Qdrant point ID
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, node.chunk_id))

            # Build payload with required fields
            payload = {
                "chunk_id": node.chunk_id,  # Human-readable ID in payload
                "paper_id": node.paper_id,
                "title": node.title,
                "section": node.section,
                "chunk_type": node.chunk_type,
                "level": node.level,
                "source_chunk_ids": node.source_chunk_ids,
                "summary": node.summary,
                "metadata": node.metadata,
                "token_count": node.metadata.get("num_source_chunks", 0),  # Optional token count proxy
            }

            point = PointStruct(
                id=point_id,
                vector=node.embedding,
                payload=payload,
            )

            points.append(point)

        # Upsert in batches
        try:
            for i in range(0, len(points), self.batch_size):
                batch = points[i:i + self.batch_size]
                self.client.upsert(
                    collection_name=config.qdrant.collection,
                    points=batch,
                    wait=True,
                )

            logger.info(f"Upserted {len(nodes)} RAPTOR summary nodes to Qdrant")

        except Exception as e:
            logger.error(f"Failed to upsert RAPTOR summaries: {e}")
            raise

    def build(
        self,
        chunks: Optional[List[Dict[str, Any]]] = None,
        paper_id: Optional[str] = None,
        limit_groups: Optional[int] = None,
    ) -> IndexStats:
        """
        Build RAPTOR summaries from child chunks.

        Args:
            chunks: List of chunk payloads (if None, retrieves from Qdrant)
            paper_id: Optional filter to build summaries for one paper only
            limit_groups: Optional limit on number of groups to process (for testing)

        Returns:
            IndexStats with operation results
        """
        # Step 1: Retrieve chunks if not provided
        if chunks is None:
            indexer = get_indexer()
            # Filter by paper_id if specified
            must_conditions = []
            if paper_id:
                must_conditions.append(FieldCondition(key="paper_id", match=MatchValue(value=paper_id)))

            if must_conditions:
                filter_obj = Filter(must=must_conditions)
                results = indexer.client.scroll(
                    collection_name=config.qdrant.collection,
                    scroll_filter=filter_obj,
                    limit=10000,
                    with_payload=True,
                    with_vectors=False,
                )
                chunks = [point.payload for point in results[0]]
            else:
                # Retrieve child chunks only
                child_filter = Filter(
                    must=[FieldCondition(key="chunk_type", match=MatchValue(value="child"))]
                )
                results = indexer.client.scroll(
                    collection_name=config.qdrant.collection,
                    scroll_filter=child_filter,
                    limit=10000,
                    with_payload=True,
                    with_vectors=False,
                )
                chunks = [point.payload for point in results[0]]

        logger.info(f"Building RAPTOR summaries from {len(chunks)} chunks")

        # Step 2: Group chunks by section
        groups = self._group_chunks_by_section(chunks)

        # Filter by paper_id if specified
        if paper_id:
            groups = {k: v for k, v in groups.items() if k.startswith(f"{paper_id}|")}

        # Sort groups and apply limit
        group_keys = list(groups.keys())
        if limit_groups:
            group_keys = group_keys[:limit_groups]

        # Step 3: Check for existing summaries (incremental building)
        existing_summary_ids = set()
        existing_summaries = []
        try:
            # Retrieve all summary nodes to check what's already indexed
            # Use scroll with filter for chunk_type == "summary"
            summary_filter = Filter(
                must=[FieldCondition(key="chunk_type", match=MatchValue(value="summary"))]
            )
            results = self.client.scroll(
                collection_name=config.qdrant.collection,
                scroll_filter=summary_filter,
                limit=10000,
                with_payload=True,
                with_vectors=False,
            )
            # Filter payloads for summary nodes
            for point in results[0]:
                payload = point.payload
                if payload.get("chunk_type") == "summary":
                    existing_summary_ids.add(payload.get("chunk_id", ""))
                    # Also create RaptorNode for deduplication check
                    existing_summaries.append(RaptorNode(
                        chunk_id=payload.get("chunk_id", ""),
                        paper_id=payload.get("paper_id", ""),
                        title=payload.get("title", ""),
                        section=payload.get("section", ""),
                        level=payload.get("level", 1),
                        chunk_type=payload.get("chunk_type", ""),
                        summary=payload.get("summary", ""),
                        source_chunk_ids=payload.get("source_chunk_ids", []),
                        metadata=payload.get("metadata", {}),
                        embedding=None,
                    ))
        except Exception as e:
            logger.warning(f"Could not check existing summaries: {e}")
            # Continue without checking - will just create all summaries

        all_new_nodes = []
        groups_processed = 0

        for group_key in group_keys:
            # Skip if summary already exists
            if not self._needs_summary(group_key, existing_summary_ids):
                logger.debug(f"Skipping {group_key} - summary already exists")
                continue

            chunks_in_group = groups[group_key]

            # Limit chunks per group
            chunks_to_summarize = chunks_in_group[: self.max_chunks_per_group]

            # Create summary node
            node = self._create_summary_node(group_key, chunks_to_summarize, existing_summaries)
            if node is None:
                continue  # Already exists

            # Embed the summary
            node = self._embed_node(node)
            all_new_nodes.append(node)
            groups_processed += 1

            # Upsert in batches
            if len(all_new_nodes) >= self.batch_size:
                self._upsert_batch(all_new_nodes)
                all_new_nodes = []

        # Upsert any remaining nodes
        if all_new_nodes:
            self._upsert_batch(all_new_nodes)

        # Return stats
        from src.ingestion.indexer import IndexStats
        stats = IndexStats(collection_name=config.qdrant.collection)
        stats.upserted = groups_processed
        stats.total_chunks = len(chunks)

        logger.info(f"RAPTOR building complete: {groups_processed} new summary groups from {len(chunks)} child chunks")
        return stats

    def get_summary_nodes(
        self,
        paper_id: Optional[str] = None,
        level: Optional[int] = None,
    ) -> List[RaptorNode]:
        """
        Retrieve RAPTOR summary nodes from Qdrant.

        Args:
            paper_id: Optional filter for a specific paper
            level: Optional filter for level (1 = section, 2 = paper)

        Returns:
            List of RaptorNode objects
        """
        must_conditions = [
            FieldCondition(key="chunk_type", match=MatchValue(value="summary"))
        ]

        if paper_id:
            must_conditions.append(FieldCondition(key="paper_id", match=MatchValue(value=paper_id)))

        if level is not None:
            must_conditions.append(FieldCondition(key="level", match=MatchValue(value=level)))

        filter_obj = Filter(must=must_conditions)

        try:
            results = self.client.scroll(
                collection_name=config.qdrant.collection,
                scroll_filter=filter_obj,
                limit=10000,
                with_payload=True,
                with_vectors=True,
            )
            nodes = []
            for point in results[0]:
                payload = point.payload
                if payload.get("chunk_type") == "summary":
                    node = RaptorNode(
                        chunk_id=payload.get("chunk_id", ""),
                        paper_id=payload.get("paper_id", ""),
                        title=payload.get("title", ""),
                        section=payload.get("section", ""),
                        level=payload.get("level", 1),
                        chunk_type=payload.get("chunk_type", ""),
                        summary=payload.get("summary", ""),
                        source_chunk_ids=payload.get("source_chunk_ids", []),
                        metadata=payload.get("metadata", {}),
                        embedding=point.vector if point.vector else None,
                    )
                    nodes.append(node)
            return nodes
        except Exception as e:
            logger.error(f"Failed to retrieve RAPTOR summary nodes: {e}")
            return []


def get_raptor_builder(
    client: Optional[Any] = None,
    embeddings: Optional[OllamaEmbeddings] = None,
    llm: Optional[OllamaLLM] = None,
    batch_size: int = 16,
    max_chunks_per_group: int = 20,
) -> RaptorBuilder:
    """Get a RAPTOR builder instance."""
    return RaptorBuilder(
        client, embeddings, llm, batch_size, max_chunks_per_group
    )