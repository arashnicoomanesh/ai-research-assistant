#!/usr/bin/env python3
"""
Build GraphRAG Knowledge Graph CLI.

Extracts entities and relations from paper content and builds a local knowledge graph.
Supports incremental building, per-paper processing, and chunk limits for testing.
Features robust entity extraction with multi-stage fallback for CPU-friendly models.

Usage:
    python scripts/build_graph.py                   # Build graph for all papers
    python scripts/build_graph.py --paper PAPER_ID  # Build for one paper
    python scripts/build_graph.py --stats           # Show graph statistics
    python scripts/build_graph.py --limit N         # Limit chunks for testing
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.graph import get_graph_store, get_entity_extractor
from src.ingestion.indexer import get_indexer
from src.core.config import config
from qdrant_client.models import Filter, FieldCondition, MatchValue

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("build_graph")

# Track chunk processing status
@dataclass
class ChunkStatus:
    """Status tracking for a chunk during graph building."""
    chunk_id: str
    paper_id: str
    section: str
    status: str = "pending"  # pending, processing, success, failed
    entities: int = 0
    relations: int = 0


def get_priority_chunks(indexer, paper_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Retrieve chunks prioritized for graph extraction:
    1. Abstract chunks (highest priority)
    2. Introduction sections
    3. Method/Related Work sections
    4. Other child chunks (limited)
    """
    must_conditions = []
    if paper_id:
        must_conditions.append(FieldCondition(key="paper_id", match=MatchValue(value=paper_id)))

    # Get abstract chunks first
    abstract_filter = Filter(
        must=[
            FieldCondition(key="chunk_type", match=MatchValue(value="abstract")),
            *[c for c in must_conditions],
        ]
    )
    abstract_results = indexer.client.scroll(
        collection_name=config.qdrant.collection,
        scroll_filter=abstract_filter,
        limit=1000,
        with_payload=True,
        with_vectors=False,
    )
    abstract_chunks = [point.payload for point in abstract_results[0]]

    # Get introduction/method child chunks
    intro_sections = ["Introduction", "Introduction.", "1 Introduction", "1. Introduction",
                      "Methods", "Method", "Methodology", "Approach", "Related Work"]
    intro_chunks = []
    for section in intro_sections:
        section_filter = Filter(
            must=[
                FieldCondition(key="chunk_type", match=MatchValue(value="child")),
                FieldCondition(key="section", match=MatchValue(value=section)),
                *[c for c in must_conditions],
            ]
        )
        results = indexer.client.scroll(
            collection_name=config.qdrant.collection,
            scroll_filter=section_filter,
            limit=200,
            with_payload=True,
            with_vectors=False,
        )
        for point in results[0]:
            intro_chunks.append(point.payload)

    # Get other child chunks (limited)
    other_filter = Filter(
        must=[
            FieldCondition(key="chunk_type", match=MatchValue(value="child")),
            *[c for c in must_conditions],
        ]
    )
    other_results = indexer.client.scroll(
        collection_name=config.qdrant.collection,
        scroll_filter=other_filter,
        limit=500,
        with_payload=True,
        with_vectors=False,
    )
    other_chunks = [point.payload for point in other_results[0]]

    # Combine with priority order, avoiding duplicates
    seen_ids = set()
    all_chunks = []

    for chunk in abstract_chunks + intro_chunks + other_chunks:
        chunk_id = chunk.get("chunk_id", "")
        if chunk_id and chunk_id not in seen_ids:
            seen_ids.add(chunk_id)
            all_chunks.append(chunk)

    return all_chunks


def main():
    parser = argparse.ArgumentParser(
        usage="python scripts/build_graph.py [options]",
        description="Build GraphRAG knowledge graph from paper chunks.\n\n"
                    "Extracts entities and relations using LLM with multi-stage fallback.\n"
                    "Prioritizes abstracts and important sections for extraction.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/build_graph.py
    Build graph for all papers (default).

  python scripts/build_graph.py --paper ai_papers
    Build graph for one paper (by paper_id).

  python scripts/build_graph.py --limit 10
    Limit to first 10 chunks (useful for testing).

  python scripts/build_graph.py --stats
    Show graph statistics without building.

  python scripts/build_graph.py --paper ai_papers --limit 5
    Build for one paper, limited to 5 chunks.

  python scripts/build_graph.py --debug
    Show debug output on JSON failures (truncated LLM output).

Workflow:
  1. Ingest papers: python scripts/ingest_papers.py
  2. Build graph: python scripts/build_graph.py
  3. Query with GraphRAG: python src/main.py "How is X related to Y?"
"""
    )
    parser.add_argument(
        "--paper", "-p",
        type=str,
        help="Build graph for one paper (by paper_id stem)"
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="Limit number of chunks to process (for testing)"
    )
    parser.add_argument(
        "--stats", "-s",
        action="store_true",
        help="Show graph statistics only (no building)"
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Build for all papers (default if --stats not set)"
    )
    parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="Show debug output on JSON failures (truncated LLM output)",
    )

    args = parser.parse_args()

    # Initialize components
    logger.info("Initializing GraphRAG components...")
    graph_store = get_graph_store()
    entity_extractor = get_entity_extractor()
    indexer = get_indexer()

    # If --stats mode, just show stats
    if args.stats:
        logger.info("Showing graph statistics...")
        stats = graph_store.stats()
        print(f"\nGraph Statistics:")
        print(f"  Nodes: {stats['node_count']}")
        print(f"  Edges: {stats['edge_count']}")
        print(f"  Entity Types: {stats['types']}")
        print(f"  Relation Types: {stats['relation_types']}")

        # Show some example entities
        entities = graph_store.all_entities()
        if entities:
            print(f"\nExample Entities (first 10):")
            for e in entities[:10]:
                print(f"  - {e.name} ({e.type.value}) [ID: {e.id}]")

        # Show some example relations
        relations = graph_store.all_relations()
        if relations:
            print(f"\nExample Relations (first 10):")
            for r in relations[:10]:
                source = graph_store.get_entity(r.source_id)
                target = graph_store.get_entity(r.target_id)
                s_name = source.name if source else r.source_id
                t_name = target.name if target else r.target_id
                print(f"  - {s_name} -[{r.relation_type.value}]-> {t_name}")

        return

    # Step 1: Retrieve chunks for extraction
    logger.info("Step 1: Retrieving chunks from Qdrant...")
    chunks = get_priority_chunks(indexer, args.paper)

    if args.limit:
        chunks = chunks[:args.limit]

    logger.info(f"Retrieved {len(chunks)} chunks for graph extraction")

    if not chunks:
        logger.warning("No chunks found. Index papers first.")
        return

    # Step 2: Filter out already processed chunks (incremental)
    logger.info("Step 2: Filtering already processed chunks...")
    to_process = []
    skipped = 0
    failed = 0
    for chunk in chunks:
        chunk_id = chunk.get("chunk_id", "")
        if chunk_id and graph_store.is_chunk_processed(chunk_id):
            skipped += 1
        elif chunk_id and graph_store.is_chunk_failed(chunk_id):
            # Retryable failed chunks
            failed += 1
            to_process.append(chunk)
        else:
            to_process.append(chunk)

    logger.info(f"Processing {len(to_process)} new chunks ({skipped} already processed, {failed} retryable failed)")

    if not to_process and not failed:
        logger.info("All chunks already processed. Nothing to do.")
        return

    # Step 3: Extract entities and relations with running totals
    logger.info("Step 3: Extracting entities and relations...")
    total_entities = 0
    total_relations = 0
    success_count = 0
    fail_count = 0

    for i, chunk in enumerate(to_process):
        chunk_id = chunk.get("chunk_id", f"chunk_{i}")
        paper_id = chunk.get("paper_id", "unknown")
        text = chunk.get("text", "")
        section = chunk.get("section", "")

        if not text.strip():
            fail_count += 1
            graph_store.mark_chunk_failed(chunk_id)
            continue

        logger.info(f"  [{i+1}/{len(to_process)}] {paper_id} [{section}] ({len(text)} chars)")

        try:
            entities, relations = entity_extractor.extract(
                text=text,
                paper_id=paper_id,
                chunk_id=chunk_id,
            )

            if entities:
                graph_store.add_entities(entities)
                total_entities += len(entities)
                success_count += 1
            else:
                success_count += 1  # No entities but didn't crash

            if relations:
                graph_store.add_relations(relations)
                total_relations += len(relations)

            # Mark as processed (even if 0 entities, it succeeded without crashing)
            graph_store.mark_chunk_processed(chunk_id)

            if entities or relations:
                logger.info(f"    Success: {len(entities)} entities, {len(relations)} relations")
            else:
                logger.debug(f"    Processed: no entities/relations extracted (chunk saved)")

        except Exception as e:
            logger.error(f"    Extraction failed for {chunk_id}: {e}")
            if args.debug:
                # Debug: we already have the error, but could log more
                pass
            fail_count += 1
            graph_store.mark_chunk_failed(chunk_id)

        # Save graph after every successful chunk (incremental safety)
        if success_count > 0 and success_count % 5 == 0:
            graph_store.save()
            logger.info(f"    Graph saved after {success_count} chunks")

        if fail_count > 0 and fail_count % 5 == 0:
            graph_store.save()
            logger.info(f"    Graph saved after {fail_count} failures")

    # Step 4: Save graph final
    logger.info("Step 4: Saving graph...")
    graph_store.save()

    # Step 5: Report results
    logger.info("Graph building complete!")
    stats = graph_store.stats()
    logger.info(f"  Total nodes: {stats['node_count']}")
    logger.info(f"  Total edges: {stats['edge_count']}")
    logger.info(f"  New entities: {total_entities}")
    logger.info(f"  New relations: {total_relations}")
    logger.info(f"  Successes: {success_count}")
    logger.info(f"  Failures: {fail_count}")
    logger.info(f"  Entity types: {stats['types']}")
    logger.info(f"  Relation types: {stats['relation_types']}")

    # Show examples
    entities = graph_store.all_entities()
    if entities:
        logger.info("Example entities:")
        for e in entities[:5]:
            logger.info(f"  - {e.name} ({e.type.value})")

    relations = graph_store.all_relations()
    if relations:
        logger.info("Example relations:")
        for r in relations[:5]:
            source = graph_store.get_entity(r.source_id)
            target = graph_store.get_entity(r.target_id)
            s_name = source.name if source else r.source_id
            t_name = target.name if target else r.target_id
            logger.info(f"  - {s_name} -[{r.relation_type.value}]-> {t_name}: {r.evidence[:60]}...")


if __name__ == "__main__":
    main()