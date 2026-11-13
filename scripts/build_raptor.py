#!/usr/bin/env python3
"""
Build RAPTOR Summaries CLI.

Builds a lightweight hierarchical summary layer from existing child chunks in Qdrant.
Supports incremental building, per-paper processing, and group limits for testing.

Usage:
    python scripts/build_raptor.py                   # Build all summaries (default)
    python scripts/build_raptor.py --paper PAPER_ID  # Build for one paper
    python scripts/build_raptor.py --limit N       # Limit to N groups (for testing)
    python scripts/build_raptor.py --check           # Check existing summaries only
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.raptor_builder import get_raptor_builder, RaptorBuilder
from src.ingestion.indexer import get_indexer
from src.core.config import config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("build_raptor")


def main():
    parser = argparse.ArgumentParser(
        usage="python scripts/build_raptor.py [options]",
        description="Build RAPTOR hierarchical summaries from child chunks in Qdrant.\n\n"
                    "RAPTOR creates summary layers by:\n"
                    "  1. Grouping child chunks by paper and section\n"
                    "  2. Iteratively summarizing groups using the LLM\n"
                    "  3. Storing summary nodes in Qdrant for enhanced retrieval\n\n"
                    "By default, builds summaries for all papers. Use --paper for single paper,\n"
                    "--limit for testing, --check to view existing summaries only.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/build_raptor.py
    Build RAPTOR summaries for all papers (default).

  python scripts/build_raptor.py --paper ai_papers
    Build summaries for one paper (by paper_id).

  python scripts/build_raptor.py --limit 5
    Limit to first 5 groups (useful for testing).

  python scripts/build_raptor.py --check
    Check existing summaries without building new ones.

  python scripts/build_raptor.py --paper ai_papers --limit 3
    Build for one paper, limited to 3 groups.

Workflow:
  1. Ingest papers: python scripts/ingest_papers.py
  2. Build RAPTOR: python scripts/build_raptor.py
  3. Query with RAPTOR: python src/main.py "Your research question"
"""
    )
    parser.add_argument(
        "--paper", "-p",
        type=str,
        help="Build summaries for one paper (by paper_id stem)"
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="Limit number of groups to process (for testing)"
    )
    parser.add_argument(
        "--check", "-c",
        action="store_true",
        help="Check existing summaries only (no new building)"
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Build all groups (default if --check not set)"
    )

    args = parser.parse_args()

    # Initialize RAPTOR builder
    builder = get_raptor_builder()

    # Initialize Qdrant indexer
    indexer = get_indexer()

    # Step 1: Retrieve chunks
    logger.info("Step 1: Retrieving child chunks from Qdrant...")

    # Get child chunks
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    child_filter = Filter(
        must=[FieldCondition(key="chunk_type", match=MatchValue(value="child"))]
    )

    chunks_results = indexer.client.scroll(
        collection_name=config.qdrant.collection,
        scroll_filter=child_filter,
        limit=10000,
        with_payload=True,
        with_vectors=False,
    )
    chunks = [point.payload for point in chunks_results[0]]

    logger.info(f"Retrieved {len(chunks)} child chunks")

    if not chunks:
        logger.warning("No child chunks found. Index papers first.")
        return

    # If --paper specified, filter chunks
    if args.paper:
        paper_chunks = [c for c in chunks if c.get("paper_id") == args.paper]
        logger.info(f"Filtering to paper '{args.paper}': {len(paper_chunks)} chunks")
        chunks = paper_chunks

    if not chunks:
        logger.warning(f"No chunks found for paper '{args.pater}'")
        return

    # If --check mode, just report existing summaries
    if args.check:
        logger.info("Step 2: Checking existing summaries...")
        existing = builder.get_summary_nodes(paper_id=None)
        # Filter by paper if specified
        if args.paper:
            paper_summaries = [s for s in existing if s.paper_id == args.paper]
            logger.info(f"Found {len(paper_summaries)} existing summaries for '{args.paper}'")
            for s in paper_summaries[:5]:  # Show first 5
                print(f"  - {s.chunk_id}: {s.section} (level {s.level})"
                      f"- {s.summary[:60]}...")
        else:
            logger.info(f"Found {len(existing)} existing summaries total")
            for s in existing[:5]:  # Show first 5
                print(f"  - {s.chunk_id}: {s.section} (level {s.level})"
                      f"- {s.summary[:60]}...")
        return

    # Step 3: Build RAPTOR summaries
    logger.info("Step 3: Building RAPTOR summaries...")
    stats = builder.build(
        chunks=chunks,
        paper_id=args.paper,
        limit_groups=args.limit,
    )

    # Step 4: Report results
    logger.info("RAPTOR building complete!")
    logger.info(f"  Groups processed: {stats.upserted}")
    logger.info(f"  Total child chunks: {stats.total_chunks}")

    # Show a few examples of created summaries
    summary_nodes = builder.get_summary_nodes(paper_id=args.paper if args.paper else None)
    if summary_nodes:
        logger.info("Example summary nodes:")
        for node in summary_nodes[:3]:
            print(f"  - {node.chunk_id}: [{node.section}] level {node.level}")
            print(f"    Summary: {node.summary[:80]}...")


def __main__():
    """Entry point for console_scripts."""
    main()


if __name__ == "__main__":
    main()