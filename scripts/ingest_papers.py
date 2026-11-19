#!/usr/bin/env python3
"""
Ingest Papers Script - Incremental Paper Ingestion Pipeline.

Scans data/papers/, checks for changes via record manager,
and processes only new or modified papers through:
  PDF Parsing -> Structure-aware Chunking -> Qdrant Indexing

Usage:
    python scripts/ingest_papers.py           # Incremental ingestion (default)
    python scripts/ingest_papers.py --force   # Force re-index all
    python scripts/ingest_papers.py --stats   # Show statistics
    python scripts/ingest_papers.py --clean   # Clean deleted file records
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.pdf_parser import PDFParser, ParsedPaper, parse_all_papers
from src.ingestion.chunker import ParentChildChunker, Chunk, chunk_paper
from src.ingestion.indexer import QdrantIndexer, get_indexer, IndexStats
from src.ingestion.record_manager import RecordManager, get_record_manager, create_record, PaperRecord
from src.retrieval.bm25_index import get_bm25_indexer
from src.core.config import config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ingest")


class IngestionPipeline:
    """Orchestrates the full ingestion pipeline."""

    def __init__(
        self,
        papers_dir: Optional[Path] = None,
        force_reindex: bool = False,
    ):
        self.papers_dir = papers_dir or config.paths.papers_dir
        self.force_reindex = force_reindex

        # Initialize components
        self.parser = PDFParser()
        self.chunker = ParentChildChunker(
            child_chunk_size=config.chunking.chunk_size,
            child_chunk_overlap=config.chunking.chunk_overlap,
            min_chunk_size=config.chunking.min_chunk_size,
        )
        self.indexer = get_indexer()
        self.record_mgr = get_record_manager()
        self.bm25_indexer = get_bm25_indexer()

    def run(self) -> dict:
        """
        Run the incremental ingestion pipeline.

        Returns:
            Dictionary with statistics
        """
        logger.info("=" * 60)
        logger.info("Starting paper ingestion pipeline")
        logger.info(f"Papers directory: {self.papers_dir}")
        logger.info(f"Force reindex: {self.force_reindex}")
        logger.info("=" * 60)

        # 1. Discover PDF files
        pdf_files = list(self.papers_dir.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files")

        if not pdf_files:
            logger.warning("No PDF files found in papers directory")
            return {"processed": 0, "skipped": 0, "failed": 0, "total_chunks": 0}

        # 2. Clean up records for deleted files (also deletes vectors from Qdrant)
        deleted_count = self.record_mgr.cleanup_deleted_files(pdf_files, indexer=self.indexer)
        if deleted_count:
            logger.info(f"Marked {deleted_count} records as deleted (files removed)")

        # 3. Process each paper
        stats = {
            "processed": 0,
            "skipped": 0,
            "failed": 0,
            "total_chunks": 0,
            "papers": [],
        }

        for pdf_file in pdf_files:
            paper_stats = self._process_paper(pdf_file)
            stats["papers"].append(paper_stats)

            if paper_stats["status"] == "indexed":
                stats["processed"] += 1
                stats["total_chunks"] += paper_stats["chunks"]
            elif paper_stats["status"] == "skipped":
                stats["skipped"] += 1
            else:
                stats["failed"] += 1

        # 4. Print summary
        self._print_summary(stats)

        # 5. Rebuild BM25 index if any papers were processed
        if stats["processed"] > 0:
            logger.info("Rebuilding BM25 index...")
            try:
                self.bm25_indexer.build_from_qdrant()
            except Exception as e:
                logger.warning(f"BM25 index rebuild failed: {e}")

        return stats

    def _process_paper(self, pdf_file: Path) -> dict:
        """Process a single paper through the pipeline."""
        paper_id = pdf_file.stem
        logger.info(f"Processing: {pdf_file.name}")

        # Compute file hash
        file_hash = self.parser.compute_file_hash(pdf_file)
        file_size = pdf_file.stat().st_size

        # Check if re-indexing needed and get existing record status
        existing_record = self.record_mgr.get_record(paper_id)
        was_indexed = existing_record is not None and existing_record.status == "indexed"
        needs_reindex = self.force_reindex or self.record_mgr.needs_reindex(paper_id, file_hash)

        if not needs_reindex:
            logger.info(f"  -> Skipping (unchanged): {paper_id}")
            return {
                "paper_id": paper_id,
                "file": pdf_file.name,
                "status": "skipped",
                "chunks": 0,
                "parser": "none",
            }

        # Create/update record with pending status
        record = create_record(paper_id, pdf_file, file_hash, file_size)
        self.record_mgr.upsert_record(record)

        # Delete old chunks BEFORE indexing if this is a re-index of a previously indexed paper
        if was_indexed:
            logger.info(f"  -> Removing old chunks for re-index...")
            self.indexer.delete_paper_chunks(paper_id)

        try:
            # Step 1: Parse PDF
            logger.info(f"  -> Parsing PDF...")
            paper = self.parser.parse(pdf_file)
            logger.info(f"     Title: {paper.title[:80]}...")
            logger.info(f"     Sections: {len(paper.sections)}, Pages: {paper.page_count}")
            logger.info(f"     Parser: {paper.parser_used}")

            # Step 2: Chunk
            logger.info(f"  -> Chunking (parent-child)...")
            chunks = self.chunker.chunk_paper(paper)
            child_count = sum(1 for c in chunks if c.chunk_type == "child")
            parent_count = sum(1 for c in chunks if c.chunk_type == "parent")
            abstract_count = sum(1 for c in chunks if c.chunk_type == "abstract")
            logger.info(f"     Chunks: {len(chunks)} total ({child_count} child, {parent_count} parent, {abstract_count} abstract)")

            # Step 3: Index
            logger.info(f"  -> Indexing to Qdrant...")
            index_stats = self.indexer.index_chunks(chunks)
            logger.info(f"     Upserted: {index_stats.upserted}, Errors: {index_stats.errors}")

            # Step 4: Check for batch errors - do not mark success if any batch failed
            if index_stats.errors > 0:
                error_msg = f"Batch upsert failed for {index_stats.errors} chunks"
                logger.error(f"  -> FAILED: {error_msg}")
                self.record_mgr.mark_failed(paper_id, error_msg)
                return {
                    "paper_id": paper_id,
                    "file": pdf_file.name,
                    "status": "failed",
                    "chunks": 0,
                    "error": error_msg,
                }

            # Step 5: Update record only on full success
            self.record_mgr.mark_indexed(paper_id, len(chunks))

            return {
                "paper_id": paper_id,
                "file": pdf_file.name,
                "status": "indexed",
                "chunks": len(chunks),
                "child_chunks": child_count,
                "parent_chunks": parent_count,
                "abstract_chunks": abstract_count,
                "parser": paper.parser_used,
                "title": paper.title,
            }

        except Exception as e:
            logger.error(f"  -> FAILED: {e}")
            self.record_mgr.mark_failed(paper_id, str(e))
            return {
                "paper_id": paper_id,
                "file": pdf_file.name,
                "status": "failed",
                "chunks": 0,
                "error": str(e),
            }

    def _print_summary(self, stats: dict) -> None:
        """Print ingestion summary."""
        logger.info("=" * 60)
        logger.info("INGESTION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Papers processed: {stats['processed']}")
        logger.info(f"Papers skipped:   {stats['skipped']}")
        logger.info(f"Papers failed:    {stats['failed']}")
        logger.info(f"Total chunks:     {stats['total_chunks']}")

        if stats['papers']:
            logger.info("\nPer-paper details:")
            for p in stats['papers']:
                status_icon = {
                    "indexed": "✓",
                    "skipped": "⊘",
                    "failed": "✗",
                }.get(p['status'], "?")
                chunk_info = f" ({p.get('chunks', 0)} chunks)" if p['status'] == "indexed" else ""
                logger.info(f"  {status_icon} {p['file']}{chunk_info}")

        # Show record manager stats
        rm_stats = self.record_mgr.get_stats()
        logger.info("\nRecord Manager:")
        logger.info(f"  Total records: {rm_stats['total']}")
        logger.info(f"  Indexed:       {rm_stats['indexed']}")
        logger.info(f"  Pending:       {rm_stats['pending']}")
        logger.info(f"  Failed:        {rm_stats['failed']}")
        logger.info(f"  Total chunks:  {rm_stats['total_chunks']}")

        # Show Qdrant stats
        qdrant_info = self.indexer.get_collection_info()
        logger.info(f"\nQdrant Collection '{self.indexer.collection_name}':")
        logger.info(f"  Points: {qdrant_info.get('points_count', 'N/A')}")
        logger.info(f"  Vectors: {qdrant_info.get('vectors_count', 'N/A')}")


def show_stats():
    """Show ingestion and Qdrant statistics."""
    record_mgr = get_record_manager()
    indexer = get_indexer()

    print("\n" + "=" * 60)
    print("INGESTION STATISTICS")
    print("=" * 60)

    rm_stats = record_mgr.get_stats()
    print(f"\nRecord Manager:")
    print(f"  Total papers:    {rm_stats['total']}")
    print(f"  Indexed:         {rm_stats['indexed']}")
    print(f"  Pending:         {rm_stats['pending']}")
    print(f"  Failed:          {rm_stats['failed']}")
    print(f"  Deleted:         {rm_stats['deleted']}")
    print(f"  Total chunks:    {rm_stats['total_chunks']}")

    print(f"\nQdrant Collection '{indexer.collection_name}':")
    qdrant_info = indexer.get_collection_info()
    for key, value in qdrant_info.items():
        print(f"  {key}: {value}")

    # List all papers
    records = record_mgr.get_all_records()
    if records:
        print("\nPaper Records:")
        print(f"  {'ID':<30} {'Status':<10} {'Chunks':<8} {'Hash':<16}")
        print(f"  {'-'*30} {'-'*10} {'-'*8} {'-'*16}")
        for r in records:
            print(f"  {r.paper_id:<30} {r.status:<10} {r.chunk_count:<8} {r.file_hash[:16]}")


def clean_deleted():
    """Clean up records for deleted files."""
    papers_dir = config.paths.papers_dir
    pdf_files = list(papers_dir.glob("*.pdf"))
    record_mgr = get_record_manager()
    indexer = get_indexer()

    deleted = record_mgr.cleanup_deleted_files(pdf_files, indexer=indexer)
    print(f"Marked {deleted} records as deleted")


def rebuild_bm25():
    """Rebuild BM25 index from Qdrant."""
    from src.retrieval.bm25_index import get_bm25_indexer
    bm25_indexer = get_bm25_indexer()
    count = bm25_indexer.build_from_qdrant()
    print(f"Rebuilt BM25 index with {count} documents")


def force_reindex_all():
    """Force re-index all papers."""
    pipeline = IngestionPipeline(force_reindex=True)
    pipeline.run()


def main():
    parser = argparse.ArgumentParser(
        usage="python scripts/ingest_papers.py [options]",
        description="Incremental paper ingestion pipeline.\n\n"
                    "Scans data/papers/ for PDF files, checks for changes via record manager,\n"
                    "and processes only new or modified papers through:\n"
                    "  1. PDF Parsing (docling - structure-aware)\n"
                    "  2. Parent-child chunking (preserves sections)\n"
                    "  3. Qdrant vector indexing\n\n"
                    "By default, only changed papers are processed (incremental mode).\n"
                    "Use --force to re-index everything.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/ingest_papers.py
    Incremental ingestion - only new/changed papers (default).

  python scripts/ingest_papers.py --force
    Force re-index all papers, ignoring file hashes.

  python scripts/ingest_papers.py --stats
    Show ingestion and Qdrant statistics and exit.

  python scripts/ingest_papers.py --clean
    Clean up records for deleted files and exit.

  python scripts/ingest_papers.py --rebuild-bm25
    Rebuild BM25 keyword index from Qdrant and exit.

  python scripts/ingest_papers.py --verbose
    Enable debug logging.

Workflow:
  1. Place PDFs in data/papers/
  2. Run: python scripts/ingest_papers.py
  3. (Optional) Build RAPTOR summaries: python scripts/build_raptor.py
  4. Query: python src/main.py "Your question"
"""
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force re-index all papers (ignore file hashes)"
    )
    parser.add_argument(
        "--stats", "-s",
        action="store_true",
        help="Show ingestion/Qdrant statistics and exit"
    )
    parser.add_argument(
        "--clean", "-c",
        action="store_true",
        help="Clean up records for deleted files and exit"
    )
    parser.add_argument(
        "--rebuild-bm25",
        action="store_true",
        help="Rebuild BM25 keyword index from Qdrant and exit"
    )
    parser.add_argument(
        "--papers-dir", "-d",
        type=Path,
        help="Override papers directory (default: data/papers/)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.stats:
        show_stats()
        return

    if args.clean:
        clean_deleted()
        return

    if args.rebuild_bm25:
        rebuild_bm25()
        return

    pipeline = IngestionPipeline(
        papers_dir=args.papers_dir,
        force_reindex=args.force,
    )
    pipeline.run()


if __name__ == "__main__":
    main()