"""
Record Manager for Incremental Ingestion.

Tracks paper processing state using SQLite:
- paper_id, file_hash, last_indexed_at, status
- Enables skipping unchanged papers on re-ingestion
"""
import sqlite3
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
from contextlib import contextmanager

from src.core.config import config

logger = logging.getLogger(__name__)


@dataclass
class PaperRecord:
    """Record of a paper's ingestion state."""
    paper_id: str
    file_path: str
    file_hash: str
    file_size: int
    status: str  # "pending", "indexed", "failed", "deleted"
    chunk_count: int = 0
    error_message: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    last_indexed_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "file_path": self.file_path,
            "file_hash": self.file_hash,
            "file_size": self.file_size,
            "status": self.status,
            "chunk_count": self.chunk_count,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "last_indexed_at": self.last_indexed_at.isoformat() if self.last_indexed_at else None,
        }

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "PaperRecord":
        return cls(
            paper_id=row["paper_id"],
            file_path=row["file_path"],
            file_hash=row["file_hash"],
            file_size=row["file_size"],
            status=row["status"],
            chunk_count=row["chunk_count"],
            error_message=row["error_message"],
            created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else None,
            updated_at=datetime.fromisoformat(row["updated_at"]) if row["updated_at"] else None,
            last_indexed_at=datetime.fromisoformat(row["last_indexed_at"]) if row["last_indexed_at"] else None,
        )


class RecordManager:
    """
    SQLite-based record manager for tracking paper ingestion state.

    Supports incremental ingestion by comparing file hashes.
    """

    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize record manager.

        Args:
            db_path: Path to SQLite database (default: data/processed/ingestion.db)
        """
        self.db_path = db_path or (config.paths.processed_dir / "ingestion.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Create tables if they don't exist."""
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS paper_records (
                    paper_id TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    file_hash TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    chunk_count INTEGER DEFAULT 0,
                    error_message TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_indexed_at TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_status ON paper_records(status)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_file_hash ON paper_records(file_hash)
            """)
            conn.commit()

    @contextmanager
    def _get_conn(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def get_record(self, paper_id: str) -> Optional[PaperRecord]:
        """Get record by paper_id."""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM paper_records WHERE paper_id = ?",
                (paper_id,)
            ).fetchone()
            return PaperRecord.from_row(row) if row else None

    def get_record_by_hash(self, file_hash: str) -> Optional[PaperRecord]:
        """Get record by file hash (for detecting duplicates)."""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM paper_records WHERE file_hash = ?",
                (file_hash,)
            ).fetchone()
            return PaperRecord.from_row(row) if row else None

    def upsert_record(self, record: PaperRecord) -> None:
        """Insert or update a paper record."""
        now = datetime.now().isoformat()
        with self._get_conn() as conn:
            existing = conn.execute(
                "SELECT paper_id FROM paper_records WHERE paper_id = ?",
                (record.paper_id,)
            ).fetchone()

            if existing:
                conn.execute("""
                    UPDATE paper_records
                    SET file_path = ?, file_hash = ?, file_size = ?,
                        status = ?, chunk_count = ?, error_message = ?,
                        updated_at = ?, last_indexed_at = ?
                    WHERE paper_id = ?
                """, (
                    record.file_path, record.file_hash, record.file_size,
                    record.status, record.chunk_count, record.error_message,
                    now, record.last_indexed_at.isoformat() if record.last_indexed_at else None,
                    record.paper_id
                ))
            else:
                conn.execute("""
                    INSERT INTO paper_records
                    (paper_id, file_path, file_hash, file_size, status,
                     chunk_count, error_message, created_at, updated_at, last_indexed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    record.paper_id, record.file_path, record.file_hash, record.file_size,
                    record.status, record.chunk_count, record.error_message,
                    now, now, record.last_indexed_at.isoformat() if record.last_indexed_at else None
                ))
            conn.commit()

    def mark_indexed(self, paper_id: str, chunk_count: int) -> None:
        """Mark paper as successfully indexed."""
        now = datetime.now().isoformat()
        with self._get_conn() as conn:
            conn.execute("""
                UPDATE paper_records
                SET status = 'indexed', chunk_count = ?, last_indexed_at = ?, updated_at = ?
                WHERE paper_id = ?
            """, (chunk_count, now, now, paper_id))
            conn.commit()

    def mark_failed(self, paper_id: str, error: str) -> None:
        """Mark paper as failed with error message."""
        now = datetime.now().isoformat()
        with self._get_conn() as conn:
            conn.execute("""
                UPDATE paper_records
                SET status = 'failed', error_message = ?, updated_at = ?
                WHERE paper_id = ?
            """, (error, now, paper_id))
            conn.commit()

    def mark_deleted(self, paper_id: str) -> None:
        """Mark paper as deleted (file removed)."""
        now = datetime.now().isoformat()
        with self._get_conn() as conn:
            conn.execute("""
                UPDATE paper_records
                SET status = 'deleted', updated_at = ?
                WHERE paper_id = ?
            """, (now, paper_id))
            conn.commit()

    def get_all_records(self) -> List[PaperRecord]:
        """Get all paper records."""
        with self._get_conn() as conn:
            rows = conn.execute("SELECT * FROM paper_records ORDER BY created_at DESC").fetchall()
            return [PaperRecord.from_row(r) for r in rows]

    def get_indexed_records(self) -> List[PaperRecord]:
        """Get all successfully indexed papers."""
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM paper_records WHERE status = 'indexed' ORDER BY last_indexed_at DESC"
            ).fetchall()
            return [PaperRecord.from_row(r) for r in rows]

    def get_pending_records(self) -> List[PaperRecord]:
        """Get papers pending indexing."""
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM paper_records WHERE status IN ('pending', 'failed') ORDER BY created_at"
            ).fetchall()
            return [PaperRecord.from_row(r) for r in rows]

    def needs_reindex(self, paper_id: str, file_hash: str) -> bool:
        """
        Check if a paper needs re-indexing.

        Returns True if:
        - No record exists (new paper)
        - File hash differs (content changed)
        - Status is not 'indexed' (failed or pending)
        """
        record = self.get_record(paper_id)
        if not record:
            return True  # New paper
        if record.file_hash != file_hash:
            return True  # Content changed
        if record.status != "indexed":
            return True  # Previously failed or pending
        return False

    def get_stats(self) -> Dict[str, Any]:
        """Get ingestion statistics."""
        with self._get_conn() as conn:
            stats = {}
            stats["total"] = conn.execute("SELECT COUNT(*) FROM paper_records").fetchone()[0]
            stats["indexed"] = conn.execute(
                "SELECT COUNT(*) FROM paper_records WHERE status = 'indexed'"
            ).fetchone()[0]
            stats["pending"] = conn.execute(
                "SELECT COUNT(*) FROM paper_records WHERE status = 'pending'"
            ).fetchone()[0]
            stats["failed"] = conn.execute(
                "SELECT COUNT(*) FROM paper_records WHERE status = 'failed'"
            ).fetchone()[0]
            stats["deleted"] = conn.execute(
                "SELECT COUNT(*) FROM paper_records WHERE status = 'deleted'"
            ).fetchone()[0]
            stats["total_chunks"] = conn.execute(
                "SELECT COALESCE(SUM(chunk_count), 0) FROM paper_records WHERE status = 'indexed'"
            ).fetchone()[0]
            return stats

    def cleanup_deleted_files(self, existing_files: List[Path]) -> int:
        """
        Mark records as deleted for files that no longer exist.

        Args:
            existing_files: List of currently existing PDF files

        Returns:
            Number of records marked as deleted
        """
        existing_ids = {f.stem for f in existing_files}
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT paper_id FROM paper_records WHERE status != 'deleted'"
            ).fetchall()
            deleted_count = 0
            for row in rows:
                if row["paper_id"] not in existing_ids:
                    self.mark_deleted(row["paper_id"])
                    deleted_count += 1
            return deleted_count


# Convenience functions
def get_record_manager(db_path: Optional[Path] = None) -> RecordManager:
    """Get a record manager instance."""
    return RecordManager(db_path)


def create_record(paper_id: str, file_path: Path, file_hash: str, file_size: int) -> PaperRecord:
    """Create a new paper record."""
    return PaperRecord(
        paper_id=paper_id,
        file_path=str(file_path),
        file_hash=file_hash,
        file_size=file_size,
        status="pending",
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )