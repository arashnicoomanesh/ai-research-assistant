"""
Qdrant Indexer for AI Research Papers.

Handles embedding and upserting chunks to Qdrant vector database.
Supports parent-child chunk structure with rich metadata payloads.
"""
import logging
import time
import uuid
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter, FieldCondition,
    MatchValue, PayloadSchemaType, CollectionStatus
)
from src.core.config import config
from src.core.embeddings import OllamaEmbeddings, get_embeddings
from src.ingestion.chunker import Chunk
from src.retrieval.qdrant_search import qdrant_search

logger = logging.getLogger(__name__)


def _deterministic_uuid(chunk_id: str) -> str:
    """
    Generate a deterministic UUID from a chunk_id string.
    
    Uses UUID v5 (SHA-1 based) with NAMESPACE_URL for consistency.
    Same chunk_id always produces the same UUID.
    
    Args:
        chunk_id: Human-readable chunk identifier
        
    Returns:
        UUID string valid for Qdrant point ID
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


@dataclass
class IndexStats:
    """Statistics from indexing operation."""
    total_chunks: int = 0
    child_chunks: int = 0
    parent_chunks: int = 0
    abstract_chunks: int = 0
    embedded: int = 0
    skipped: int = 0
    upserted: int = 0
    errors: int = 0
    collection_name: str = ""


class QdrantIndexer:
    """
    Indexes paper chunks into Qdrant.

    Stores child and abstract chunks for retrieval (dense vectors).
    Optionally stores parent chunks (metadata only, no vectors) for context retrieval.
    All chunks carry full metadata in payload.
    """

    def __init__(
        self,
        client: Optional[QdrantClient] = None,
        embeddings: Optional[OllamaEmbeddings] = None,
        collection_name: Optional[str] = None,
    ):
        """
        Initialize indexer.

        Args:
            client: Qdrant client (creates default if None)
            embeddings: Embeddings client (creates default if None)
            collection_name: Collection name (from config if None)
        """
        self.client = client or QdrantClient(
            host=config.qdrant.host,
            port=config.qdrant.port,
        )
        self.embeddings = embeddings or get_embeddings()
        self.collection_name = collection_name or config.qdrant.collection
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Create collection if it doesn't exist."""
        try:
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)

            if not exists:
                # Get embedding dimension
                dim = self.embeddings.dimension
                if dim == 0:
                    # Force dimension detection
                    test_emb = self.embeddings.embed("test")
                    dim = len(test_emb)

                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=dim,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info(f"Created collection '{self.collection_name}' with dim={dim}")

            # Create payload indexes for filtering
            self._create_payload_indexes()

        except Exception as e:
            logger.error(f"Failed to ensure collection: {e}")
            raise

    def _create_payload_indexes(self) -> None:
        """Create payload indexes for efficient filtering."""
        indexes = [
            ("paper_id", PayloadSchemaType.KEYWORD),
            ("chunk_type", PayloadSchemaType.KEYWORD),
            ("section", PayloadSchemaType.KEYWORD),
            ("parent_id", PayloadSchemaType.KEYWORD),
        ]

        for field_name, field_type in indexes:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=field_type,
                )
            except Exception:
                # Index might already exist
                pass

    def index_chunks(
        self,
        chunks: List[Chunk],
        batch_size: Optional[int] = None,
        embed_chunk_types: Optional[List[str]] = None,
    ) -> IndexStats:
        """
        Embed and upsert chunks to Qdrant.

        By default, only embeds chunk types in {"child", "abstract"} for vector search.
        Parent chunks are stored without vectors (metadata only) for later context expansion.

        Args:
            chunks: List of Chunk objects
            batch_size: Batch size for embedding and upsert (default from config)
            embed_chunk_types: Chunk types to embed (default from config: child, abstract)

        Returns:
            IndexStats with operation results
        """
        if not chunks:
            return IndexStats(collection_name=self.collection_name)

        if batch_size is None:
            batch_size = config.embedding.batch_size

        # Determine which chunk types to embed
        if embed_chunk_types is None:
            if config.embedding.embed_parents:
                embed_chunk_types = ["child", "parent", "abstract"]
            else:
                embed_chunk_types = ["child", "abstract"]

        stats = IndexStats(collection_name=self.collection_name)

        # Count chunk types
        for chunk in chunks:
            stats.total_chunks += 1
            if chunk.chunk_type == "child":
                stats.child_chunks += 1
            elif chunk.chunk_type == "parent":
                stats.parent_chunks += 1
            elif chunk.chunk_type == "abstract":
                stats.abstract_chunks += 1

        logger.info(f"Indexing {stats.total_chunks} chunks: {stats.child_chunks} child, {stats.parent_chunks} parent, {stats.abstract_chunks} abstract")
        logger.info(f"Will embed chunk types: {embed_chunk_types}")

        # Separate chunks to embed vs store without vectors
        to_embed = [c for c in chunks if c.chunk_type in embed_chunk_types]
        no_vector = [c for c in chunks if c.chunk_type not in embed_chunk_types]

        stats.embedded = len(to_embed)
        stats.skipped = len(no_vector)

        logger.info(f"Embedding {stats.embedded} chunks, storing {stats.skipped} without vectors")

        # Process embed chunks in batches with retry
        total_batches = (len(to_embed) + batch_size - 1) // batch_size
        
        for batch_idx in range(0, len(to_embed), batch_size):
            batch = to_embed[batch_idx:batch_idx + batch_size]
            batch_num = (batch_idx // batch_size) + 1
            
            logger.info(f"Indexing batch {batch_num}/{total_batches} ({len(batch)} chunks)")
            
            success = self._upsert_batch_with_retry(batch, batch_num, total_batches)
            if success:
                stats.upserted += len(batch)
            else:
                stats.errors += len(batch)
                logger.error(f"Batch {batch_num} failed after all retries, continuing with next batch")

        # Store parent chunks without vectors (metadata only)
        if no_vector:
            logger.info(f"Storing {len(no_vector)} parent chunks without vectors (metadata only)")
            self._upsert_no_vector_chunks(no_vector)
            stats.upserted += len(no_vector)

        logger.info(f"Indexed {stats.upserted}/{stats.total_chunks} chunks to {self.collection_name} "
                    f"(embedded={stats.embedded}, skipped={stats.skipped}, errors={stats.errors})")
        return stats

    def _upsert_no_vector_chunks(self, chunks: List[Chunk]) -> None:
        """
        Store chunks without vectors (metadata only) for context expansion.

        Args:
            chunks: List of Chunk objects (typically parent chunks)
        """
        points = []
        for chunk in chunks:
            payload = self._build_payload(chunk)
            # Store without vector - use a zero vector of correct dimension
            # Qdrant requires a vector, so we use a zero vector
            dim = self.embeddings.dimension
            if dim == 0:
                dim = 768  # fallback
            point = PointStruct(
                id=_deterministic_uuid(chunk.chunk_id),
                vector=[0.0] * dim,
                payload=payload,
            )
            points.append(point)

        if points:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True,
            )
            logger.info(f"Upserted {len(points)} parent chunks without vectors")

    def _upsert_batch_with_retry(
        self,
        chunks: List[Chunk],
        batch_num: int,
        total_batches: int,
        max_retries: int = 3,
    ) -> bool:
        """
        Embed and upsert a batch of chunks with retry logic.

        Args:
            chunks: List of Chunk objects
            batch_num: Current batch number
            total_batches: Total number of batches
            max_retries: Maximum retry attempts

        Returns:
            True if successful, False otherwise
        """
        for attempt in range(1, max_retries + 1):
            try:
                # Prepare texts for embedding
                texts = [chunk.text for chunk in chunks]

                # Generate embeddings with retry logic (handled by embeddings client)
                embeddings = self.embeddings.embed_batch(texts)

                # Create points with deterministic UUIDs as point IDs
                points = []
                for chunk, vector in zip(chunks, embeddings):
                    payload = self._build_payload(chunk)
                    point = PointStruct(
                        id=_deterministic_uuid(chunk.chunk_id),
                        vector=vector,
                        payload=payload,
                    )
                    points.append(point)

                # Upsert
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points,
                    wait=True,
                )

                logger.info(f"Batch {batch_num}/{total_batches} upserted successfully ({len(points)} points)")
                return True

            except Exception as e:
                wait_time = 2 ** attempt
                logger.warning(f"Batch {batch_num} attempt {attempt}/{max_retries} failed: {e}")
                if attempt < max_retries:
                    logger.info(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Batch {batch_num} failed after {max_retries} attempts")
                    return False

        return False

    def _upsert_batch(self, chunks: List[Chunk]) -> None:
        """Embed and upsert a batch of chunks (legacy method)."""
        # Prepare texts for embedding
        texts = [chunk.text for chunk in chunks]

        # Generate embeddings
        embeddings = self.embeddings.embed_batch(texts)

        # Create points with deterministic UUIDs as point IDs
        points = []
        for chunk, vector in zip(chunks, embeddings):
            payload = self._build_payload(chunk)
            point = PointStruct(
                id=_deterministic_uuid(chunk.chunk_id),
                vector=vector,
                payload=payload,
            )
            points.append(point)

        # Upsert
        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )

    def _build_payload(self, chunk: Chunk) -> Dict[str, Any]:
        """Build Qdrant payload from chunk."""
        payload = {
            "chunk_id": chunk.chunk_id,
            "paper_id": chunk.paper_id,
            "title": chunk.title,
            "section": chunk.section,
            "chunk_type": chunk.chunk_type,
            "text": chunk.text,
            "chunk_index": chunk.chunk_index,
            "token_count": chunk.token_count,
        }

        # Add parent reference for child chunks
        if chunk.parent_id:
            payload["parent_id"] = chunk.parent_id

        # Add extra metadata
        for key, value in chunk.metadata.items():
            if key not in payload:  # Don't overwrite core fields
                payload[key] = value

        return payload

    def delete_paper_chunks(self, paper_id: str) -> int:
        """
        Delete all chunks for a paper (used before re-indexing).

        Args:
            paper_id: Paper identifier

        Returns:
            Number of deleted points
        """
        try:
            result = self.client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[FieldCondition(key="paper_id", match=MatchValue(value=paper_id))]
                ),
                wait=True,
            )
            logger.info(f"Deleted chunks for paper {paper_id}")
            return result.operation_id if hasattr(result, 'operation_id') else 0
        except Exception as e:
            logger.error(f"Failed to delete chunks for {paper_id}: {e}")
            return 0

    def get_paper_chunks(self, paper_id: str, chunk_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieve all chunks for a paper.

        Args:
            paper_id: Paper identifier
            chunk_type: Optional filter ("child", "parent", "abstract")

        Returns:
            List of chunk payloads
        """
        must_conditions = [FieldCondition(key="paper_id", match=MatchValue(value=paper_id))]
        if chunk_type:
            must_conditions.append(FieldCondition(key="chunk_type", match=MatchValue(value=chunk_type)))

        results = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(must=must_conditions),
            limit=1000,
            with_payload=True,
            with_vectors=False,
        )

        return [point.payload for point in results[0]]

    def search_children(
        self,
        query_vector: List[float],
        top_k: int = 10,
        paper_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search child chunks by vector similarity.

        Args:
            query_vector: Query embedding
            top_k: Number of results
            paper_id: Optional paper filter

        Returns:
            List of matching child chunks with scores
        """
        must_conditions = [FieldCondition(key="chunk_type", match=MatchValue(value="child"))]
        if paper_id:
            must_conditions.append(FieldCondition(key="paper_id", match=MatchValue(value=paper_id)))

        results = qdrant_search(
            client=self.client,
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter=Filter(must=must_conditions),
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )

        return [
            {"score": hit.score, **hit.payload}
            for hit in results
        ]

    def search_all(
        self,
        query_vector: List[float],
        top_k: int = 10,
        chunk_types: Optional[List[str]] = None,
        paper_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search across all chunk types.

        Args:
            query_vector: Query embedding
            top_k: Number of results
            chunk_types: Optional list of chunk types to search
            paper_id: Optional paper filter

        Returns:
            List of matching chunks with scores
        """
        must_conditions = []
        if chunk_types:
            # Search multiple chunk types
            should_conditions = [
                FieldCondition(key="chunk_type", match=MatchValue(value=ct))
                for ct in chunk_types
            ]
            must_conditions.append(Filter(should=should_conditions))
        if paper_id:
            must_conditions.append(FieldCondition(key="paper_id", match=MatchValue(value=paper_id)))

        query_filter = Filter(must=must_conditions) if must_conditions else None

        results = qdrant_search(
            client=self.client,
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )

        return [
            {"score": hit.score, **hit.payload}
            for hit in results
        ]

    def get_collection_info(self) -> Dict[str, Any]:
        """Get collection statistics."""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "status": info.status,
                "vectors_count": info.vectors_count,
                "indexed_vectors_count": info.indexed_vectors_count,
                "points_count": info.points_count,
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {"error": str(e)}


def get_indexer(
    client: Optional[QdrantClient] = None,
    embeddings: Optional[OllamaEmbeddings] = None,
    collection_name: Optional[str] = None,
) -> QdrantIndexer:
    """Get an indexer instance."""
    return QdrantIndexer(client, embeddings, collection_name)