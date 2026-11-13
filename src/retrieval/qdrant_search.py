"""
Qdrant Search Compatibility Layer.

Provides a unified search function compatible with qdrant-client >=1.0.
The modern API uses `query_points` instead of deprecated `search`.
"""
from typing import List, Optional, Any, Dict
from qdrant_client import QdrantClient
from qdrant_client.models import Filter


def qdrant_search(
    client: QdrantClient,
    collection_name: str,
    query_vector: List[float],
    query_filter: Optional[Filter] = None,
    limit: int = 10,
    with_payload: bool = True,
    with_vectors: bool = False,
) -> List[Any]:
    """
    Search Qdrant collection using modern query_points API.

    Args:
        client: QdrantClient instance
        collection_name: Target collection
        query_vector: Query embedding vector
        query_filter: Optional payload filter
        limit: Maximum results to return
        with_payload: Include payload in results
        with_vectors: Include vectors in results

    Returns:
        List of ScoredPoint objects (same format as legacy client.search)
    """
    # Use modern query_points API (qdrant-client >=1.0)
    # Falls back to legacy search if query_points unavailable
    if hasattr(client, "query_points"):
        response = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=limit,
            with_payload=with_payload,
            with_vectors=with_vectors,
        )
        return response.points
    else:
        # Legacy API (qdrant-client <1.0)
        return client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            query_filter=query_filter,
            limit=limit,
            with_payload=with_payload,
            with_vectors=with_vectors,
        )