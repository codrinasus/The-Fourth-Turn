"""A thin wrapper over qdrant-client.

Enough to store chunks and search them. The interesting improvements live in the
TODO comments — this is a plain single-vector cosine index and nothing more.
"""

from __future__ import annotations

from functools import lru_cache

from qdrant_client import QdrantClient, models

from ..config import get_settings


class VectorStore:
    def __init__(self, url: str, collection: str):
        self.client = QdrantClient(url=url)
        self.collection = collection

    # --- inspection ---------------------------------------------------------
    def list_collections(self) -> list[str]:
        return [c.name for c in self.client.get_collections().collections]

    def exists(self) -> bool:
        return self.client.collection_exists(self.collection)

    def count(self) -> int:
        if not self.exists():
            return 0
        return self.client.count(self.collection).count

    # --- write --------------------------------------------------------------
    def ensure_collection(self, dim: int, reset: bool = False) -> None:
        """Create the collection sized to the embedding dimension.

        The vector size is fixed at creation, so if you change embedding models you
        must re-ingest (or ingest into a differently named collection).
        """
        if reset and self.exists():
            self.client.delete_collection(self.collection)
        if not self.exists():
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
            )
            # TODO(level-3): a payload index on e.g. `page` lets you filter searches
            # (search only the references section, only tables, ...). See
            # client.create_payload_index(...).

    def upsert(self, points: list[models.PointStruct]) -> None:
        self.client.upsert(collection_name=self.collection, points=points)

    def scroll_all(self, limit: int = 10_000) -> list[models.Record]:
        """Every point in the collection, payload only. Used to read the section index back."""
        if not self.exists():
            return []
        records, _ = self.client.scroll(
            collection_name=self.collection, limit=limit, with_payload=True, with_vectors=False
        )
        return records

    # --- read ---------------------------------------------------------------
    def search(
        self,
        vector: list[float],
        top_k: int,
        query_filter: models.Filter | None = None,
    ) -> list[models.ScoredPoint]:
        # client.search() was removed in qdrant-client 1.15; the Query API replaces it.
        return self.client.query_points(
            collection_name=self.collection,
            query=vector,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        ).points


@lru_cache
def get_store(collection: str | None = None) -> VectorStore:
    """The store for `collection`, defaulting to the chunk index.

    Cached per collection name so the section index (Level 3) gets its own client
    without the rest of the app having to thread one through.
    """
    s = get_settings()
    return VectorStore(s.qdrant_url, collection or s.qdrant_collection)
