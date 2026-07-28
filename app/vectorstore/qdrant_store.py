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

    # --- read ---------------------------------------------------------------
    def search(self, vector: list[float], top_k: int) -> list[models.ScoredPoint]:
        # TODO(level-1): plain dense search. Consider hybrid (dense + sparse/BM25),
        #                which Qdrant supports with named vectors + Query API.
        # TODO(level-3): pass a query_filter to scope retrieval to part of the doc.
        # client.search() was removed in qdrant-client 1.15; the Query API replaces it.
        return self.client.query_points(
            collection_name=self.collection,
            query=vector,
            limit=top_k,
            with_payload=True,
        ).points


@lru_cache
def get_store() -> VectorStore:
    s = get_settings()
    return VectorStore(s.qdrant_url, s.qdrant_collection)
