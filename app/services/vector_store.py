from __future__ import annotations

from typing import Sequence
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from .types import RetrievedChunk, VectorStore
from ..config import settings


class QdrantVectorStore(VectorStore):
    def __init__(self, collection: str | None = None, dim: int | None = None):
        self.collection = collection or settings.qdrant_collection
        self.dim = dim or settings.embedding_dim
        # Support in-memory instance when configured
        if settings.qdrant_url == ":memory:":
            self.client = QdrantClient(":memory:")
        else:
            self.client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key, prefer_grpc=False)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        collections = self.client.get_collections()
        existing = {c.name for c in collections.collections}
        if self.collection not in existing:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=qmodels.VectorParams(size=self.dim, distance=qmodels.Distance.COSINE),
            )

    def upsert(self, vectors: Sequence[tuple[str, list[float], dict]]) -> None:
        points = [
            qmodels.PointStruct(id=vid, vector=vec, payload=payload) for vid, vec, payload in vectors
        ]
        self.client.upsert(collection_name=self.collection, points=points)

    def query(self, embedding: list[float], top_k: int) -> list[RetrievedChunk]:
        res = self.client.search(
            collection_name=self.collection,
            query_vector=embedding,
            limit=top_k,
            with_payload=True,
        )
        out: list[RetrievedChunk] = []
        for p in res:
            payload = p.payload or {}
            out.append(
                RetrievedChunk(
                    chunk_id=int(payload["chunk_id"]),
                    document_id=int(payload["document_id"]),
                    text=str(payload["text"]),
                    score=float(p.score),
                )
            )
        return out


def get_vector_store() -> VectorStore:
    # Only Qdrant implemented by default; others can be added with same interface
    return QdrantVectorStore()

