from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence


@dataclass
class RetrievedChunk:
    chunk_id: int
    document_id: int
    text: str
    score: float


class VectorStore(Protocol):
    def upsert(self, vectors: Sequence[tuple[str, list[float], dict]]) -> None: ...
    def query(self, embedding: list[float], top_k: int) -> list[RetrievedChunk]: ...

