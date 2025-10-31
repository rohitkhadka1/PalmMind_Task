from __future__ import annotations

from typing import List
from sentence_transformers import SentenceTransformer
from . import types as svc_types
from ..config import settings


class EmbeddingsService:
    def __init__(self, model_name: str | None = None):
        name = model_name or settings.embedding_model_name
        self._model = SentenceTransformer(name)

    def embed(self, texts: List[str]) -> List[list[float]]:
        embeddings = self._model.encode(texts, convert_to_numpy=False, normalize_embeddings=True)
        return [list(map(float, vec)) for vec in embeddings]

