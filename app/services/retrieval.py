from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from .embeddings import EmbeddingsService
from .vector_store import get_vector_store
from .memory import ChatMemoryManager
from .llm import LLMProvider, SYSTEM_PROMPT
from ..models import Chunk


class RAGService:
    def __init__(self):
        self.embedder = EmbeddingsService()
        self.vstore = get_vector_store()
        self.memory = ChatMemoryManager()
        self.llm = LLMProvider()

    async def query(self, db: AsyncSession, *, session_id: str, query: str, top_k: int = 5) -> tuple[str, list[int]]:
        q_emb = self.embedder.embed([query])[0]
        results = self.vstore.query(q_emb, top_k)
        chunk_ids = [r.chunk_id for r in results]

        # Fetch chunk texts in case the store did not return full text
        if any(not r.text for r in results):
            rows = (await db.execute(select(Chunk).where(Chunk.id.in_(chunk_ids)))).scalars().all()
            id_to_text = {c.id: c.text for c in rows}
            for r in results:
                if not r.text:
                    r.text = id_to_text.get(r.chunk_id, "")

        context = "\n\n".join([f"[Doc {r.document_id} | Score {r.score:.3f}]\n{r.text}" for r in results])
        history = await self.memory.get_chat_history(session_id)

        messages: list[dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": f"Context:\n{context}"},
        ]
        # Convert ChatMessage objects to dict format for LLM
        for msg in history[-10:]:
            messages.append({"role": msg.role, "content": msg.content})
        messages.append({"role": "user", "content": query})

        answer = await self.llm.generate(messages)
        await self.memory.add_interaction(session_id, query, answer)

        return answer, chunk_ids

