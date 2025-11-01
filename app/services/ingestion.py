from __future__ import annotations

from typing import Iterable
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pdfminer.high_level import extract_text as pdf_extract_text

from ..models import Document, Chunk
from .chunking import split_recursive, split_fixed
from .embeddings import EmbeddingsService
from .vector_store import get_vector_store


async def extract_text_from_file(content: bytes, filename: str, content_type: str) -> str:
    if filename.lower().endswith(".pdf") or content_type == "application/pdf":
        # pdfminer expects a file path or a file-like object
        from io import BytesIO
        from pdfminer.high_level import extract_text
        pdf_file = BytesIO(content)
        try:
            # The extract_text function can work with file-like objects
            text = extract_text(pdf_file)
            if not text:
                raise ValueError("No text could be extracted from the PDF")
            return text
        except Exception as e:
            raise ValueError(f"Failed to extract text from PDF: {str(e)}")
        finally:
            pdf_file.close()
    else:
        try:
            return content.decode("utf-8", errors="ignore")
        except Exception as e:
            raise ValueError(f"Failed to decode text file: {str(e)}")


async def ingest_document(
    db: AsyncSession,
    *,
    filename: str,
    content_type: str,
    raw_bytes: bytes,
    strategy: str = "recursive",
    fixed_size: int = 500,
    fixed_overlap: int = 50,
) -> tuple[int, int]:
    text = await extract_text_from_file(raw_bytes, filename, content_type)
    if not text.strip():
        raise ValueError("No text extracted from file")

    doc = Document(filename=filename, content_type=content_type)
    db.add(doc)
    await db.flush()

    if strategy == "fixed":
        chunks = split_fixed(text, size=fixed_size, overlap=fixed_overlap)
    else:
        chunks = split_recursive(text, max_tokens=fixed_size, overlap=fixed_overlap)

    chunk_models: list[Chunk] = []
    for idx, chunk_text in enumerate(chunks):
        chunk_models.append(Chunk(document_id=doc.id, index_in_document=idx, text=chunk_text))

    db.add_all(chunk_models)
    await db.flush()

    # Embed and upsert to vector store
    embedder = EmbeddingsService()
    vectors = embedder.embed([c.text for c in chunk_models])
    vs = get_vector_store()
    payloads = []
    
    import uuid
    for c, vec in zip(chunk_models, vectors):
        # Generate a UUID for the vector store
        point_id = str(uuid.uuid4())
        # Store the UUID in the chunk for reference
        c.embedding_id = point_id
        payloads.append((point_id, vec, {"chunk_id": c.id, "document_id": doc.id, "text": c.text}))
    vs.upsert(payloads)

    await db.commit()
    return doc.id, len(chunk_models)

