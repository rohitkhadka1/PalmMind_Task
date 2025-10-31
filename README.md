## PalmMind Backend (FastAPI)

### Features
- Document ingestion: .pdf/.txt, two chunkers (recursive, fixed)
- Embeddings: SentenceTransformers (all-MiniLM-L6-v2)
- Vector store: Qdrant (in-memory by default)
- Metadata: SQLite via SQLAlchemy
- Conversational RAG: custom retrieval + prompt, Redis chat memory
- Booking: create/list interview bookings

### Setup
1. Python 3.11+ recommended. Create/activate venv.
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Copy env and edit as needed:
```bash
cp .env.example .env
```
4. Run API:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Environment
See `.env.example` for all values. Key ones:
- `DATABASE_URL=sqlite+aiosqlite:///./app.db`
- `VECTOR_PROVIDER=qdrant`, `QDRANT_URL=:memory:`
- `LLM_PROVIDER=openai`, `OPENAI_API_KEY=...` or use `local` fallback
- `REDIS_URL=redis://localhost:6379/0`

### Endpoints
- POST `/ingest/upload` (multipart): file, strategy, fixed_size, fixed_overlap
- POST `/rag/query`: { session_id, query, top_k }
- POST `/booking/create`: { name, email, date, time }
- GET `/booking/list`
- GET `/health`

### Notes
- No FAISS/Chroma/Chains used; custom RAG pipeline.
- Replace Qdrant with Pinecone/Weaviate/Milvus by implementing `VectorStore` adapter.

