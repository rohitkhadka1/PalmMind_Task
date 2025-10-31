from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from .routers import ingestion, rag, booking, health
from .db import engine, Base


def create_app() -> FastAPI:
    app = FastAPI(
        title="PalmMind Backend",
        description="""
        Document Ingestion and Conversational RAG API

        ## Features
        * 📄 Document Ingestion API (/ingest)
            * Upload PDF or TXT files
            * Extract text with recursive or fixed chunking
            * Generate embeddings & store in vector database
        * 💬 Conversational RAG API (/rag)
            * Submit queries with context from documents
            * Multi-turn conversation support
            * Chat history management
        * 📅 Booking API (/booking)
            * Schedule interviews
            * View bookings
        """,
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router, prefix="/health", tags=["health"]) 
    app.include_router(ingestion.router, prefix="/ingest", tags=["ingestion"]) 
    app.include_router(rag.router, prefix="/rag", tags=["rag"]) 
    app.include_router(booking.router, prefix="/booking", tags=["booking"]) 
    return app


app = create_app()

