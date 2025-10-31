from datetime import datetime
from typing import Literal, Optional, List
from pydantic import BaseModel, Field, EmailStr


class IngestionResponse(BaseModel):
    document_id: int
    num_chunks: int


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    timestamp: datetime


class ChatHistoryResponse(BaseModel):
    session_id: str
    messages: List[ChatMessage]


class ChunkingStrategyRequest(BaseModel):
    strategy: Literal["recursive", "fixed"] = Field(default="recursive")
    fixed_size: int | None = Field(default=500)
    fixed_overlap: int | None = Field(default=50)


class RAGQueryRequest(BaseModel):
    session_id: str
    query: str
    top_k: int = 5


class RAGQueryResponse(BaseModel):
    answer: str
    sources: list[int]


class BookingCreate(BaseModel):
    name: str
    email: EmailStr
    date: str  # ISO date string
    time: str  # HH:MM


class BookingOut(BaseModel):
    id: int
    name: str
    email: EmailStr
    date: str
    time: str
    created_at: datetime

    model_config = {
        "from_attributes": True
    }

