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
    strategy: Literal["recursive", "fixed"] = Field(
        default="recursive",
        description="Chunking strategy to use"
    )
    fixed_size: int = Field(
        default=500,
        ge=50,
        le=2000,
        description="Size of each chunk in characters"
    )
    fixed_overlap: int = Field(
        default=50,
        ge=0,
        description="Number of characters to overlap between chunks"
    )


from pydantic import BaseModel, Field, EmailStr, constr

class RAGQueryRequest(BaseModel):
    session_id: constr(min_length=1) = Field(..., description="Unique session identifier")
    query: constr(min_length=1, max_length=1000) = Field(..., description="The question to ask")
    top_k: int = Field(default=5, ge=1, le=10, description="Number of relevant chunks to retrieve (1-10)")


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

