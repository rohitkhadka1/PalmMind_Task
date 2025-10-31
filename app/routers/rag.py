from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi.responses import JSONResponse

from ..db import get_db
from ..schemas import RAGQueryRequest, RAGQueryResponse, ChatHistoryResponse
from ..services.retrieval import RAGService
from ..services.memory import ChatMemoryManager


router = APIRouter()
rag_service = RAGService()
memory_manager = ChatMemoryManager()


@router.get("/")
async def rag_root():
    """
    Root endpoint for RAG API.
    """
    return JSONResponse({
        "message": "Conversational RAG API",
        "endpoints": {
            "/query": "Submit a query",
            "/history/{session_id}": "Get chat history for a session"
        }
    })


@router.post("/query", response_model=RAGQueryResponse)
@router.get("/query", summary="Get query endpoint information", description="Returns information about how to use the query endpoint")
async def rag_query_info():
    return JSONResponse({
        "message": "This is a POST endpoint. Please send a POST request with the following JSON body:",
        "example_payload": {
            "session_id": "unique_session_id",
            "query": "your question here",
            "top_k": 5
        }
    })

@router.post("/query", response_model=RAGQueryResponse)
async def rag_query(payload: RAGQueryRequest, db: AsyncSession = Depends(get_db)):
    try:
        answer, sources = await rag_service.query(
            db, 
            session_id=payload.session_id, 
            query=payload.query, 
            top_k=payload.top_k
        )
        # Store the interaction in chat memory
        await memory_manager.add_interaction(
            session_id=payload.session_id,
            user_message=payload.query,
            assistant_message=answer
        )
        return RAGQueryResponse(answer=answer, sources=sources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history", summary="Get history endpoint information", description="Returns information about how to use the history endpoint")
async def history_info():
    return JSONResponse({
        "message": "Please use /history/{session_id} to get chat history for a specific session",
        "example": "/history/your_session_id"
    })

@router.get("/history/{session_id}", response_model=ChatHistoryResponse)
async def get_chat_history(session_id: str):
    try:
        history = await memory_manager.get_chat_history(session_id)
        return ChatHistoryResponse(
            session_id=session_id,
            messages=history
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

