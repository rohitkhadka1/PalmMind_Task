from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi.responses import JSONResponse

from ..db import get_db, engine, Base
from ..schemas import IngestionResponse
from ..services.ingestion import ingest_document


router = APIRouter()


@router.on_event("startup")
async def on_startup() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


@router.get("/")
async def ingestion_root():
    """
    Root endpoint for document ingestion API.
    """
    return JSONResponse({
        "message": "Document Ingestion API",
        "endpoints": {
            "/upload": "Upload PDF or TXT documents",
        }
    })


@router.get("/upload", summary="Get upload endpoint information", description="Returns information about how to upload documents")
async def upload_info():
    return JSONResponse({
        "message": "This is a POST endpoint for uploading documents.",
        "supported_files": [".pdf", ".txt"],
        "form_data": {
            "file": "(file): The document to upload",
            "strategy": "(str, optional): 'recursive' or 'fixed', default='recursive'",
            "fixed_size": "(int, optional): chunk size, default=500",
            "fixed_overlap": "(int, optional): overlap size, default=50"
        },
        "example_curl": """
        curl -X POST "http://localhost:8080/ingest/upload" \\
             -H "accept: application/json" \\
             -H "Content-Type: multipart/form-data" \\
             -F "file=@document.pdf" \\
             -F "strategy=recursive" \\
             -F "fixed_size=500" \\
             -F "fixed_overlap=50"
        """
    })

@router.post("/upload", response_model=IngestionResponse)
async def upload_document(
    file: UploadFile = File(...),
    strategy: str = Form(default="recursive"),
    fixed_size: int = Form(default=500),
    fixed_overlap: int = Form(default=50),
    db: AsyncSession = Depends(get_db),
):
    # Validate file type
    if file.content_type not in {"application/pdf", "text/plain"} and not (
        file.filename.lower().endswith(".pdf") or file.filename.lower().endswith(".txt")
    ):
        raise HTTPException(status_code=400, detail="Only .pdf or .txt files are supported")
    
    # Validate strategy
    if strategy not in ["recursive", "fixed"]:
        raise HTTPException(status_code=400, detail="Strategy must be either 'recursive' or 'fixed'")
    
    # Validate chunking parameters
    if fixed_size < 50 or fixed_size > 2000:
        raise HTTPException(status_code=400, detail="fixed_size must be between 50 and 2000")
    if fixed_overlap < 0 or fixed_overlap >= fixed_size:
        raise HTTPException(status_code=400, detail="fixed_overlap must be between 0 and fixed_size")
    
    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty file content")
            
        doc_id, num = await ingest_document(
            db,
            filename=file.filename,
            content_type=file.content_type or "",
            raw_bytes=content,
            strategy=strategy,
            fixed_size=fixed_size,
            fixed_overlap=fixed_overlap,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")
    return IngestionResponse(document_id=doc_id, num_chunks=num)

