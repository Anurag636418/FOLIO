import os
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from src.extraction import process_file_sync
from src.vector_store import vector_store
from src.llm import generate_response_stream

load_dotenv()

# --- Rate limiter ---
limiter = Limiter(key_func=get_remote_address)

app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# --- CORS ---
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:5173"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["Content-Type"],
)

UPLOADS_DIR = Path("uploads")
UPLOADS_DIR.mkdir(exist_ok=True)

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    query: str
    filename: str
    session_token: str
    history: List[Message] = []


@app.post("/api/upload")
@limiter.limit("5/minute")
async def upload_file(request: Request, file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    # Sanitize filename — strip all directory components
    safe_name = Path(file.filename).name
    if not safe_name:
        raise HTTPException(status_code=400, detail="Invalid filename")

    # Whitelist extensions
    ext = Path(safe_name).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File type '{ext}' is not allowed. Please upload a PDF, DOCX, or TXT file."
        )

    file_path = UPLOADS_DIR / safe_name

    try:
        # Stream to disk in 64KB chunks — never reads entire file into RAM
        total_size = 0
        with open(file_path, "wb") as f:
            while True:
                chunk = await file.read(65536)
                if not chunk:
                    break
                total_size += len(chunk)
                if total_size > MAX_FILE_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail="File is too large. Maximum allowed size is 10MB."
                    )
                f.write(chunk)

        chunks = process_file_sync(str(file_path))
        if not chunks:
            raise HTTPException(status_code=400, detail="Could not extract text from file. Is it empty or scanned?")

        success, session_token = vector_store.add_document(safe_name, chunks)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to build vector index. Please try again.")

        meta = vector_store.get_doc_meta(safe_name, session_token)
        return {
            "filename": safe_name,
            "session_token": session_token,
            "message": "Successfully indexed",
            "meta": meta
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] upload_file: {e}")
        raise HTTPException(status_code=500, detail="Failed to process file. Please try again.")

    finally:
        # Always clean up — no need to keep file on disk after indexing
        if file_path.exists():
            file_path.unlink(missing_ok=True)


@app.post("/api/chat")
@limiter.limit("30/minute")
async def chat(request: Request, req: ChatRequest):
    key = (req.session_token, req.filename)
    if key not in vector_store.indexes:
        raise HTTPException(
            status_code=404,
            detail="Document not found. It may have expired — please re-upload."
        )

    context_chunks = vector_store.retrieve(req.query, req.filename, req.session_token)
    history_dict = [{"role": msg.role, "content": msg.content} for msg in req.history]

    return StreamingResponse(
        generate_response_stream(req.query, context_chunks, history_dict),
        media_type="text/event-stream"
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
