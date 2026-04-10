import os
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict
import json
from dotenv import load_dotenv

from src.extraction import process_file_sync
from src.vector_store import vector_store
from src.llm import generate_response_stream

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOADS_DIR = Path("uploads")
UPLOADS_DIR.mkdir(exist_ok=True)

class Message(BaseModel):
    role: str
    content: str
    
class ChatRequest(BaseModel):
    query: str
    filename: str
    history: List[Message] = []

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")
        
    file_path = UPLOADS_DIR / file.filename
    try:
        content = await file.read()
        file_path.write_bytes(content)
        
        chunks = process_file_sync(str(file_path))
        if not chunks:
            raise HTTPException(status_code=400, detail="Could not extract text from file")
            
        success = vector_store.add_document(file.filename, chunks)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to build vector index")
            
        meta = vector_store.get_doc_meta(file.filename)
        return {"filename": file.filename, "message": "Successfully indexed", "meta": meta}
        
    except Exception as e:
        print(f"Error processing upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
async def chat(req: ChatRequest):
    if req.filename not in vector_store.indexes:
        raise HTTPException(status_code=404, detail="Document not indexed. Upload it first.")
        
    context_chunks = vector_store.retrieve(req.query, req.filename)
    history_dict = [{"role": msg.role, "content": msg.content} for msg in req.history]
    
    return StreamingResponse(
        generate_response_stream(req.query, context_chunks, history_dict), 
        media_type="text/event-stream"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
