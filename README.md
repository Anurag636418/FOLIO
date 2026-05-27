# 📓 Folio

A document Q&A system using Retrieval-Augmented Generation (RAG). Upload PDFs, DOCX, or TXT files and ask natural language questions over them. Uses FAISS for vector search, HuggingFace Inference API for embeddings, and Groq (LLaMA 3.3-70b) as the LLM.

## Stack

- **Frontend**: React + Vite + TypeScript → deployed on Vercel
- **Backend**: FastAPI (Python) → deployed on Render
- **Vector Store**: FAISS (in-memory)
- **Embeddings**: HuggingFace Inference API (`all-MiniLM-L6-v2`)
- **LLM**: LLaMA 3.3-70b via Groq API (Gemini 2.0 Flash as fallback)

## Local Development

### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create `backend/.env`:
```env
GROQ_API_KEY=your_groq_key
HF_API_KEY=your_huggingface_key
GEMINI=your_gemini_key          # optional fallback
ALLOWED_ORIGINS=http://localhost:5173
```

```bash
uvicorn main:app --reload
```

### Frontend

```bash
cd frontend
npm install
```

Create `frontend/.env.local`:
```env
VITE_API_URL=http://localhost:8000/api
```

```bash
npm run dev
```

## Deployment

### Render (backend)
Set these environment variables in the Render dashboard:
- `GROQ_API_KEY`
- `HF_API_KEY`
- `ALLOWED_ORIGINS=https://folio-omega-pied.vercel.app`
- `GEMINI` (optional)

### Vercel (frontend)
Set this environment variable in Vercel project settings:
- `VITE_API_URL=https://folio-backend-b20p.onrender.com/api`

## Limits (free tier)
- Max file size: 10MB
- Max PDF pages indexed: 80
- Max chunks per document: 400
- Upload rate limit: 5 per minute per IP
- Chat rate limit: 30 per minute per IP
