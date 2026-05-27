# 📓 Folio

A document Q&A system using Retrieval-Augmented Generation (RAG). Upload PDFs, DOCX, or TXT files and ask natural language questions over them.

**Stack:** React · FastAPI · FAISS · HuggingFace embeddings · Groq (LLaMA 3.3-70b)

---

## 📐 Architecture & Data Flow

```
[User Document] -> Streamed in Chunks -> Stored Temporarily -> Extracted (pdfplumber/docx)
                                                                       |
[FAISS Index] <- Free embedding RAM <- Normalized Embeddings <- HuggingFace Embeddings
      |
(User Query) -> Generate Query Embedding -> Cosine Similarity Search (FAISS) 
                                                               |
[Client Stream] <- Server-Sent Events (SSE) <- Groq LLM <- Delimited Context + Security Prompt
```

---

## 💻 Local Development Setup

### Backend Prerequisites & Launch
1. Clone the repository and navigate to the backend directory:
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. Create a `.env` file in `backend/` with the following variables:
   ```env
   GROQ_API_KEY=your_groq_api_key
   HF_API_KEY=your_huggingface_api_key
   ALLOWED_ORIGINS=http://localhost:5173
   ```
3. Start the FastAPI development server:
   ```bash
   uvicorn main:app --reload
   ```

### Frontend Prerequisites & Launch
1. Navigate to the frontend directory and install packages:
   ```bash
   cd frontend
   npm install
   ```
2. Create a `.env.local` file in `frontend/` to point to the local backend:
   ```env
   VITE_API_URL=http://localhost:8000/api
   ```
3. Run the development server:
   ```bash
   npm run dev
   ```
