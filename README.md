# 📓 Folio
Built a document Q&A system using Retrieval-Augmented Generation (RAG) that allows users to upload PDFs, DOCX, and TXT files and ask natural language questions over them. Used FAISS for vector similarity search, sentence-transformers for embeddings, and Groq (LLaMA 3.1) as the generation backend. Solves the problem of querying large private documents without expensive fine-tuning or exceeding LLM context limits.

## 🛠️ Core Stack
* **Frontend**: React (Vite) + TypeScript
* **Backend**: FastAPI (Python)
* **Vector Store**: FAISS
* **LLM**: LLaMA 3.1 via Groq API

## 🚀 Quick Start

**1. Clone and Install Backend**
```bash
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**2. Add your API Keys**
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_api_key_here
```

**3. Install Frontend and Run**
```bash
cd frontend
npm install
npm run dev
```

**4. Start the Python Backend** *(In a separate terminal)*
```bash
cd backend
uvicorn main:app --reload
```
