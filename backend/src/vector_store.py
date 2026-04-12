try:
    import faiss
    import numpy as np
    import requests
    import os
    FAISS_OK = True
except ImportError:
    FAISS_OK = False

TOP_K = 4

HF_API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"

class VectorStore:
    def __init__(self):
        self.indexes = {} # map filename -> (index, chunks)
        
    def _get_embeddings(self, texts):
        api_key = os.environ.get("HF_API_KEY")
        if not api_key:
            print("Warning: HF_API_KEY not set. Using empty embeddings.")
            return np.zeros((len(texts), 384), dtype=np.float32)
            
        headers = {"Authorization": f"Bearer {api_key}"}
        payload = {"inputs": texts, "options": {"wait_for_model": True}}
        
        try:
            response = requests.post(HF_API_URL, headers=headers, json=payload)
            response.raise_for_status()
            embeddings = response.json()
            return np.array(embeddings, dtype=np.float32)
        except Exception as e:
            print(f"Error calling HF Inference API: {e}")
            return np.zeros((len(texts), 384), dtype=np.float32)

    def add_document(self, filename, chunks):
        if not chunks: 
            return False
            
        texts = [c["text"] for c in chunks]
        emb = self._get_embeddings(texts)
        
        faiss.normalize_L2(emb)
        idx = faiss.IndexFlatIP(emb.shape[1])
        idx.add(emb)
        
        self.indexes[filename] = {
            "index": idx,
            "chunks": chunks
        }
        return True

    def retrieve(self, query, filename, top_k=TOP_K):
        doc_data = self.indexes.get(filename)
        if not doc_data:
            return []
            
        index = doc_data["index"]
        chunks = doc_data["chunks"]
        
        if not index or not chunks: 
            return chunks[:top_k]
            
        q = self._get_embeddings([query])
        faiss.normalize_L2(q)
        scores, ids = index.search(q, top_k)
        
        out = []
        for score, i in zip(scores[0], ids[0]):
            if i < len(chunks):
                c = chunks[i].copy()
                c["score"] = float(score)
                out.append(c)
        return out

    def get_doc_meta(self, filename):
        doc_data = self.indexes.get(filename)
        if not doc_data: return None
        chunks = doc_data["chunks"]
        pages = max(c["page"] for c in chunks) if chunks else 0
        return {"chunks_count": len(chunks), "pages": pages}

vector_store = VectorStore()
