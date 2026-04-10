try:
    from sentence_transformers import SentenceTransformer
    import faiss
    import numpy as np
    FAISS_OK = True
except ImportError:
    FAISS_OK = False

TOP_K = 4

class VectorStore:
    def __init__(self):
        self.model = self._load_embed_model()
        self.indexes = {} # map filename -> (index, chunks)
        
    def _load_embed_model(self):
        if FAISS_OK:
            return SentenceTransformer("all-MiniLM-L6-v2")
        return None

    def add_document(self, filename, chunks):
        if not self.model or not chunks: 
            return False
        
        emb = self.model.encode([c["text"] for c in chunks], convert_to_numpy=True, show_progress_bar=False)
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
            
        q = self.model.encode([query], convert_to_numpy=True, show_progress_bar=False)
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
