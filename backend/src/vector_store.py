try:
    import faiss
    import numpy as np
    import requests
    import os
    FAISS_OK = True
except ImportError:
    FAISS_OK = False

TOP_K = 12

HF_API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"

class VectorStore:
    def __init__(self):
        self.indexes = {} # map filename -> (index, chunks)
        
    def _get_embeddings(self, texts):
        import time
        api_key = os.environ.get("HF_API_KEY")
        if not api_key:
            raise ValueError("HF_API_KEY environment variable is not set. Cannot generate embeddings.")
            
        headers = {"Authorization": f"Bearer {api_key}"}
        
        # Batch texts to avoid HF API payload limits
        BATCH_SIZE = 50
        all_embeddings = []
        
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i + BATCH_SIZE]
            payload = {"inputs": batch, "options": {"wait_for_model": True}}
            
            max_retries = 3
            batch_success = False
            
            for attempt in range(max_retries):
                try:
                    response = requests.post(HF_API_URL, headers=headers, json=payload)
                    response.raise_for_status()
                    embeddings = response.json()
                    all_embeddings.extend(embeddings)
                    batch_success = True
                    # Small delay between successful batches to respect rate limits
                    time.sleep(0.5)
                    break
                except Exception as e:
                    print(f"Error calling HF Inference API for batch {i//BATCH_SIZE} (Attempt {attempt + 1}/{max_retries}): {e}")
                    # Only retry if we haven't exhausted attempts
                    if attempt < max_retries - 1:
                        # For HTTP 429 Too Many Requests or 503 Service Unavailable
                        if hasattr(e, 'response') and e.response is not None and e.response.status_code in (429, 503, 504):
                            wait_time = 2 ** (attempt + 1) # Exponential backoff: 2s, 4s
                            print(f"Rate limited or server busy. Waiting {wait_time}s...")
                            time.sleep(wait_time)
                        else:
                            time.sleep(2)
            
            if not batch_success:
                print(f"FAILED to process batch {i//BATCH_SIZE} after {max_retries} attempts.")
                # Fall back to zero vectors for failed batch
                all_embeddings.extend([[0.0] * 384] * len(batch))
        
        return np.array(all_embeddings, dtype=np.float32)

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
