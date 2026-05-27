import os
import secrets
import requests
import time
import numpy as np

try:
    import faiss
except ImportError as e:
    raise RuntimeError(
        "faiss-cpu is not installed. Run: pip install faiss-cpu"
    ) from e

TOP_K = 12
MAX_CHUNKS = 400  # Hard cap per document to prevent OOM

HF_API_URL = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"


class VectorStore:
    def __init__(self):
        # key: (session_token, filename) -> {index, chunks}
        self.indexes = {}

    def _get_embeddings(self, texts):
        api_key = os.environ.get("HF_API_KEY")
        if not api_key:
            raise ValueError("HF_API_KEY environment variable is not set.")

        headers = {"Authorization": f"Bearer {api_key}"}
        BATCH_SIZE = 50
        all_embeddings = []

        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i + BATCH_SIZE]
            payload = {"inputs": batch, "options": {"wait_for_model": True}}
            batch_success = False

            for attempt in range(3):
                try:
                    response = requests.post(HF_API_URL, headers=headers, json=payload)
                    response.raise_for_status()
                    all_embeddings.extend(response.json())
                    batch_success = True
                    time.sleep(0.5)
                    break
                except Exception as e:
                    print(f"[ERROR] HF API batch {i // BATCH_SIZE} attempt {attempt + 1}: {e}")
                    if attempt < 2:
                        wait = 2 ** (attempt + 1)
                        time.sleep(wait)

            if not batch_success:
                print(f"[WARN] Batch {i // BATCH_SIZE} failed — using zero vectors as fallback")
                all_embeddings.extend([[0.0] * 384] * len(batch))

        return np.array(all_embeddings, dtype=np.float32)

    def add_document(self, filename, chunks):
        if not chunks:
            return False, None

        # Enforce chunk cap to prevent OOM on huge documents
        if len(chunks) > MAX_CHUNKS:
            print(f"[WARN] {filename}: {len(chunks)} chunks exceeds cap of {MAX_CHUNKS}. Truncating.")
            chunks = chunks[:MAX_CHUNKS]

        session_token = secrets.token_urlsafe(32)
        texts = [c["text"] for c in chunks]
        emb = self._get_embeddings(texts)

        faiss.normalize_L2(emb)
        idx = faiss.IndexFlatIP(emb.shape[1])
        idx.add(emb)
        del emb  # Free numpy array — FAISS has its own internal copy

        key = (session_token, filename)
        self.indexes[key] = {
            "index": idx,
            "chunks": chunks
        }
        return True, session_token

    def retrieve(self, query, filename, session_token, top_k=TOP_K):
        doc_data = self.indexes.get((session_token, filename))
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

    def get_doc_meta(self, filename, session_token):
        doc_data = self.indexes.get((session_token, filename))
        if not doc_data:
            return None
        chunks = doc_data["chunks"]
        pages = max(c["page"] for c in chunks) if chunks else 0
        return {"chunks_count": len(chunks), "pages": pages}


vector_store = VectorStore()
