import os
from pathlib import Path

try:
    import pdfplumber
except ImportError:
    pdfplumber = None
try:
    import docx as python_docx
except ImportError:
    python_docx = None

CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 200

def extract_pdf(path):
    pages = []
    if pdfplumber:
        try:
            with pdfplumber.open(path) as pdf:
                for i, page in enumerate(pdf.pages, 1):
                    t = page.extract_text() or ""
                    if t.strip(): pages.append((i, t))
        except Exception as e:
            print(f"Error extracting PDF: {e}")
            raise e
    else:
        raise ImportError("pdfplumber is not installed.")
    return pages

def extract_docx(path):
    if python_docx:
        try:
            doc = python_docx.Document(path)
            txt = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
            return [(1, txt)]
        except Exception as e:
            print(f"Error extracting DOCX: {e}")
            raise e
    else:
        raise ImportError("python-docx is not installed.")
    return []

def extract_txt(path):
    try:
        return [(1, Path(path).read_text(encoding="utf-8", errors="ignore"))]
    except Exception as e:
        print(f"Error extracting TXT: {e}")
        raise e
    return []

def chunk_pages(pages, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    for page_num, text in pages:
        text = text.replace("\n", " ").strip()
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            c = text[start:end].strip()
            if len(c) > 20:
                chunks.append({"text": c, "page": page_num, "chunk_id": len(chunks)})
            start += chunk_size - overlap
    return chunks

def process_file_sync(path):
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        pages = extract_pdf(path)
    elif ext == ".docx":
        pages = extract_docx(path)
    else:
        pages = extract_txt(path)
        
    if not pages: return []
    chunks = chunk_pages(pages)
    return chunks
