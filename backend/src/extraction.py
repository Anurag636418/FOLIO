from pathlib import Path

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    import docx as python_docx
except ImportError:
    python_docx = None

CHUNK_SIZE = 750
CHUNK_OVERLAP = 100
MAX_PDF_PAGES = 80  # Hard cap to prevent OOM on large PDFs


def extract_pdf(path):
    pages = []
    if not pdfplumber:
        raise ImportError("pdfplumber is not installed.")
    try:
        with pdfplumber.open(path) as pdf:
            total = len(pdf.pages)
            if total > MAX_PDF_PAGES:
                print(f"[WARN] PDF has {total} pages — indexing first {MAX_PDF_PAGES} only.")
            for i, page in enumerate(pdf.pages[:MAX_PDF_PAGES], 1):
                t = page.extract_text() or ""
                if t.strip():
                    pages.append((i, t))
    except Exception as e:
        print(f"[ERROR] extract_pdf: {e}")
        raise
    return pages


def extract_docx(path):
    if not python_docx:
        raise ImportError("python-docx is not installed.")
    try:
        doc = python_docx.Document(path)
        txt = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        return [(1, txt)]
    except Exception as e:
        print(f"[ERROR] extract_docx: {e}")
        raise


def extract_txt(path):
    try:
        return [(1, Path(path).read_text(encoding="utf-8", errors="ignore"))]
    except Exception as e:
        print(f"[ERROR] extract_txt: {e}")
        raise


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

    if not pages:
        return []
    return chunk_pages(pages)
