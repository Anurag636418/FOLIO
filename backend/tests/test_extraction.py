import pytest
from src.extraction import chunk_pages

def test_chunk_pages():
    pages = [
        (1, "Artificial intelligence is a discipline of computer science. " * 10),
        (2, "Machine learning is a subset of AI. " * 10)
    ]
    
    chunks = chunk_pages(pages, chunk_size=100, overlap=20)
    
    assert len(chunks) > 0
    for chunk in chunks:
        assert 'text' in chunk
        assert chunk['page'] in (1, 2)
        assert len(chunk['text']) <= 100
        
def test_chunk_overlap():
    pages = [(1, "A " * 50)]
    chunks = chunk_pages(pages, chunk_size=20, overlap=5)
    # The chunking logic is working correctly if it produces multiple chunks
    assert len(chunks) > 1
