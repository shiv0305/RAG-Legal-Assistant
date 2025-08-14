# utils/utils.py
import os
from ingestion.parse_files import parse_file
from ingestion.chunking import chunk_text_with_metadata
from embeddings.openai_embed_and_store import add_chunks_to_chroma

def ingest_and_index_file(path):
    pages = parse_file(path)
    all_chunks = []
    filename = os.path.basename(path)
    for p in pages:
        page_num = p["page"]
        text = p["text"] or ""
        chunks = chunk_text_with_metadata(text, filename, page_num)
        all_chunks.extend(chunks)
    add_chunks_to_chroma(all_chunks)
    return len(all_chunks)
