# ingestion/chunking.py
import re

def simple_heading_extractor(text):
    """
    Very simple heading detection: lines in ALL CAPS or starting with 'Section' / 'Article'.
    Returns a list of (heading, start_char_index)
    """
    headings = []
    for match in re.finditer(r'^(?:Section|Article|CHAPTER|PART|[A-Z][A-Z0-9\s]{5,})', text, flags=re.MULTILINE):
        headings.append((match.group(0).strip(), match.start()))
    return headings

def chunk_text_with_metadata(text, filename, page, chunk_size=1000, overlap=200):
    """
    Returns list of dicts: { "doc_id", "filename", "page", "chunk_id", "start", "end", "text", "section" }
    """
    # Detect headings on page (for simple section metadata)
    headings = simple_heading_extractor(text)
    # fallback single empty heading
    if not headings:
        headings = [("UNKNOWN_SECTION", 0)]

    chunks = []
    start = 0
    n = 0
    text_length = len(text)
    while start < text_length:
        end = min(text_length, start + chunk_size)
        chunk_text = text[start:end].strip()
        # find nearest heading preceding 'start'
        section = "UNKNOWN_SECTION"
        for h, idx in reversed(headings):
            if idx <= start:
                section = h
                break
        chunk = {
            "doc_id": filename,
            "filename": filename,
            "page": page,
            "chunk_id": f"{filename}::p{page}::chunk{n}",
            "start": start,
            "end": end,
            "text": chunk_text,
            "section": section
        }
        chunks.append(chunk)
        n += 1
        start += chunk_size - overlap
    return chunks
