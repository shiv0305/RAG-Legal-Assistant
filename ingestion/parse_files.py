# ingestion/parse_files.py
import pdfplumber
from docx import Document
import os

def extract_text_from_pdf(path):
    pages = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            pages.append({"page": i, "text": text})
    return pages

def extract_text_from_docx(path):
    doc = Document(path)
    text = "\n".join([p.text for p in doc.paragraphs])
    return [{"page": 1, "text": text}]

def extract_text_from_txt(path):
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        return [{"page": 1, "text": f.read()}]

def parse_file(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(path)
    elif ext == ".docx":
        return extract_text_from_docx(path)
    elif ext == ".txt":
        return extract_text_from_txt(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
