# RAG — Multi-Document Legal Research Assistant (Starter)

## Overview
This project is a Retrieval-Augmented Generation (RAG) demo for legal documents (contracts, case law, statutes).  
It ingests documents (PDF/DOCX/TXT), chunks them with metadata, builds local sentence-transformer embeddings stored in ChromaDB, retrieves relevant passages for a user query, and synthesizes answers using a local HuggingFace generation model. Outputs include provenance (filename | page | section | chunk_id) and an extractive fallback when generation is unavailable.

> **Note:** This demo uses local models (Sentence-Transformers + HuggingFace pipeline) so no paid API is required. You *can* re-enable OpenAI calls if you have an API key, but it is optional.

---

## Features
- Upload and index PDF / DOCX / TXT via Streamlit UI.
- Chunking with per-chunk metadata (filename, page, section, start/end).
- Local embeddings using `sentence-transformers` (default `all-MiniLM-L6-v2`).
- Vector storage & retrieval with ChromaDB.
- Local generation (Transformers pipeline, default `gpt2`) + extractive fallback.
- Basic evaluation script (`precision@k`) to measure retrieval quality.
- Simple UX to show retrieved excerpts + synthesized answer with citations.

---

## Requirements
Tested on Windows & Linux with Python 3.10+.

Main packages:
- `streamlit`
- `chromadb`
- `sentence-transformers`
- `transformers`
- `torch`
- `pdfplumber`
- `python-docx`
- `tqdm`
- `pandas`

See `requirements.txt` for full pinned versions.

---

## Quick setup (local)

1. Clone repository:
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd rag-legal-assistant
```

2. Create & activate virtual environment:
```bash
python -m venv venv
# Windows
.\venv\Scripts\activate
# macOS / Linux
# source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. (Optional) If you want to use a different local embedding or generation model, set env var:
```bash
# Example: use a different SBERT model
set SBERT_MODEL=paraphrase-MiniLM-L6-v2   # Windows (temporary for session)
# or on macOS/Linux:
# export SBERT_MODEL=paraphrase-MiniLM-L6-v2
```

---

## Run the app

From project root with venv active:
```bash
streamlit run app/app.py
```

Open the displayed URL (usually `http://localhost:8501`) in your browser.

How to use:
1. Upload one or more documents in the left sidebar (PDF / DOCX / TXT).
2. Click **Index uploaded files** to parse, chunk, embed, and store chunks in Chroma.
3. Enter a legal question in the main panel and click **Get answer**.
4. The UI shows retrieved excerpts with provenance and a synthesized (or extractive) answer below.

---

## Evaluation (precision@k)
A basic evaluation script is included at `eval/precision_at_k.py`. It uses hand-labeled query -> expected file mappings in `eval/tests.json`.

Run:
```bash
python -m eval.precision_at_k
```

The script prints per-query retrieval time, a top-k HIT/MISS, Precision@k and average retrieval latency. Save the output to `eval/results.txt` for documentation:
```bash
python -m eval.precision_at_k > eval/results.txt
```

---

## Sample data
You can test with `sample_data/` (if present) or upload your documents via the Streamlit UI. Example sample files used for evaluation in this repo:
- `sample_data/contract_a.txt`
- `sample_data/contract_b.txt`
- `sample_data/sample.txt.txt`

---

## Deployment
You can deploy on Streamlit Cloud (https://share.streamlit.io) or Hugging Face Spaces. Steps for Streamlit Cloud:
1. Push your repo to GitHub.
2. On Streamlit Cloud, **New app** → select repo, branch `main`, and file `app/app.py`.
3. Deploy. (First deploy will download HF models — may take extra time.)

**Note:** No secrets are required for local models. If you later enable OpenAI generation, add your `OPENAI_API_KEY` as a secret in the deployment UI.

---

## Troubleshooting & Tips
- If you see **dimension mismatch** errors in Chroma after switching embedding models, delete `./chroma_db` and re-index:
  ```bash
  rmdir /s /q chroma_db   # Windows
  rm -rf chroma_db        # macOS / Linux
  ```
- If `embeddings` import fails when running scripts from `eval/`, run as a module from repo root:
  ```bash
  python -m eval.precision_at_k
  ```
  or add `sys.path` adjustment in script (already handled in `eval` script included).
- If local generation quality is poor, try a different model in `retrieval/retrieve_and_answer.py`:
  ```python
  from transformers import pipeline
  generator = pipeline("text-generation", model="distilgpt2")
  ```
  Distil models are a good balance of speed and quality.

---

## Files of interest
- `app/app.py` — Streamlit front-end
- `ingestion/parse_files.py` — PDF / DOCX / TXT parsing
- `ingestion/chunking.py` — chunking strategy and metadata
- `embeddings/openai_embed_and_store.py` — local embedding + Chroma init
- `retrieval/retrieve_and_answer.py` — context formatting + local generation
- `utils/utils.py` — high-level ingest helper
- `eval/precision_at_k.py` — evaluation script

---

## License & Author
- Author: Shivendra Shivhare