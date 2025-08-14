# retrieval/retrieve_and_answer.py
import os
import time
import re
from typing import List

# --- OpenAI v1 client (optional) ---
openai_client = None
_use_openai = False
try:
    from openai import OpenAI
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        openai_client = OpenAI(api_key=api_key)
        _use_openai = True
except Exception:
    _use_openai = False

# --- Optional local generator (transformers) ---
_USE_LOCAL_GEN = os.environ.get("USE_LOCAL_GEN", "false").lower() == "true"
_local_gen = None
if _USE_LOCAL_GEN:
    try:
        from transformers import pipeline
        local_model = os.environ.get("LOCAL_GEN_MODEL", "distilgpt2")
        _local_gen = pipeline("text-generation", model=local_model)
    except Exception:
        _local_gen = None
        _USE_LOCAL_GEN = False

# Default model name for OpenAI generation if used
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo")

# -------------------------
# Formatting & helpers
# -------------------------
def build_context_from_retrieval(results) -> str:
    blocks = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    ids = results.get("ids", [[]])[0]
    for idx, doc_text in enumerate(docs):
        meta = metas[idx] if metas and len(metas) > idx else {}
        chunk_id = ids[idx] if ids and len(ids) > idx else f"chunk{idx}"
        block = (
            f"Source: {meta.get('filename','unknown')}  | page {meta.get('page','?')}  | "
            f"section: {meta.get('section','?')}  | id: {chunk_id}\n"
            f"{doc_text}\n----\n"
        )
        blocks.append(block)
    return "\n".join(blocks)

def _format_citation(meta, chunk_id):
    return f"[{meta.get('filename','?')} | p.{meta.get('page','?')} | {meta.get('section','?')} | {chunk_id}]"

def _extractive_answer_from_chunks(retrieved_results, max_chars=1200):
    """
    Improved extractive answer:
      - Tries structured extraction for 'Parties' sections and common legal phrasing.
      - Falls back to the original simple first-sentence approach if nothing matched.
    """
    docs = retrieved_results.get("documents", [[]])[0]
    metas = retrieved_results.get("metadatas", [[]])[0]
    ids = retrieved_results.get("ids", [[]])[0]

    out_lines = []

    # 1) Try targeted extraction for party names or clause text
    party_text = None
    party_meta = None
    party_id = None

    # patterns to try (ordered)
    patterns = [
        # Section N: Parties ... (capture everything until next "Section" or end)
        re.compile(r"Section\s*\d+\s*[:\-]?\s*Parties\s*[:\-]?\s*(.+?)(?:Section\s*\d+|$)", re.IGNORECASE | re.DOTALL),
        # "Parties" header without "Section", capture trailing sentence(s)
        re.compile(r"Parties\s*[:\-]?\s*(.+?)(?:Section\s*\d+|$|\n\n)", re.IGNORECASE | re.DOTALL),
        # "<Name> and <Name> hereby" or "<Name> and <Name> agree"
        re.compile(r"([A-Z][A-Za-z0-9&\-,\. ]{1,80}?(?: and | & |, and |, )+[A-Z][A-Za-z0-9&\-,\. ]{1,80}?) (?:hereby|agree|enter|are|shall)", re.IGNORECASE),
        # simple "X shall pay Y" (fallback patterns — not parties but may help)
        re.compile(r"([A-Z][A-Za-z0-9&\-,\. ]{1,120}?) (?:shall pay|shall provide|shall deliver|agrees to)", re.IGNORECASE),
    ]

    for i, text in enumerate(docs):
        if not text:
            continue
        for pat in patterns:
            m = pat.search(text)
            if m:
                party_text = m.group(1).strip()
                party_meta = metas[i] if metas and len(metas) > i else {}
                party_id = ids[i] if ids and len(ids) > i else f"chunk{i}"
                break
        if party_text:
            break

    # 2) Build concise answer
    if party_text:
        # Clean up whitespace and trailing artifacts
        party_text = re.sub(r"\s+", " ", party_text).strip()
        # If the result looks like a short header (e.g., just "Parties"), try to get additional context/sentence
        if len(party_text.split()) <= 3 and docs:
            # try next sentence in the chunk we found or the first retrieved doc
            source_doc = docs[0]
            sentences = re.split(r'(?<=[\.\?\!])\s+', source_doc)
            if len(sentences) > 1 and len(sentences[0].strip()) > 2:
                concise = sentences[0].strip()
            else:
                concise = party_text
        else:
            concise = party_text
    else:
        # fallback: use first sentence of first retrieved doc
        concise = ""
        if docs:
            first = docs[0].strip()
            sentences = re.split(r'(?<=[\.\?\!])\s+', first)
            if sentences and sentences[0].strip():
                concise = sentences[0].strip()
            else:
                concise = first[:200].strip()
        if not concise:
            concise = "No concise answer could be extracted from the retrieved excerpts."

    # 3) Compose output
    out_lines.append("**Extractive concise answer:**")
    out_lines.append(concise + "\n")

    out_lines.append("**Supporting citations / excerpts:**")
    for i, text in enumerate(docs):
        meta = metas[i] if metas and len(metas) > i else {}
        cid = ids[i] if ids and len(ids) > i else f"chunk{i}"
        excerpt = text.strip()
        if len(excerpt) > 800:
            excerpt = excerpt[:800].rsplit(" ",1)[0] + " ... (truncated)"
        out_lines.append(f"{i+1}. Citation: {_format_citation(meta, cid)}")
        out_lines.append(f"> {excerpt}\n")

    # out_lines.append("**Note:** This is an extractive answer (no external LLM). Configure an OpenAI key or set USE_LOCAL_GEN to true for a generative answer.")
    return "\n".join(out_lines)

# -------------------------
# Synthesis: OpenAI / local / extractive
# -------------------------
def synthesize_answer(query: str, retrieved_results) -> str:
    """
    Generate a synthesized answer. Priority:
    1) Use OpenAI if configured
    2) Else use local generator if enabled
    3) Else return extractive fallback
    """

    # 1) OpenAI path (if available)
    if _use_openai and openai_client is not None:
        try:
            context = build_context_from_retrieval(retrieved_results)
            prompt = f"""
You are a legal research assistant. Use ONLY the provided CONTEXT excerpts to answer the user's question.
Cite each factual claim with the source tag: [filename | page | section | id].
If two or more sources conflict, explicitly state each conflicting claim and cite the sources that support it.

CONTEXT:
{context}

QUESTION:
{query}

Provide:
1) A concise answer (2-6 sentences).
2) A numbered list of supporting citations with exact excerpt(s).
3) If conflicts exist, a 'Conflicts' section.
"""
            messages = [{"role": "user", "content": prompt}]
            resp = openai_client.chat.completions.create(model=OPENAI_MODEL, messages=messages, max_tokens=700)
            return resp.choices[0].message.content
        except Exception as e:
            return f"OpenAI generation failed ({e}). Falling back to extractive output.\n\n" + _extractive_answer_from_chunks(retrieved_results)

    # 2) Local generator path (if configured)
    if _USE_LOCAL_GEN and _local_gen is not None:
        try:
            context = build_context_from_retrieval(retrieved_results)
            prompt = f"CONTEXT:\n{context}\nQUESTION: {query}\nAnswer concisely and cite sources in square brackets (e.g. [file | p.1 | section | id])."
            gen = _local_gen(prompt, max_length=200, do_sample=False, num_return_sequences=1)
            txt = gen[0]["generated_text"]
            if prompt in txt:
                txt = txt.split(prompt, 1)[-1].strip()
            return txt
        except Exception as e:
            return f"Local generator failed ({e}). Falling back to extractive output.\n\n" + _extractive_answer_from_chunks(retrieved_results)

    # 3) Extractive fallback
    return _extractive_answer_from_chunks(retrieved_results)
