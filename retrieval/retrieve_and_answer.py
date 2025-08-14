# retrieval/retrieve_and_answer.py
import os
import time
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
        # CPU by default; set device if you have GPU (not needed on Streamlit Cloud typical)
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
    Build a concise extractive answer using top retrieved chunks.
    """
    docs = retrieved_results.get("documents", [[]])[0]
    metas = retrieved_results.get("metadatas", [[]])[0]
    ids = retrieved_results.get("ids", [[]])[0]

    out_lines = []
    # Short concise answer: try to form from first relevant chunk
    concise = ""
    if docs:
        # try to pick a short sentence that answers the query (best-effort)
        first = docs[0].strip()
        # if there's 'Section 1: Parties' we can extract after that
        if "Section 1" in first or "Parties" in first:
            # heuristics: find line containing "Parties" or the first sentence
            for line in first.splitlines():
                if "Parties" in line or "party" in line.lower():
                    concise = line.strip()
                    break
        if not concise:
            concise = first.split(".")[0].strip()
    if not concise:
        concise = "No concise answer could be extracted from the retrieved excerpts."

    out_lines.append("**Extractive concise answer:**")
    out_lines.append(concise + "\n")

    # Supporting citations (numbered)
    out_lines.append("**Supporting citations / excerpts:**")
    for i, text in enumerate(docs):
        meta = metas[i] if metas and len(metas) > i else {}
        cid = ids[i] if ids and len(ids) > i else f"chunk{i}"
        excerpt = text.strip()
        if len(excerpt) > 800:
            excerpt = excerpt[:800].rsplit(" ",1)[0] + " ... (truncated)"
        out_lines.append(f"{i+1}. Citation: {_format_citation(meta, cid)}")
        out_lines.append(f"> {excerpt}\n")

    out_lines.append("**Note:** This is an extractive answer (no external LLM). Configure an OpenAI key or set USE_LOCAL_GEN to true for a generative answer.")
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
            # call OpenAI Chat Completions (v1 client)
            messages = [{"role": "user", "content": prompt}]
            resp = openai_client.chat.completions.create(model=OPENAI_MODEL, messages=messages, max_tokens=700)
            return resp.choices[0].message.content
        except Exception as e:
            # if OpenAI fails, fallback to extractive
            return f"OpenAI generation failed ({e}). Falling back to extractive output.\n\n" + _extractive_answer_from_chunks(retrieved_results)

    # 2) Local generator path (if configured)
    if _USE_LOCAL_GEN and _local_gen is not None:
        try:
            context = build_context_from_retrieval(retrieved_results)
            # Simple user friendly prompt for local generator
            prompt = f"CONTEXT:\n{context}\nQUESTION: {query}\nAnswer concisely and cite sources in square brackets (e.g. [file | p.1 | section | id])."
            # generate short text
            gen = _local_gen(prompt, max_length=200, do_sample=False, num_return_sequences=1)
            txt = gen[0]["generated_text"]
            # cut prompt echo if generator returns it
            if prompt in txt:
                txt = txt.split(prompt, 1)[-1].strip()
            return txt
        except Exception as e:
            return f"Local generator failed ({e}). Falling back to extractive output.\n\n" + _extractive_answer_from_chunks(retrieved_results)

    # 3) Extractive fallback
    return _extractive_answer_from_chunks(retrieved_results)
