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

# -------------------------
# Improved extractive answer with intent-detection
# -------------------------
def _extractive_answer_from_chunks(retrieved_results, query: str = None, max_chars=1200):
    """
    Improved extractive answer:
      - Intent-detects common legal questions (payment amount, who pays, parties, termination).
      - Uses targeted regex extraction from top retrieved chunks.
      - Falls back to first-sentence extractive behavior otherwise.
    """
    docs = retrieved_results.get("documents", [[]])[0]
    metas = retrieved_results.get("metadatas", [[]])[0]
    ids = retrieved_results.get("ids", [[]])[0]

    out_lines = []

    # Normalize query
    q = (query or "").lower()

    # Helper: search chunks for a regex and return (match_text, meta, id, full_sentence)
    def search_chunks_for_regex(pattern):
        for i, text in enumerate(docs):
            if not text:
                continue
            m = pattern.search(text)
            if m:
                # prefer capturing group 1 if present
                match_txt = m.group(1).strip() if m.groups() else m.group(0).strip()
                # find enclosing sentence for context
                sentences = re.split(r'(?<=[\.\?\!])\s+', text.strip())
                sent = None
                for s in sentences:
                    if match_txt in s:
                        sent = s.strip()
                        break
                if not sent:
                    sent = text.strip().split(".")[0].strip()
                meta = metas[i] if metas and len(metas) > i else {}
                cid = ids[i] if ids and len(ids) > i else f"chunk{i}"
                return match_txt, meta, cid, sent
        return None, None, None, None

    # 1) Payment amount intent
    if any(tok in q for tok in ["payment amount", "amount", "how much", "payment", "amount payable"]):
        # currency patterns: INR, Rs., ₹ or numeric plus currency words
        pat_currency = re.compile(r"(?:INR|Rs\.?|₹)\s*([0-9][0-9,]*(?:\.\d+)?)", re.IGNORECASE)
        match_txt, meta, cid, sent = search_chunks_for_regex(pat_currency)
        if match_txt:
            concise = f"{meta.get('filename','unknown')} payment amount: {('INR ' + match_txt) if not match_txt.lower().startswith(('inr','rs','₹')) else match_txt}"
            out_lines.append("**Extractive concise answer:**")
            out_lines.append(concise + "\n")
            out_lines.append("**Supporting citations / excerpts:**")
            out_lines.append(f"1. Citation: {_format_citation(meta, cid)}")
            out_lines.append(f"> {sent}\n")
            return "\n".join(out_lines)

        # fallback: look for patterns like "Beta shall pay Alpha INR 1,00,000..."
        pat_pay_sentence = re.compile(r"([A-Z][\w\-\s\,\.&]{0,120}?)\s+shall\s+pay\s+([A-Z][\w\-\s\,\.&]{0,120}?).*?(?:INR|Rs\.?|₹)?\s*([0-9][0-9,]*(?:\.\d+)?)", re.IGNORECASE)
        match_txt, meta, cid, sent = search_chunks_for_regex(pat_pay_sentence)
        if match_txt:
            # the group logic in search function returns group(1) — to be safe, re-run on match chunk
            m = pat_pay_sentence.search(docs[0]) if docs else None
            # Instead, just use the sentence found
            concise = sent
            out_lines.append("**Extractive concise answer:**")
            out_lines.append(concise + "\n")
            out_lines.append("**Supporting citations / excerpts:**")
            out_lines.append(f"1. Citation: {_format_citation(meta, cid)}")
            out_lines.append(f"> {sent}\n")
            return "\n".join(out_lines)

    # 2) Who pays whom / payer payee intent
    if any(tok in q for tok in ["who pays", "who will pay", "payer", "pay whom", "who pays whom"]):
        pat = re.compile(r"([A-Z][\w\-\s\,\.&]{0,120}?)\s+shall\s+pay\s+([A-Z][\w\-\s\,\.&]{0,120}?)", re.IGNORECASE)
        match_txt, meta, cid, sent = search_chunks_for_regex(pat)
        if match_txt:
            out_lines.append("**Extractive concise answer:**")
            out_lines.append(sent + "\n")
            out_lines.append("**Supporting citations / excerpts:**")
            out_lines.append(f"1. Citation: {_format_citation(meta, cid)}")
            out_lines.append(f"> {sent}\n")
            return "\n".join(out_lines)

    # 3) Termination / notice period intent
    if any(tok in q for tok in ["termination", "notice period", "terminate", "notice"]):
        # find e.g., "30 days' notice" or "30 days notice" or "30-day notice"
        pat_days = re.compile(r"(\d{1,3})\s*(?:-day|days|'s)?\s*notice", re.IGNORECASE)
        match_txt, meta, cid, sent = search_chunks_for_regex(pat_days)
        if match_txt:
            concise = f"Termination notice period: {match_txt} days (as stated in excerpt)."
            out_lines.append("**Extractive concise answer:**")
            out_lines.append(concise + "\n")
            out_lines.append("**Supporting citations / excerpts:**")
            out_lines.append(f"1. Citation: {_format_citation(meta, cid)}")
            out_lines.append(f"> {sent}\n")
            return "\n".join(out_lines)

        # fallback: sentence containing 'terminate'
        pat_term = re.compile(r".{0,200}\bterminate\b.{0,200}", re.IGNORECASE)
        match_txt, meta, cid, sent = search_chunks_for_regex(pat_term)
        if match_txt:
            out_lines.append("**Extractive concise answer:**")
            out_lines.append(sent + "\n")
            out_lines.append("**Supporting citations / excerpts:**")
            out_lines.append(f"1. Citation: {_format_citation(meta, cid)}")
            out_lines.append(f"> {sent}\n")
            return "\n".join(out_lines)

    # 4) Parties intent
    if any(tok in q for tok in ["who are the parties", "parties", "who are the parties in", "who are the parties in the agreement", "party"]):
        # Section style or inline "Alpha and Beta hereby enter..."
        pat_section_parties = re.compile(r"Section\s*\d+\s*[:\-]?\s*Parties\s*[:\-]?\s*(.+?)(?:Section\s*\d+|$)", re.IGNORECASE | re.DOTALL)
        pat_inline = re.compile(r"([A-Z][A-Za-z0-9&\-\., ]{1,80}?)\s+(?:and|&|, and)\s+([A-Z][A-Za-z0-9&\-\., ]{1,80}?)\s+(?:hereby|enter|agree|are)", re.IGNORECASE)
        # try section pattern first
        match_txt, meta, cid, sent = search_chunks_for_regex(pat_section_parties)
        if match_txt:
            # clean header artifacts
            cleaned = re.sub(r"\s+", " ", match_txt).strip()
            concise = cleaned
            out_lines.append("**Extractive concise answer:**")
            out_lines.append(concise + "\n")
            out_lines.append("**Supporting citations / excerpts:**")
            out_lines.append(f"1. Citation: {_format_citation(meta, cid)}")
            out_lines.append(f"> {sent}\n")
            return "\n".join(out_lines)

        match_txt, meta, cid, sent = search_chunks_for_regex(pat_inline)
        if match_txt:
            concise = match_txt
            out_lines.append("**Extractive concise answer:**")
            out_lines.append(concise + "\n")
            out_lines.append("**Supporting citations / excerpts:**")
            out_lines.append(f"1. Citation: {_format_citation(meta, cid)}")
            out_lines.append(f"> {sent}\n")
            return "\n".join(out_lines)

    # 5) Generic: try to answer small factual queries by extracting the most relevant short sentence
    if docs:
        # pick the top doc and try to extract a short sentence likely answering
        top_doc = docs[0]
        sentences = re.split(r'(?<=[\.\?\!])\s+', top_doc.strip())
        # select the first sentence that contains a token from the query (if query provided)
        chosen = None
        if q:
            q_tokens = [t for t in re.split(r'\W+', q) if t and len(t) > 2]
            for s in sentences:
                s_low = s.lower()
                if any(tok in s_low for tok in q_tokens):
                    chosen = s.strip()
                    break
        if not chosen:
            chosen = sentences[0].strip() if sentences else top_doc[:200].strip()

        out_lines.append("**Extractive concise answer:**")
        out_lines.append(chosen + "\n")

        out_lines.append("**Supporting citations / excerpts:**")
        for i, text in enumerate(docs):
            meta = metas[i] if metas and len(metas) > i else {}
            cid = ids[i] if ids and len(ids) > i else f"chunk{i}"
            excerpt = text.strip()
            if len(excerpt) > 800:
                excerpt = excerpt[:800].rsplit(" ",1)[0] + " ... (truncated)"
            out_lines.append(f"{i+1}. Citation: {_format_citation(meta, cid)}")
            out_lines.append(f"> {excerpt}\n")
        return "\n".join(out_lines)

    # If nothing matched and no docs:
    return "No relevant excerpts found in the indexed documents."

# -------------------------
# Synthesis: OpenAI / local / extractive
# -------------------------
def synthesize_answer(query: str, retrieved_results) -> str:
    """
    Generate a synthesized answer. Priority:
    1) Use OpenAI if configured
    2) Else use local generator if enabled
    3) Else return extractive fallback (now uses query-aware extraction)
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
            return f"OpenAI generation failed ({e}). Falling back to extractive output.\n\n" + _extractive_answer_from_chunks(retrieved_results, query)

    # 2) Local generator path (if configured)
    if _USE_LOCAL_GEN and _local_gen is not None:
        try:
            context = build_context_from_retrieval(retrieved_results)
            prompt = f"CONTEXT:\n{context}\nQUESTION: {query}\nAnswer concisely and cite sources in square brackets (e.g. [file | p.1 | section | id])."
            gen = _local_gen(prompt, max_length=200, do_sample=False, num_return_sequences=1)
            txt = gen[0].get("generated_text", "")
            if prompt in txt:
                txt = txt.split(prompt, 1)[-1].strip()
            return txt
        except Exception as e:
            return f"Local generator failed ({e}). Falling back to extractive output.\n\n" + _extractive_answer_from_chunks(retrieved_results, query)

    # 3) Extractive fallback (query-aware)
    return _extractive_answer_from_chunks(retrieved_results, query)
