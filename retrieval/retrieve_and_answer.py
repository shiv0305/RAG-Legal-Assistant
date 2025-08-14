# retrieval/retrieve_and_answer.py
"""
Retrieval + answer synthesis module with automatic (semantic) intent detection.
No external intents.json required — intents are built-in and matched to user queries
using SentenceTransformers semantic similarity.

Functions:
- synthesize_answer(query, retrieved_results) -> str
"""

import os
import re
import time
from typing import Optional

# Optional OpenAI v1 client (if you want LLM synthesis)
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

# Optional local generator (transformers)
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

OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo")

# --- Semantic intent detection using sentence-transformers ---
try:
    from sentence_transformers import SentenceTransformer, util
    _INTENT_EMBEDDER = SentenceTransformer(os.environ.get("SBERT_MODEL", "all-MiniLM-L6-v2"))
except Exception:
    _INTENT_EMBEDDER = None

# ------------------------------
# Built-in intents (editable here if desired)
# ------------------------------
# Each intent has: name, description (used for semantic match), extractors (regex list)
INTENTS = [
    {
        "name": "parties",
        "description": "Find the parties to the agreement (who are the parties).",
        "extractors": [
            r"Section\s*\d+\s*[:\-]?\s*Parties\s*[:\-]?\s*(.+?)(?:Section\s*\d+|$)",
            r"([A-Z][A-Za-z0-9&\-\., ]{1,80}?)\s+(?:and|&|, and|,)\s+([A-Z][A-Za-z0-9&\-\., ]{1,80}?)\s+(?:hereby|enter|agree|are)"
        ],
    },
    {
        "name": "which_section",
        "description": "Which section contains a given clause (payment, termination, parties, confidentiality)?",
        "extractors": [
            r"(Section\s*\d+\s*[:\-]\s*[^.\n]+)"
        ],
    },
    {
        "name": "payment_amount",
        "description": "Extract the payment amount (currency like INR, Rs., ₹ and numeric value).",
        "extractors": [
            r"(?:INR|Rs\.?|₹)\s*([0-9][0-9,]*(?:\.\d+)?)",
            r"shall\s+pay\s+.*?(?:INR|Rs\.?|₹)?\s*([0-9][0-9,]*(?:\.\d+)?)"
        ],
    },
    {
        "name": "who_pays",
        "description": "Who is required to make payment (payer / payee)?",
        "extractors": [
            r"([A-Z][\w\-\s\,\.&]{0,120}?)\s+shall\s+pay\s+([A-Z][\w\-\s\,\.&]{0,120}?)"
        ],
    },
    {
        "name": "termination_notice",
        "description": "Find termination or notice period (e.g., 30 days notice).",
        "extractors": [
            r"(\d{1,3})\s*(?:-day|days|'s)?\s*notice",
            r".{0,200}\bterminate\b.{0,200}"
        ],
    },
    {
        "name": "section_content",
        "description": "Return the content of a specific section (e.g., Section 1, Section 3).",
        "extractors": [
            r"(Section\s*\d+\s*[:\-]?\s*[^.\n]+(?:\n|.){0,400})"
        ],
    }
]

# Precompute intent embeddings if embedder is available
_INTENT_EMBS = None
if _INTENT_EMBEDDER is not None:
    try:
        _INTENT_EMBS = _INTENT_EMBEDDER.encode(
            [intent["description"] for intent in INTENTS],
            convert_to_tensor=True,
            normalize_embeddings=True
        )
    except Exception:
        _INTENT_EMBS = None

def _choose_intent_semantic(query: str, threshold: float = 0.45):
    """
    Return the best intent dict (or None) based on semantic similarity between the query
    and intent descriptions. Threshold is adjustable. If no embedder available, fallback
    to keyword detection using simple tokens.
    """
    q = (query or "").strip()
    if not q:
        return None

    # 1) semantic match if embedding model available
    if _INTENT_EMBS is not None and _INTENT_EMBEDDER is not None:
        q_emb = _INTENT_EMBEDDER.encode(q, convert_to_tensor=True, normalize_embeddings=True)
        sims = util.pytorch_cos_sim(q_emb, _INTENT_EMBS)[0]  # tensor of similarities
        best_idx = int(sims.argmax().cpu().numpy())
        best_score = float(sims[best_idx].cpu().numpy())
        if best_score >= threshold:
            return INTENTS[best_idx]
        # if below threshold, still return best match (useful for short queries)
        return INTENTS[best_idx] if best_score >= 0.30 else None

    # 2) fallback simple keyword matching without external file
    q_low = q.lower()
    for intent in INTENTS:
        # check if any important token from description appears in query
        for tok in intent["description"].split():
            if tok.lower().strip(",.") in q_low and len(tok) > 3:
                return intent
    return None

# -------------------------
# Utility helpers (same pattern as previous file)
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

def _search_chunks(docs, metas, ids, pattern: re.Pattern):
    """Return the first match tuple (match_obj, meta, cid, sentence, index) or Nones."""
    for i, text in enumerate(docs):
        if not text:
            continue
        m = pattern.search(text)
        if m:
            match_text = (m.group(1).strip() if m.groups() else m.group(0).strip())
            # find sentence containing match
            sentences = re.split(r'(?<=[\.\?\!])\s+', text.strip())
            sentence = None
            for s in sentences:
                if match_text and match_text in s:
                    sentence = s.strip()
                    break
            if not sentence:
                sentence = sentences[0].strip() if sentences else text.strip()
            meta = metas[i] if metas and len(metas) > i else {}
            cid = ids[i] if ids and len(ids) > i else f"chunk{i}"
            return m, meta, cid, sentence, i
    return None, None, None, None, None

# Simple conflict detection (amounts and notice periods)
def _detect_conflicts(retrieved_results):
    docs = retrieved_results.get("documents", [[]])[0]
    metas = retrieved_results.get("metadatas", [[]])[0]
    ids = retrieved_results.get("ids", [[]])[0]
    amount_pat = re.compile(r"(?:INR|Rs\.?|₹)\s*([0-9][0-9,]*(?:\.\d+)?)", re.IGNORECASE)
    notice_pat = re.compile(r"(\d{1,3})\s*(?:-day|days|'s)?\s*notice", re.IGNORECASE)
    amounts = {}
    notices = {}
    for i, text in enumerate(docs):
        meta = metas[i] if metas and len(metas) > i else {}
        cid = ids[i] if ids and len(ids) > i else f"chunk{i}"
        citation = _format_citation(meta, cid)
        m = amount_pat.search(text)
        if m:
            v = m.group(1).replace(",", "").strip()
            amounts.setdefault(v, []).append(citation)
        n = notice_pat.search(text)
        if n:
            d = n.group(1).lstrip("0") or "0"
            notices.setdefault(d, []).append(citation)
    parts = []
    if len(amounts) > 1:
        parts.append("Payment amount conflicts:")
        for k, v in amounts.items():
            parts.append(f"- INR {k} — {', '.join(v)}")
    if len(notices) > 1:
        parts.append("Termination/notice period conflicts:")
        for k, v in notices.items():
            parts.append(f"- {k} days — {', '.join(v)}")
    return ("\n".join(parts)) if parts else ""

# Core extractive engine (uses matched intent)
def _extractive_answer_from_chunks(retrieved_results, query: Optional[str] = None):
    docs = retrieved_results.get("documents", [[]])[0]
    metas = retrieved_results.get("metadatas", [[]])[0]
    ids = retrieved_results.get("ids", [[]])[0]
    if not docs:
        return "No indexed documents available."

    # automatic intent choice
    intent = _choose_intent_semantic(query or "")
    # if no intent chosen, do simple fallback: attempt a few known patterns
    if intent is None:
        # naive keyword-based fallback
        q = (query or "").lower()
        if any(tok in q for tok in ["payment", "amount", "how much", "how much is"]):
            intent = next((it for it in INTENTS if it["name"] == "payment_amount"), None)
        elif any(tok in q for tok in ["party", "parties", "who are the parties"]):
            intent = next((it for it in INTENTS if it["name"] == "parties"), None)

    # If we have an intent, run its extractors
    if intent is not None:
        for ex in intent.get("extractors", []):
            pat = re.compile(ex, re.IGNORECASE | re.DOTALL)
            m, meta, cid, sent, idx = _search_chunks(docs, metas, ids, pat)
            if m:
                name = intent.get("name")
                # Payment amount
                if name == "payment_amount":
                    try:
                        val = m.group(1)
                        concise = f"Payment amount: INR {val}"
                    except Exception:
                        concise = sent
                    out = f"**Extractive concise answer:**\n{concise}\n\n**Supporting citations / excerpts:**\n1. Citation: {_format_citation(meta, cid)}\n> {sent}\n"
                    return out
                # Who pays
                if name == "who_pays":
                    out = f"**Extractive concise answer:**\n{sent}\n\n**Supporting citations / excerpts:**\n1. Citation: {_format_citation(meta, cid)}\n> {sent}\n"
                    return out
                # Parties
                if name == "parties":
                    if m.groups():
                        if len(m.groups()) >= 2:
                            p1 = m.group(1).strip()
                            p2 = m.group(2).strip()
                            concise = f"{p1} and {p2}"
                        else:
                            concise = m.group(1).strip()
                    else:
                        concise = sent.split(".")[0].strip()
                    out = f"**Extractive concise answer:**\n{concise}\n\n**Supporting citations / excerpts:**\n1. Citation: {_format_citation(meta, cid)}\n> {sent}\n"
                    return out
                # Termination / notice
                if name == "termination_notice":
                    try:
                        days = m.group(1)
                        concise = f"Termination notice period: {days} days."
                    except Exception:
                        concise = sent
                    out = f"**Extractive concise answer:**\n{concise}\n\n**Supporting citations / excerpts:**\n1. Citation: {_format_citation(meta, cid)}\n> {sent}\n"
                    return out
                # Section header / content
                if name in ("which_section", "section_content"):
                    header = m.group(1).strip() if m.groups() else m.group(0).strip()
                    sec_num = re.search(r"(Section\s*\d+)", header, re.IGNORECASE)
                    sec_label = sec_num.group(1) if sec_num else header
                    concise = f"{sec_label}: {header.split(':',1)[-1].strip()}" if ":" in header else header
                    out = f"**Extractive concise answer:**\n{concise}\n\n**Supporting citations / excerpts:**\n1. Citation: {_format_citation(meta, cid)}\n> {header}\n"
                    return out
                # generic fallback for intent
                out = f"**Extractive concise answer:**\n{sent.split('.')[0].strip()}\n\n**Supporting citations / excerpts:**\n1. Citation: {_format_citation(meta, cid)}\n> {sent}\n"
                return out

    # Generic fallback: return top sentence that matches query tokens or first sentence
    top_doc = docs[0]
    sentences = re.split(r'(?<=[\.\?\!])\s+', top_doc.strip())
    chosen = None
    q = (query or "").lower()
    if q:
        q_tokens = [t for t in re.split(r'\W+', q) if t and len(t) > 2]
        for s in sentences:
            s_low = s.lower()
            if any(tok in s_low for tok in q_tokens):
                chosen = s.strip()
                break
    if not chosen:
        chosen = sentences[0].strip() if sentences else top_doc[:200].strip()

    out_lines = ["**Extractive concise answer:**", chosen, "", "**Supporting citations / excerpts:**"]
    for i, text in enumerate(docs):
        meta = metas[i] if metas and len(metas) > i else {}
        cid = ids[i] if ids and len(ids) > i else f"chunk{i}"
        excerpt = text.strip()
        if len(excerpt) > 800:
            excerpt = excerpt[:800].rsplit(" ",1)[0] + " ... (truncated)"
        out_lines.append(f"{i+1}. Citation: {_format_citation(meta, cid)}")
        out_lines.append(f"> {excerpt}\n")
    return "\n".join(out_lines)

# Synthesis: OpenAI / local / extractive + conflict append
def synthesize_answer(query: str, retrieved_results) -> str:
    """
    1) If OpenAI available -> use it
    2) Else if local generator enabled -> use it
    3) Else use extractive fallback
    Append detected conflicts if any.
    """
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
            answer = resp.choices[0].message.content
        except Exception as e:
            answer = f"OpenAI generation failed ({e}). Falling back to extractive output.\n\n" + _extractive_answer_from_chunks(retrieved_results, query)
    elif _USE_LOCAL_GEN and _local_gen is not None:
        try:
            context = build_context_from_retrieval(retrieved_results)
            prompt = f"CONTEXT:\n{context}\nQUESTION: {query}\nAnswer concisely and cite sources in square brackets (e.g. [file | p.1 | section | id])."
            gen = _local_gen(prompt, max_length=200, do_sample=False, num_return_sequences=1)
            txt = gen[0].get("generated_text", "")
            if prompt in txt:
                txt = txt.split(prompt, 1)[-1].strip()
            answer = txt
        except Exception as e:
            answer = f"Local generation failed ({e}). Falling back to extractive output.\n\n" + _extractive_answer_from_chunks(retrieved_results, query)
    else:
        answer = _extractive_answer_from_chunks(retrieved_results, query)

    conflicts_txt = _detect_conflicts(retrieved_results)
    if conflicts_txt:
        answer = answer.rstrip() + "\n\nConflicts:\n" + conflicts_txt
    return answer
