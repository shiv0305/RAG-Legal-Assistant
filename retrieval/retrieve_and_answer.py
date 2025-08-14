# retrieval/retrieve_and_answer.py
import os
import re
import json
from typing import List, Optional

# --- Optional OpenAI v1 client (if you want LLM synthesis) ---
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

OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo")


# ----------------------
# Intent config loader
# ----------------------
INTENTS_PATH = os.path.join("intents", "intents.json")

DEFAULT_INTENTS = [
    {
        "name": "which_section",
        "match_keywords": ["which section", "what section", "section contains", "which clause"],
        # attempt to find section headers containing the target word (see code logic)
        "extractors": [
            r"(Section\s*\d+\s*[:\-]\s*[^.\n]+)"   # generic section header
        ],
        "notes": "Return Section N: Header if header contains target word from query"
    },
    {
        "name": "payment_amount",
        "match_keywords": ["payment amount", "amount", "how much", "payment", "amount payable"],
        "extractors": [
            r"(?:INR|Rs\.?|₹)\s*([0-9][0-9,]*(?:\.\d+)?)",
            r"shall\s+pay\s+.*?(?:INR|Rs\.?|₹)?\s*([0-9][0-9,]*(?:\.\d+)?)"
        ],
        "notes": "Extract numeric currency amounts"
    },
    {
        "name": "who_pays",
        "match_keywords": ["who pays", "who will pay", "who shall pay", "payer"],
        "extractors": [
            r"([A-Z][\w\-\s\,\.&]{0,120}?)\s+shall\s+pay\s+([A-Z][\w\-\s\,\.&]{0,120}?)"
        ]
    },
    {
        "name": "termination_notice",
        "match_keywords": ["termination", "notice period", "terminate", "notice", "termination notice"],
        "extractors": [
            r"(\d{1,3})\s*(?:-day|days|'s)?\s*notice",
            r".{0,200}\bterminate\b.{0,200}"
        ]
    },
    {
        "name": "parties",
        "match_keywords": ["who are the parties", "parties", "who are the parties in the agreement", "party"],
        "extractors": [
            r"Section\s*\d+\s*[:\-]?\s*Parties\s*[:\-]?\s*(.+?)(?:Section\s*\d+|$)",
            r"([A-Z][A-Za-z0-9&\-\., ]{1,80}?)\s+(?:and|&|, and|,)\s+([A-Z][A-Za-z0-9&\-\., ]{1,80}?)\s+(?:hereby|enter|agree|are)"
        ]
    }
]

def load_intents():
    if os.path.exists(INTENTS_PATH):
        try:
            with open(INTENTS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list) and data:
                return data
        except Exception:
            pass
    # fallback
    return DEFAULT_INTENTS

INTENTS = load_intents()


# -------------------------
# Helpers
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

# -------------------------
# Conflict detection (simple numeric claims)
# -------------------------
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

# -------------------------
# Core extraction using intents
# -------------------------
def _extractive_answer_from_chunks(retrieved_results, query: Optional[str] = None):
    docs = retrieved_results.get("documents", [[]])[0]
    metas = retrieved_results.get("metadatas", [[]])[0]
    ids = retrieved_results.get("ids", [[]])[0]
    if not docs:
        return "No indexed documents available."

    q = (query or "").lower()
    # find matching intent by keywords first (configurable)
    matched_intent = None
    for intent in INTENTS:
        kws = intent.get("match_keywords", [])
        if any(kw in q for kw in kws):
            matched_intent = intent
            break
    # if none matched, try fuzzy match by scanning for common words from intents
    if matched_intent is None:
        for intent in INTENTS:
            for kw in intent.get("match_keywords", []):
                if kw.split()[0] in q:  # partial token match
                    matched_intent = intent
                    break
            if matched_intent:
                break

    # extraction workflow
    if matched_intent:
        extractors = matched_intent.get("extractors", [])
        # when handling "which_section" specially, we also consider target words from query
        if matched_intent.get("name") == "which_section":
            # attempt to find section header that includes the clause term from query
            target_words = ["payment", "terminate", "termination", "parties", "notice"]
            found_target = None
            for t in target_words:
                if t in q:
                    found_target = t
                    break
            # search for Section header patterns, prefer those that mention target
            header_pat = re.compile(r"(Section\s*\d+\s*[:\-]\s*[^.\n]+)", re.IGNORECASE)
            m, meta, cid, sent, idx = _search_chunks(docs, metas, ids, header_pat)
            if m:
                header = m.group(1)
                if (found_target and found_target in header.lower()) or found_target is None:
                    sec_num = re.search(r"(Section\s*\d+)", header, re.IGNORECASE)
                    sec_label = sec_num.group(1) if sec_num else header
                    concise = f"{sec_label}: {header.split(':',1)[-1].strip()}" if ":" in header else header
                    out = f"**Extractive concise answer:**\n{concise}\n\n**Supporting citations / excerpts:**\n1. Citation: {_format_citation(meta, cid)}\n> {sent}\n"
                    return out
            return "No explicit section header found in the retrieved excerpts that matches the requested clause."
        # for other intents, run configured extractor regexes in order
        for ex in extractors:
            pat = re.compile(ex, re.IGNORECASE)
            m, meta, cid, sent, idx = _search_chunks(docs, metas, ids, pat)
            if m:
                # build good concise answer based on intent name
                name = matched_intent.get("name")
                if name == "payment_amount":
                    # group 1 expected numeric; fallback to full sentence
                    val = None
                    try:
                        val = m.group(1)
                    except Exception:
                        val = None
                    concise = f"Payment amount: INR {val}" if val else sent
                    out = f"**Extractive concise answer:**\n{concise}\n\n**Supporting citations / excerpts:**\n1. Citation: {_format_citation(meta, cid)}\n> {sent}\n"
                    return out
                elif name == "who_pays":
                    # return the enclosing sentence
                    out = f"**Extractive concise answer:**\n{sent}\n\n**Supporting citations / excerpts:**\n1. Citation: {_format_citation(meta, cid)}\n> {sent}\n"
                    return out
                elif name == "termination_notice":
                    days = None
                    try:
                        days = m.group(1)
                    except Exception:
                        days = None
                    concise = f"Termination notice period: {days} days." if days else sent
                    out = f"**Extractive concise answer:**\n{concise}\n\n**Supporting citations / excerpts:**\n1. Citation: {_format_citation(meta, cid)}\n> {sent}\n"
                    return out
                elif name == "parties":
                    # for parties extractor might capture the party names
                    if m.groups():
                        # join groups if two groups represent parties
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
                else:
                    # generic return
                    out = f"**Extractive concise answer:**\n{sent.split('.')[0].strip()}\n\n**Supporting citations / excerpts:**\n1. Citation: {_format_citation(meta, cid)}\n> {sent}\n"
                    return out

    # Generic fallback: return top sentence matching query tokens, else first sentence
    top_doc = docs[0]
    sentences = re.split(r'(?<=[\.\?\!])\s+', top_doc.strip())
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

# -------------------------
# Synthesis wrapper
# -------------------------
def synthesize_answer(query: str, retrieved_results) -> str:
    """
    1. If OpenAI configured => send context + prompt (LLM synth).
    2. Else if local gen enabled => generate locally.
    3. Else use extractive engine above.
    After main answer, append conflict section (if any).
    """
    answer = None
    # OpenAI path
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

    # detect conflicts and append
    conflicts = _detect_conflicts(retrieved_results)
    if conflicts:
        answer = answer.rstrip() + "\n\nConflicts:\n" + conflicts
    return answer
