# retrieval/retrieve_and_answer.py
from typing import List
from transformers import pipeline

# Load a local model for generation (you can change to a bigger one if GPU available)
generator = pipeline("text-generation", model="gpt2")  # Change to legal-tuned model if needed

def build_context_from_retrieval(results) -> str:
    """
    Format retrieved chunks for display or for including in the prompt.
    """
    blocks = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    ids = results.get("ids", [[]])[0]
    for idx, doc_text in enumerate(docs):
        meta = metas[idx] if idx and len(metas) > idx else (metas[idx] if metas else {})
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
    Build a concise extractive (no-LLM) answer using top retrieved chunks.
    Returns a string suitable for display.
    """
    docs = retrieved_results.get("documents", [[]])[0]
    metas = retrieved_results.get("metadatas", [[]])[0]
    ids = retrieved_results.get("ids", [[]])[0]

    out_lines = []
    out_lines.append("**Extractive summary (local model synthesis).**")
    out_lines.append("Below are the top retrieved excerpts and their citations.\n")

    summary_parts = []
    for i, text in enumerate(docs):
        if not text:
            continue
        first_sentence = text.strip().split(".")[0]
        pick = first_sentence if len(first_sentence) < 240 else text.strip()[:240]
        meta = metas[i] if metas and len(metas) > i else {}
        cid = ids[i] if ids and len(ids) > i else f"chunk{i}"
        summary_parts.append(f"{pick.strip()} ({_format_citation(meta, cid)})")
        if sum(len(s) for s in summary_parts) > max_chars:
            break
    if summary_parts:
        out_lines.append("**Quick summary:**")
        for s in summary_parts:
            out_lines.append("- " + s)
        out_lines.append("")

    out_lines.append("**Supporting excerpts:**")
    for i, text in enumerate(docs):
        meta = metas[i] if metas and len(metas) > i else {}
        cid = ids[i] if ids and len(ids) > i else f"chunk{i}"
        out_lines.append(f"{i+1}. Citation: {_format_citation(meta, cid)}")
        excerpt = text.strip()
        if len(excerpt) > 1000:
            excerpt = excerpt[:1000].rsplit(" ", 1)[0] + " ... (truncated)"
        out_lines.append(f"> {excerpt}\n")

    return "\n".join(out_lines)

def synthesize_answer(query: str, retrieved_results) -> str:
    """
    Synthesize answer using local HuggingFace model instead of OpenAI.
    """
    context = build_context_from_retrieval(retrieved_results)
    prompt = f"""
You are a legal research assistant. Use ONLY the provided CONTEXT excerpts to answer the user's question.
Cite each factual claim with the source tag: [filename | page | section | id].
If two or more sources conflict, explicitly state each conflicting claim and cite the sources.

CONTEXT:
{context}

QUESTION:
{query}

Provide:
1) A concise answer (2-6 sentences).
2) A numbered list of supporting citations.
3) If conflicts exist, a 'Conflicts' section.
"""

    try:
        result = generator(prompt, max_length=512, do_sample=True, temperature=0.7)
        return result[0]["generated_text"]
    except Exception as e:
        return f"Local model generation failed ({e}).\n\n" + _extractive_answer_from_chunks(retrieved_results)
