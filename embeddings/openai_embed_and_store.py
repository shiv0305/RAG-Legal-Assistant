# embeddings/openai_embed_and_store.py
import os
from tqdm import tqdm
import chromadb

# Use Sentence Transformers for local embeddings (free)
from sentence_transformers import SentenceTransformer

EMBED_MODEL_NAME = os.environ.get("SBERT_MODEL", "all-MiniLM-L6-v2")
_embedder = SentenceTransformer(EMBED_MODEL_NAME)

def get_embedding(text: str):
    """Return embedding vector (list) using SentenceTransformers."""
    if text is None or text.strip() == "":
        dim = _embedder.get_sentence_embedding_dimension()
        return [0.0] * dim
    vec = _embedder.encode(text, normalize_embeddings=True)
    return vec.tolist()

# ----------------------------
# Robust Chroma client init
# ----------------------------
chroma_client = None
collection = None

def _create_or_get_collection(client, name="legal_docs"):
    """Try various APIs to get/create a collection safely."""
    # 1) get_or_create_collection (newer versions)
    if hasattr(client, "get_or_create_collection"):
        return client.get_or_create_collection(name)
    # 2) get_collection (if exists) else create_collection
    if hasattr(client, "get_collection"):
        try:
            return client.get_collection(name)
        except Exception:
            # create if not exists
            if hasattr(client, "create_collection"):
                return client.create_collection(name)
            raise
    # 3) create_collection only (last resort)
    if hasattr(client, "create_collection"):
        return client.create_collection(name)
    # otherwise raise
    raise RuntimeError("Chroma client does not support collection creation/getting methods.")

# Try new-style PersistentClient first (preferred)
try:
    if hasattr(chromadb, "PersistentClient"):
        chroma_client = chromadb.PersistentClient(path="./chroma_db")
        collection = _create_or_get_collection(chroma_client, "legal_docs")
    else:
        # Fallback to generic Client (covers many versions)
        chroma_client = chromadb.Client()
        collection = _create_or_get_collection(chroma_client, "legal_docs")
except Exception as e_new:
    # As a last resort, try legacy Settings-based initialization (if available)
    try:
        from chromadb.config import Settings
        chroma_client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory="./chroma_db"))
        collection = _create_or_get_collection(chroma_client, "legal_docs")
    except Exception as e_legacy:
        # Give a helpful error that includes both messages
        raise RuntimeError(f"Failed to initialize Chroma client. New-client error: {e_new}; Legacy-client error: {e_legacy}")

# ----------------------------
# Add / query functions
# ----------------------------
def add_chunks_to_chroma(chunks, persist=True):
    """
    chunks: list of dicts with keys: chunk_id, text, filename, page, section, start, end
    """
    ids, docs, metadatas, embeds = [], [], [], []
    for c in tqdm(chunks, desc="Embedding chunks"):
        ids.append(c["chunk_id"])
        docs.append(c["text"])
        metadatas.append({
            "filename": c["filename"],
            "page": c["page"],
            "section": c.get("section", "UNKNOWN"),
            "start": c.get("start"),
            "end": c.get("end"),
        })
        embeds.append(get_embedding(c["text"]))

    # Add to Chroma (works for client versions that support embeddings)
    collection.add(documents=docs, metadatas=metadatas, ids=ids, embeddings=embeds)

    # persist if supported
    if persist and hasattr(chroma_client, "persist"):
        try:
            chroma_client.persist()
        except Exception:
            # some clients persist differently or not at all; ignore safely
            pass
    return True

def query_chroma_by_embedding(query: str, k: int = 5):
    q_emb = get_embedding(query)
    # Many versions accept query_embeddings; handle both possible arg names
    try:
        return collection.query(query_embeddings=[q_emb], n_results=k)
    except TypeError:
        # fallback for some versions
        return collection.query(query_embeddings=[q_emb], top_k=k)
