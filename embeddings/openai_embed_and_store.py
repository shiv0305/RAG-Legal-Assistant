# embeddings/openai_embed_and_store.py
import os
import json
from tqdm import tqdm
import math

# Attempt to import chromadb; if not available we'll use a local fallback index.
try:
    import chromadb
    CHROMA_AVAILABLE = True
except Exception:
    chromadb = None
    CHROMA_AVAILABLE = False

# SentenceTransformers for embeddings (already in requirements)
from sentence_transformers import SentenceTransformer
import numpy as np

EMBED_MODEL_NAME = os.environ.get("SBERT_MODEL", "all-MiniLM-L6-v2")
_embedder = SentenceTransformer(EMBED_MODEL_NAME)

LOCAL_INDEX_PATH = os.path.join(".", "local_index.json")
# ensure local path exists
if not os.path.exists(LOCAL_INDEX_PATH):
    # create an empty index file
    with open(LOCAL_INDEX_PATH, "w", encoding="utf8") as f:
        json.dump({"ids": [], "documents": [], "metadatas": [], "embeddings": []}, f)

def get_embedding(text: str):
    """Return embedding vector (list) using SentenceTransformers."""
    if text is None or str(text).strip() == "":
        dim = _embedder.get_sentence_embedding_dimension()
        return [0.0] * dim
    vec = _embedder.encode(text, normalize_embeddings=True)
    # ensure it's a Python list (json-serializable)
    return np.array(vec).astype(float).tolist()

# ----------------------------
# Chroma init (if available)
# ----------------------------
chroma_client = None
collection = None

def _create_or_get_collection(client, name="legal_docs"):
    if hasattr(client, "get_or_create_collection"):
        return client.get_or_create_collection(name)
    if hasattr(client, "get_collection"):
        try:
            return client.get_collection(name)
        except Exception:
            if hasattr(client, "create_collection"):
                return client.create_collection(name)
            raise
    if hasattr(client, "create_collection"):
        return client.create_collection(name)
    raise RuntimeError("Chroma client does not support collection creation/getting methods.")

if CHROMA_AVAILABLE:
    try:
        # Try newer PersistentClient API first
        if hasattr(chromadb, "PersistentClient"):
            chroma_client = chromadb.PersistentClient(path="./chroma_db")
            collection = _create_or_get_collection(chroma_client, "legal_docs")
        else:
            chroma_client = chromadb.Client()
            collection = _create_or_get_collection(chroma_client, "legal_docs")
    except Exception as e_new:
        try:
            from chromadb.config import Settings
            chroma_client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory="./chroma_db"))
            collection = _create_or_get_collection(chroma_client, "legal_docs")
        except Exception as e_legacy:
            CHROMA_AVAILABLE = False
            chroma_client = None
            collection = None

# ----------------------------
# Local index helpers (fallback)
# ----------------------------
def _load_local_index():
    with open(LOCAL_INDEX_PATH, "r", encoding="utf8") as f:
        data = json.load(f)
    # ensure numeric arrays
    return data

def _save_local_index(data):
    with open(LOCAL_INDEX_PATH, "w", encoding="utf8") as f:
        json.dump(data, f)

class _CollectionFallback:
    """A tiny object so app can call collection.count() even when chromadb not present."""
    def count(self):
        data = _load_local_index()
        return len(data.get("ids", []))

# expose a collection-like object in fallback
if not CHROMA_AVAILABLE:
    collection = _CollectionFallback()

# ----------------------------
# Add / query functions
# ----------------------------
def add_chunks_to_chroma(chunks, persist=True):
    """
    chunks: list of dicts with keys: chunk_id, text, filename, page, section, start, end
    """
    if CHROMA_AVAILABLE and collection is not None:
        ids = []
        docs = []
        metadatas = []
        embeddings = []
        for c in tqdm(chunks, desc="Embedding chunks"):
            ids.append(c["chunk_id"])
            docs.append(c["text"])
            metadatas.append({
                "filename": c["filename"],
                "page": c.get("page"),
                "section": c.get("section", "UNKNOWN"),
                "start": c.get("start"),
                "end": c.get("end"),
            })
            embeddings.append(get_embedding(c["text"]))

        # Add to Chroma
        collection.add(documents=docs, metadatas=metadatas, ids=ids, embeddings=embeddings)
        if persist and hasattr(chroma_client, "persist"):
            try:
                chroma_client.persist()
            except Exception:
                pass
        return True

    # FALLBACK: store in local JSON file and append embeddings
    data = _load_local_index()
    for c in tqdm(chunks, desc="Embedding chunks (local fallback)"):
        data["ids"].append(c["chunk_id"])
        data["documents"].append(c["text"])
        data["metadatas"].append({
            "filename": c["filename"],
            "page": c.get("page"),
            "section": c.get("section", "UNKNOWN"),
            "start": c.get("start"),
            "end": c.get("end"),
        })
        data["embeddings"].append(get_embedding(c["text"]))
    _save_local_index(data)
    return True

def _cosine_sim(a: np.ndarray, b: np.ndarray):
    # a: (n_dim,), b: (n_dim,)
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0 or b_norm == 0:
        return 0.0
    return float(np.dot(a, b) / (a_norm * b_norm))

def query_chroma_by_embedding(query: str, k: int = 5):
    """
    Returns a dict compatible with earlier code:
    {
      'ids': [[...]],
      'documents': [[...]],
      'metadatas': [[...]],
      'distances': [[...]]  # optional (1 - cosine_similarity)
    }
    """
    if CHROMA_AVAILABLE and collection is not None:
        q_emb = get_embedding(query)
        res = collection.query(query_embeddings=[q_emb], n_results=k)
        return res

    # Local fallback brute-force search
    data = _load_local_index()
    all_embs = data.get("embeddings", [])
    if len(all_embs) == 0:
        # return empty-style chroma-like response
        return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

    q_vec = np.array(get_embedding(query))
    embs = np.array(all_embs)  # shape (N, dim)
    # compute cosine similarity vectorized
    # handle shapes robustly
    try:
        dots = embs @ q_vec
        norms = np.linalg.norm(embs, axis=1) * np.linalg.norm(q_vec)
        # avoid divide-by-zero
        norms[norms == 0] = 1e-12
        sims = dots / norms
    except Exception:
        # fallback loop
        sims = np.array([_cosine_sim(np.array(e), q_vec) for e in all_embs])

    # top-k indices (descending similarity)
    topk_idx = np.argsort(-sims)[:k]
    docs = [data["documents"][int(i)] for i in topk_idx]
    metas = [data["metadatas"][int(i)] for i in topk_idx]
    ids = [data["ids"][int(i)] for i in topk_idx]
    distances = [float(1.0 - float(sims[int(i)])) for i in topk_idx]

    return {
        "ids": [ids],
        "documents": [docs],
        "metadatas": [metas],
        "distances": [distances]
    }
