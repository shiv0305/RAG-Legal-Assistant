import os
from tqdm import tqdm

# Try importing Chroma, but handle case where it's missing locally
try:
    import chromadb
    CHROMA_AVAILABLE = True
except ImportError:
    chromadb = None
    CHROMA_AVAILABLE = False
    print("[WARNING] chromadb not installed locally. Retrieval will not work without it.")

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
        if hasattr(chromadb, "PersistentClient"):
            chroma_client = chromadb.PersistentClient(path="./chroma_db")
            collection = _create_or_get_collection(chroma_client, "legal_docs")
        else:
            chroma_client = chromadb.Client()
            collection = _create_or_get_collection(chroma_client, "legal_docs")
    except Exception as e_new:
        try:
            from chromadb.config import Settings
            chroma_client = chromadb.Client(Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory="./chroma_db"
            ))
            collection = _create_or_get_collection(chroma_client, "legal_docs")
        except Exception as e_legacy:
            raise RuntimeError(f"Failed to initialize Chroma client. "
                               f"New-client error: {e_new}; Legacy-client error: {e_legacy}")

# ----------------------------
# Add / query functions
# ----------------------------
def add_chunks_to_chroma(chunks, persist=True):
    """
    chunks: list of dicts with keys: chunk_id, text, filename, page, section, start, end
    """
    if not CHROMA_AVAILABLE:
        raise RuntimeError("chromadb is not available. Cannot store chunks.")

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

    collection.add(documents=docs, metadatas=metadatas, ids=ids, embeddings=embeds)

    if persist and hasattr(chroma_client, "persist"):
        try:
            chroma_client.persist()
        except Exception:
            pass
    return True

def query_chroma_by_embedding(query: str, k: int = 5):
    if not CHROMA_AVAILABLE:
        raise RuntimeError("chromadb is not available. Cannot query database.")

    q_emb = get_embedding(query)
    try:
        return collection.query(query_embeddings=[q_emb], n_results=k)
    except TypeError:
        return collection.query(query_embeddings=[q_emb], top_k=k)
