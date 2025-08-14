# app/app.py
import sys, os
# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import os
from utils.utils import ingest_and_index_file
from embeddings.openai_embed_and_store import collection, query_chroma_by_embedding
from retrieval.retrieve_and_answer import synthesize_answer

st.set_page_config(page_title="RAG Legal Assistant", layout="wide")

st.title("RAG — Multi-Document Legal Research Assistant (Demo)")

# Sidebar: upload and index docs
st.sidebar.header("Ingest documents")
uploaded_files = st.sidebar.file_uploader("Upload PDF / DOCX / TXT (multiple)", accept_multiple_files=True)

if st.sidebar.button("Index uploaded files"):
    if not uploaded_files:
        st.sidebar.warning("Upload at least one file.")
    else:
        total_chunks = 0
        for uf in uploaded_files:
            # save temporary
            save_path = os.path.join("uploaded", uf.name)
            os.makedirs("uploaded", exist_ok=True)
            with open(save_path, "wb") as f:
                f.write(uf.getbuffer())
            st.sidebar.info(f"Parsing & indexing {uf.name} ...")
            c = ingest_and_index_file(save_path)
            total_chunks += c
        st.sidebar.success(f"Indexed {total_chunks} chunks across uploaded files.")

st.sidebar.markdown("### Existing indexed collections")
try:
    col_stats = collection.count()
    st.sidebar.write("Indexed items:", col_stats)
except Exception:
    st.sidebar.write("No collection or not initialized.")

st.header("Ask a legal question")
query = st.text_area("Type your question here", height=120)
k = st.slider("Number of retrieved passages (k)", 1, 10, 4)

if st.button("Get answer"):
    if not query.strip():
        st.warning("Please type a question.")
    else:
        with st.spinner("Retrieving relevant passages..."):
            res = query_chroma_by_embedding(query, k=k)
        st.subheader("Retrieved excerpts (provenance shown)")
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        ids = res.get("ids", [[]])[0]
        for i, doc in enumerate(docs):
            meta = metas[i]
            st.markdown(f"**Source:** `{meta.get('filename')}` | page {meta.get('page')} | section: {meta.get('section')} | id: `{ids[i]}`")
            st.write(doc)
            st.markdown("---")

        with st.spinner("Synthesizing answer with citations..."):
            answer = synthesize_answer(query, res)
        st.subheader("Answer (synthesized)")
        st.write(answer)

st.caption("Notes: This is a demo. Do not treat outputs as legal advice.")
