# retrieval.py

from langchain_community.vectorstores import FAISS
from embedding import embed_query, _embedding_model
from config import VECTOR_DB_PATH

def retrieve(text: str, vector_db_path: str = VECTOR_DB_PATH):
    """Return the top‑5 most similar docs for `text` from your FAISS store."""
    q_vec = embed_query(text)
    store = FAISS.load_local(
        vector_db_path,
        _embedding_model,
        allow_dangerous_deserialization=True
    )
    return store.similarity_search_by_vector(q_vec, k=5)
