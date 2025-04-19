# embedding.py

from langchain_community.embeddings import HuggingFaceBgeEmbeddings
import numpy as np
from typing import List

# single shared model instance
_embedding_model = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-large-en",
    encode_kwargs={"normalize_embeddings": False}
)

def embed_texts(texts: List[str]) -> np.ndarray:
    """Embed a list of texts."""
    return _embedding_model.embed_documents(texts)

def embed_query(text: str) -> np.ndarray:
    """Embed a single query string."""
    return _embedding_model.embed_query(text)
