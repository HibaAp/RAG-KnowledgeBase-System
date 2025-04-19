from langchain_community.embeddings import HuggingFaceBgeEmbeddings
import numpy as np
from typing import List


def embed_texts(texts: List[str]) -> np.ndarray:
    """Embed a list of texts using HuggingFace embeddings."""
    embedding_model = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-large-en", encode_kwargs={'normalize_embeddings': False})
    return embedding_model.embed_documents(texts)