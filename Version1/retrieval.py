from langchain_community.vectorstores import FAISS
from embedding import embed_texts
from config import VECTOR_DB_PATH
from langchain_huggingface import HuggingFaceEmbeddings

def retrieve(web_cont,vector_db_path=VECTOR_DB_PATH):
  embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en",
    encode_kwargs={'normalize_embeddings': False}
  )
  encoded_query=embedding_model.embed_query(web_cont)
  vectorstore = FAISS.load_local(vector_db_path, embedding_model, allow_dangerous_deserialization=True)
  return vectorstore.similarity_search_by_vector(encoded_query, k=5)