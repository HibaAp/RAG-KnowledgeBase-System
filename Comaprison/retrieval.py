from langchain_community.vectorstores import FAISS
from embedding import embed_texts
from config import VECTOR_DB_PATH
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from text_extraction import extract_text,get_header_footer

def retrieve(web_cont, pdf_path):
    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en",
        encode_kwargs={'normalize_embeddings': False}
    )

    def get_vectorstore1(text):
        texts = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_text(text)
        docs = [Document(page_content=t) for t in texts if t.strip()]
        vectorstore = FAISS.from_documents(docs, embedding_model)
        return vectorstore.as_retriever(search_kwargs={"k": 6})

    text = extract_text(pdf_path)  # Extract text from the PDF
    retriever1 = get_vectorstore1(text)  # This is now the retriever

    encoded_query = embedding_model.embed_query(web_cont)
    
    return retriever1.vectorstore.similarity_search_by_vector(encoded_query, k=5)
