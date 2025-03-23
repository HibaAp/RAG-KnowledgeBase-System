import pdfplumber
import random
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain.embeddings import HuggingFaceBgeEmbeddings




embedding_model = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-large-en",
                               encode_kwargs={'normalize_embeddings': False})
    
def embed_texts(texts):
        # return FastEmbedEmbeddings.embed_documents(embedding_model,texts = texts)
        return embedding_model.embed_documents(texts)

def get_header_footer(pdf_path, threshold=0.71):
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            if total_pages >= 15:
                random_page_nos = random.sample(range(5, total_pages), 10)
            else:
                random_page_nos = list(range(total_pages))
            
            avg_similarity = 1
            header_lines = -1
            
            while avg_similarity > threshold and header_lines < 4:
                header_lines += 1
                five_lines = []
            
                for page_no in random_page_nos:
                    lines = pdf.pages[page_no].extract_text().split('\n')
                    if len(lines) > header_lines:
                        five_lines.append(lines[header_lines])
                similarities = cosine_similarity(embed_texts(five_lines))
                avg_similarity = np.mean(similarities[np.triu_indices(len(similarities), k=1)])
                
            avg_similarity = 1
            footer_lines = -1
            
            while avg_similarity > threshold and footer_lines < 4:
                footer_lines += 1
                five_lines = []
                
                for page_no in random_page_nos:
                    lines = pdf.pages[page_no].extract_text().split('\n')
                    if len(lines) > footer_lines:
                        five_lines.append(lines[-(footer_lines+1)])
                similarities = cosine_similarity(embed_texts(five_lines))
                avg_similarity = np.mean(similarities[np.triu_indices(len(similarities), k=1)]) 
            return header_lines, footer_lines
        
def extract_text(pdf_path):
    header_lines, footer_lines = get_header_footer(pdf_path)
    with pdfplumber.open(pdf_path) as pdf:
        text = ''
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                lines = page_text.split('\n')
                if lines:
                    page_text = '\n'.join(lines[header_lines:-(footer_lines+1)])
                    text += page_text + '\n'
        return text