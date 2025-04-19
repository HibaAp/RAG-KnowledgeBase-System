import os
import re
import pdfplumber
import numpy as np
import faiss
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import random
def detect_headers_footers(pdf_path, threshold=0.71):
    """Detect headers and footers based on repeated content across multiple pages."""
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        sample_pages = random.sample(range(min(5, total_pages), total_pages), min(10, total_pages))
        
        def detect_region(lines, position):
            line_count, avg_similarity = -1, 1
            while avg_similarity > threshold and line_count < 4:
                line_count += 1
                sampled_lines = [lines[page][line_count if position == 'header' else -(line_count+1)] 
                                 for page in sample_pages if len(lines[page]) > line_count]
                if len(sampled_lines) > 1:
                    similarities = cosine_similarity(embed_texts(sampled_lines))
                    avg_similarity = np.mean(similarities[np.triu_indices(len(similarities), k=1)])
                else:
                    break
            return line_count
        
        all_text_lines = {i: pdf.pages[i].extract_text().split('\n') for i in range(total_pages)}
        return detect_region(all_text_lines, 'header'), detect_region(all_text_lines, 'footer')

def embed_texts(texts, model_name="BAAI/bge-large-en"):
    """Embed input texts using HuggingFace embedding model."""
    embedding_model = HuggingFaceBgeEmbeddings(model_name=model_name, encode_kwargs={'normalize_embeddings': True})
    return np.array(embedding_model.embed_documents(texts))

def extract_hierarchical_sections(pdf_path):
    """Extract structured sections, subsections, and sub-subsections from a PDF."""
    header_lines, footer_lines = detect_headers_footers(pdf_path)
    sections = []
    section_stack = []  # Maintains hierarchy
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            page_text = page.extract_text()
            if not page_text:
                continue
            
            lines = page_text.split('\n')[header_lines:-(footer_lines+1)]
            for line in lines:
                line = line.strip()
                match = re.match(r'^(\d+(\.\d+)*\.?\s+)([A-Z][a-zA-Z\s]+)$', line)
                if match:
                    section_number, title = match.group(1).strip(), match.group(3).strip()
                    level = section_number.count('.')
                    
                    # Adjust stack to match hierarchy
                    while len(section_stack) > level:
                        section_stack.pop()
                    
                    new_section = {'title': title, 'number': section_number, 'text': '', 'page_start': page_num, 'page_end': page_num, 'subsections': []}
                    if section_stack:
                        section_stack[-1]['subsections'].append(new_section)
                    else:
                        sections.append(new_section)
                    section_stack.append(new_section)
                else:
                    if section_stack:
                        section_stack[-1]['text'] += line + '\n'
                        section_stack[-1]['page_end'] = page_num
    return sections

def flatten_sections(sections):
    """Flatten hierarchical sections into a list for embedding and retrieval."""
    flat_list = []
    def traverse(section, parent_title=""):
        full_title = f"{parent_title} {section['title']}".strip()
        flat_list.append({
            'title': full_title,
            'number': section['number'],
            'text': section['text'],
            'page_start': section['page_start'],
            'page_end': section['page_end']
        })
        for sub in section['subsections']:
            traverse(sub, full_title)
    for sec in sections:
        traverse(sec)
    return flat_list