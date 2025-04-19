# content_extraction.py

import requests
import pdfplumber
from bs4 import BeautifulSoup
from io import BytesIO
from docx import Document
import re

def clean_text(text: str) -> str:
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text

def fetch_content_from_link(url: str) -> str:
    """Fetches and extracts text from PDF, DOCX or HTML pages."""
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()

        if url.lower().endswith(".pdf"):
            with pdfplumber.open(BytesIO(resp.content)) as pdf:
                text = "\n".join(p.extract_text() or "" for p in pdf.pages)
        elif url.lower().endswith(".docx") or "docx?" in url:
            with BytesIO(resp.content) as b:
                doc = Document(b)
                text = "\n".join(p.text for p in doc.paragraphs)
        else:
            soup = BeautifulSoup(resp.text, "html.parser")
            text = soup.get_text(separator="\n", strip=True)

        return clean_text(text)
    except Exception:
        # on any error, return empty so HyDE won’t break
        return ""
