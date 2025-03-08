import requests
import pdfplumber
from bs4 import BeautifulSoup
from io import BytesIO
from docx import Document
import re

def clean_text(text):
      text = text.replace("\n", " ")  # Remove newlines
      text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
      return text



def fetch_content_from_link(url: str) -> str:
    """Fetches and extracts relevant text from a given URL."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise an error for bad responses

        # Check if it's a PDF file
        if url.lower().endswith(".pdf"):
            with pdfplumber.open(BytesIO(response.content)) as pdf:
                text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
            return clean_text(text.strip())

        # Check if it's a DOCX file
        elif url.lower().endswith(".docx") or "docx?" in url:
            with BytesIO(response.content) as docx_file:
                doc = Document(docx_file)
                text = "\n".join(para.text for para in doc.paragraphs)
            return clean_text(text.strip())

        # Assume it's an HTML page
        else:
            soup = BeautifulSoup(response.text, "html.parser")
            text = soup.get_text(separator="\n", strip=True)
            return clean_text(text.strip())

    except Exception as e:
        return f"Error fetching content: {e}"