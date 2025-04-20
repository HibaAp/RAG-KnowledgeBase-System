# answer_generation.py

import dspy
import os
from config import GROQ_API_KEY
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq

# Initialize DSPy with consistent model and key
api_key = os.environ.get("GROQ_API_KEY", GROQ_API_KEY)

if not dspy.settings.lm:
    lm = dspy.LM("groq/llama3-8b-8192", api_key=api_key, temperature=0.05)
    dspy.configure(lm=lm)

def get_answer(
    query: str,
    retriever: dict,
    groq_api_key: str = None
) -> str:
    """
    Combine retrieved docs and generate a step-by-step answer with section references.
    """
    key = groq_api_key or GROQ_API_KEY

    # 1. Build combined context
    context = ""
    for docs in retriever.values():
        for doc in docs:
            text = getattr(doc, "page_content", None) or str(doc)
            if text.strip():
                context += text.strip() + "\n---\n"

    # 2. Define detailed step-by-step prompt
    prompt = PromptTemplate(
        template="""
You are an expert in legal document analysis. Answer using this format:

**Section Referenced:** [e.g., Section 6.2.3]

**Step-by-Step Procedure:**
1. [First action with details]
2. [Next action with details]
3. [...]

**Final Answer:**
[Concise conclusion]

Guidelines:
- Include the regulation section or clause where each instruction originates.
- Provide 5–7 detailed steps.
- Avoid phrases like "based on the context" or "according to the text".

CONTEXT:
{context}

QUESTION:
{question}

FINAL ANSWER:
""",
        input_variables=["context", "question"]
    )

    # 3. Invoke Groq LLM via DSPy
    chain = prompt | ChatGroq(
        groq_api_key=key,
        model="llama3-8b-8192",
        temperature=0.05
    )
    response = chain.invoke({"context": context, "question": query})
    return response.content
