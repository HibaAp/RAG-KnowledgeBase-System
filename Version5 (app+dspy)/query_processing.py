# query_processing.py

import os
import dspy
from config import GROQ_API_KEY
from retrieval import retrieve
from web_search import web_search
from content_extraction import fetch_content_from_link
from typing import Dict, List


def retrieve_results(
    query: str,
    retrieve_function=retrieve,
    groq_api_key: str = None,
    use_web: bool = False
) -> Dict[str, list]:
    """
    Decompose `query` into 3-5 sub-queries, optionally HyDE‑generate a doc for each,
    then retrieve top‑5 docs via FAISS for each sub-query, printing progress steps.
    """
    # 0. Setup API key and DSPy
    print("\n" + "=" * 80)
    print(f"PROCESSING QUERY: '{query}'")
    print("=" * 80)
    api_key = groq_api_key if groq_api_key is not None else GROQ_API_KEY
    os.environ["GROQ_API_KEY"] = api_key

    print("\n1. CONFIGURING DSPy...")
    lm = dspy.LM("groq/llama3-8b-8192", api_key=api_key, temperature=0.05)
    dspy.configure(lm=lm)

    # 2. Define DSPy signatures with explicit List[str] type
    class SubQueryExtraction(dspy.Signature):
        question: str = dspy.InputField()
        sub_queries: List[str] = dspy.OutputField(desc="List of 3 specific sub-queries. the sub -queries must be questions that will be answered by the documents")

    class HyDEGeneration(dspy.Signature):
        sub_query: str = dspy.InputField()
        web_results: str = dspy.InputField()
        document: str = dspy.OutputField(desc="Hypothetical document text")

    # 3. Create DSPy modules
    class SubQueryGenerator(dspy.Module):
        def __init__(self):
            self.gen = dspy.Predict(
                SubQueryExtraction,
                prompt="""
You are an expert at query decomposition.
Break the user question into between 3 and 5 highly specific sub-queries.
Make the first sub-query the main one.
Respond with a JSON list of 3 to 5 sub-queries.
"""
            )
        def forward(self, question):
            return self.gen(question=question)

    class HypotheticalDocumentGenerator(dspy.Module):
        def __init__(self):
            self.gen = dspy.Predict(
                HyDEGeneration,
                prompt="""
You are an expert at generating concise hypothetical documents (50–100 words).
Given a sub-query and web content, produce a plausible mini‑document.
Respond with just the document text.
"""
            )
        def forward(self, sub_query, web_results):
            return self.gen(sub_query=sub_query, web_results=web_results)

    subgen  = SubQueryGenerator()
    hydegen = HypotheticalDocumentGenerator()

    # 4. Generate sub-queries
    print("\n2. GENERATING SUB-QUERIES...")
    resp = subgen(question=query)
    raw = resp.sub_queries
    # Handle if raw is a string
    if isinstance(raw, str):
        sub_queries = [s.strip("\" ' \n") for s in raw.strip().splitlines() if s.strip()]
    else:
        sub_queries = raw

    # Enforce exactly 3-5 sub-queries
    if len(sub_queries) > 5:
        sub_queries = sub_queries[:5]
    elif len(sub_queries) < 3:
        print(f"Warning: only {len(sub_queries)} sub-queries generated; expected at least 3.")

    print("SUB-QUERIES GENERATED:")
    for idx, sq in enumerate(sub_queries, 1):
        print(f"  {idx}. {sq}")

    results: Dict[str, list] = {}
    # 5. Process each sub-query
    for i, sq in enumerate(sub_queries, 1):
        print(f"\n3.{i} PROCESSING SUB-QUERY: '{sq}'")
        print("-" * 80)

        # a. Web search + extraction
        if use_web:
            print("  3.a Fetching web results...")
            links = web_search(sq) or []
            print(f"     Found {len(links)} links")
            texts = []
            for j, u in enumerate(links, 1):
                content = fetch_content_from_link(u)
                texts.append(content)
                print(f"     {j}. Extracted content from: {u}")
            web_ctx = "\n".join(texts)[:1500]
        else:
            print("  3.a Web search disabled")
            web_ctx = ""

        # b. HyDE document
        print("  3.b Generating hypothetical document...")
        doc = hydegen(sub_query=sq, web_results=web_ctx).document
        print(f"     Hypothetical doc length: {len(doc)} chars")

        # c. FAISS retrieval
        print("  3.c Retrieving documents via FAISS...")
        try:
            docs = retrieve_function(doc)
        except Exception as e:
            print(f"     Embedding failed ({e}), falling back to sub-query text")
            docs = retrieve_function(sq)
        print(f"     Retrieved {len(docs)} documents")

        results[sq] = docs

    print("\nRETRIEVAL PROCESS COMPLETE")
    print(f"Successfully processed {len(sub_queries)} sub-queries")
    print("=" * 80 + "\n")

    return results
