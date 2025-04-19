from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticToolsParser
from langchain_huggingface import HuggingFaceEmbeddings
from config import GROQ_API_KEY
from retrieval import retrieve
from web_search import web_search 
from content_extraction import fetch_content_from_link


# Define a class for structured output
class SubQuery(BaseModel):
    """Extracts multiple sub-queries from a user query for retrieval."""
    sub_queries: list[str] = Field(..., description="List of highly specific database queries.")


# Define a class for hypothetical document output
class HypotheticalDocument(BaseModel):
    """Generates a hypothetical document for a given sub-query."""
    document: str = Field(..., description="A hypothetical document answering the sub-query.")


# Initialize the embedding model globally (if needed later)
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en",
    encode_kwargs={'normalize_embeddings': False}
)


def generate_hypothetical_document(sub_query, web_results, groq_api_key=GROQ_API_KEY):
    system_prompt = """You are an expert at generating concise, high-quality hypothetical documents.
    Given a specific query and relevant web content, generate a short, realistic document (50-100 words) 
    that plausibly answers the query, incorporating insights from the web content where relevant.
    Focus on a factual tone and relevance to the query."""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "Query: {sub_query}\nWeb Content: {web_results}"),
        ]
    )

    llm = ChatGroq(groq_api_key=groq_api_key, model="llama3-70b-8192", temperature=0.05)
    llm_with_tools = llm.bind_tools([HypotheticalDocument])
    parser = PydanticToolsParser(tools=[HypotheticalDocument])
    doc_generator = prompt | llm_with_tools | parser

    response = doc_generator.invoke({"sub_query": sub_query, "web_results": web_results})
    return response[0].document


def retrieve_results(query, retrieve_function=retrieve, groq_api_key=GROQ_API_KEY, use_web=True):
    """
    Converts a user query into sub-queries, optionally uses web search + HyDE,
    and retrieves relevant documents.

    Args:
        query (str): The user query.
        retrieve_function (function): Function to retrieve documents from a database.
        groq_api_key (str): Groq API key.
        use_web (bool): Whether to use web search for HyDE document generation.

    Returns:
        dict: Dictionary of sub-queries and their retrieved documents.
    """
    print("\n" + "=" * 80)
    print(f"PROCESSING QUERY: '{query}'")
    print("=" * 80)

    # Step 1: Sub-query decomposition
    system_prompt = """You are an expert at query decomposition.
    Your task is to break a user question into multiple highly specific sub-queries
    that must be answered to fully respond to the original question.
    Ensure sub-queries are specific and relevant to the context.
    Make the first sub-query the main one."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}"),
    ])

    llm = ChatGroq(groq_api_key=groq_api_key, model="llama3-70b-8192", temperature=0.05)
    llm_with_tools = llm.bind_tools([SubQuery])
    parser = PydanticToolsParser(tools=[SubQuery])
    query_analyzer = prompt | llm_with_tools | parser

    print("\n1. GENERATING SUB-QUERIES...")
    response = query_analyzer.invoke({"question": query})
    sub_queries = [sq for item in response for sq in item.sub_queries]

    print("-" * 80)
    print("SUB-QUERIES GENERATED:")
    for i, sq in enumerate(sub_queries, 1):
        print(f"  {i}. {sq}")
    print("-" * 80)

    # Step 2: Process each sub-query
    results = {}

    for i, sub_query in enumerate(sub_queries, 1):
        print(f"\n2.{i} PROCESSING SUB-QUERY: '{sub_query}'")
        print("-" * 80)

        # a. Web search (conditional)
        if use_web:
            print(f"  a. Fetching web results...")
            web_links = web_search(sub_query)
            print(f"     Found {len(web_links)} web links")

            web_results = []
            for j, link in enumerate(web_links, 1):
                content = fetch_content_from_link(link)
                web_results.append(content)
                print(f"     {j}. Extracted content from: {link[:50]}...")
            web_content = '\n'.join(web_results)[:1500]
        else:
            print("  a. Web search disabled — skipping.")
            web_content = ""

        # b. Generate HyDE document
        print(f"  b. Generating hypothetical document...")
        hypo_doc = generate_hypothetical_document(sub_query, web_content, groq_api_key)
        print(f"     Hypothetical document generated ({len(hypo_doc)} chars)")
        print(f"     Preview: {hypo_doc[:100]}...")

        # c. Retrieve documents
        print(f"  c. Retrieving documents using hypothetical document embedding...")
        retrieved_docs = retrieve_function(hypo_doc)
        print(f"     Retrieved {len(retrieved_docs)} documents")

        results[sub_query] = retrieved_docs

        print("\n  RETRIEVED DOCUMENTS SUMMARY:")
        for j, doc in enumerate(retrieved_docs, 1):
            doc_preview = str(doc)[:100].replace('\n', ' ')
            print(f"     Doc {j}: {doc_preview}...")

        print(f"\n  COMPLETED PROCESSING SUB-QUERY {i}/{len(sub_queries)}")
        print("-" * 80)

    print("\nRETRIEVAL PROCESS COMPLETE")
    print(f"Successfully processed {len(sub_queries)} sub-queries")
    print("=" * 80 + "\n")

    return results
