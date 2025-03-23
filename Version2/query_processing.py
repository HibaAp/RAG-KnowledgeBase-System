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

# Initialize the embedding model globally
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en",
    encode_kwargs={'normalize_embeddings': False}
)
def generate_hypothetical_document(sub_query, web_results, groq_api_key=GROQ_API_KEY):
    system_prompt = """You are an expert at generating concise, high-quality hypothetical documents.
    Given a specific query and relevant web content, generate a short, realistic document (100-200 words) 
    that plausibly answers the query, incorporating insights from the web content where relevant.
    Focus on a factual tone and relevance to the query."""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "Query: {sub_query}\nWeb Content: {web_results}"),
        ]
    )

    # Initialize the LLM
    llm = ChatGroq(groq_api_key=groq_api_key, model="llama3-70b-8192", temperature=0.05)

    # Bind the LLM with tools
    llm_with_tools = llm.bind_tools([HypotheticalDocument])
    parser = PydanticToolsParser(tools=[HypotheticalDocument])

    # Create the pipeline
    doc_generator = prompt | llm_with_tools | parser

    # Generate the hypothetical document
    response = doc_generator.invoke({"sub_query": sub_query, "web_results": web_results})
    return response[0].document

# Updated retrieve_results function with your HyDE approach
def retrieve_results(query, retrieve_function=retrieve, groq_api_key=GROQ_API_KEY):
    """
    Converts a user query into sub-queries, generates hypothetical documents using web results and sub-queries,
    and retrieves relevant documents.

    Args:
        query (str): The user query.
        retrieve_function (function): Function to retrieve documents from a database.

    Returns:
        dict: Dictionary of sub-queries and their retrieved documents.
    """

    # System prompt for query decomposition
    system_prompt = """You are an expert at query decomposition.
    Your task is to break a user question into multiple highly specific sub-queries
    that must be answered to fully respond to the original question.
    Ensure sub-queries are specific and relevant to the context."""

    # Create the chat prompt for sub-query decomposition
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{question}"),
        ]
    )

    # Initialize the LLM for sub-query decomposition
    llm = ChatGroq(groq_api_key=groq_api_key, model="llama3-70b-8192", temperature=0.05)
    llm_with_tools = llm.bind_tools([SubQuery])
    parser = PydanticToolsParser(tools=[SubQuery])

    # Create the processing pipeline for sub-queries
    query_analyzer = prompt | llm_with_tools | parser

    # Invoke the query analyzer
    response = query_analyzer.invoke({"question": query})

    # Extract sub-queries
    sub_queries = []
    for item in response:
        sub_queries.extend(item.sub_queries)

    # Retrieve results for each sub-query using HyDE
    results = {}
    for sub_query in sub_queries:
        # Step 1: Fetch web results for the sub-query
        web_results = '\n'.join([fetch_content_from_link(link) for link in web_search(sub_query)])[:3000]

        # Step 2: Generate a hypothetical document using the sub-query and web results
        hypo_doc = generate_hypothetical_document(sub_query, web_results, groq_api_key)

        # Step 3: Retrieve documents using the hypothetical document embedding
        retrieved_docs = retrieve_function(hypo_doc)
        results[sub_query] = retrieved_docs

    return results
