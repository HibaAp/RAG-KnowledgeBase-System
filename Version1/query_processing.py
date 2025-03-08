from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticToolsParser
from config import GROQ_API_KEY
from retrieval import retrieve
from web_search import web_search 
from content_extraction import fetch_content_from_link


# Define a class for structured output
class SubQuery(BaseModel):
    """Extracts multiple sub-queries from a user query for retrieval."""
    sub_queries: list[str] = Field(..., description="List of highly specific database queries.")

# Define the function
def retrieve_results(query, retrieve_function=retrieve,groq_api_key=GROQ_API_KEY):
    """
    Converts a user query into sub-queries and retrieves relevant documents.

    Args:
        query (str): The user query.
        retrieve_function (function): Function to retrieve documents from a database.

    Returns:
        dict: Dictionary of sub-queries and their retrieved documents.
    """

    system_prompt = """You are an expert at query decomposition.
    Your task is to break a user question into multiple highly specific sub-queries
    that must be answered to fully respond to the original question.

    Ensure sub-queries are specific and relevant to the context.
    """

    # Create the chat prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{question}"),
        ]
    )

    # Initialize the LLM (Llama 3 via Groq)
    llm = ChatGroq(groq_api_key=groq_api_key, model="llama3-70b-8192", temperature=0.05)

    # Bind the LLM with tools
    llm_with_tools = llm.bind_tools([SubQuery])

    # Create a parser to extract structured responses
    parser = PydanticToolsParser(tools=[SubQuery])

    # Create the processing pipeline
    query_analyzer = prompt | llm_with_tools | parser

    # Invoke the query analyzer
    response = query_analyzer.invoke({"question": query})

    # Extract sub-queries
    sub_queries = []
    for item in response:
        sub_queries.extend(item.sub_queries)

    # Retrieve results for each sub-query
    results = {}
    for sub_query in sub_queries:
        web_results = '\n'.join([fetch_content_from_link(link) for link in web_search(sub_query)])[:3000]
        retrieved_docs = retrieve_function(web_results)  # Call the retrieval function
        results[sub_query] = retrieved_docs

    return results
