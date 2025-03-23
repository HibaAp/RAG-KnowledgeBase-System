from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from config import GROQ_API_KEY

# Define the function
def get_answer(query: str, retrieved_docs, groq_api_key=GROQ_API_KEY) -> str:
    """Retrieve the most relevant documents and generate an answer."""

    # Construct the context from retrieved documents
    doc_context = "\n\n".join(
        f"Sub-query: {sub_query}\nRelevant Information:\n" +
        "\n---\n".join([doc.page_content for doc in docs if doc.page_content.strip()])
        for sub_query, docs in retrieved_docs.items()
        if docs  # Ensure there are retrieved docs
    )

    # If no relevant context is found
    if not doc_context.strip():
        return "No relevant information is available."

    # Define the prompt
    prompt = PromptTemplate(
        template="""
        You are an intelligent chatbot answering legal document-related queries.
        Answer accurately using only the provided sub-queries and their corresponding answers given as context.
        If no relevant information is found, state that no relevant information is available.

        CONTEXT: {context}

        QUESTION: {question}

        FINAL ANSWER:
        """,
        input_variables=["context", "question"]
    )

    # Initialize the model
    llm = ChatGroq(groq_api_key=groq_api_key, model='llama3-70b-8192', temperature=0.05)

    # Generate the response
    response = llm.invoke({"context": doc_context, "question": query})

    return response.get("text", "No response generated.")  # Ensure safe access
