from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_groq import ChatGroq
from config import GROQ_API_KEY
from langchain_core.runnables import RunnableSequence


# Define the function
def get_answer(query: str, retriever, groq_api_key=GROQ_API_KEY) -> str:
    """Retrieve the most relevant documents and generate an answer."""
    doc_context=""
    for sub_query, retrieved_docs in retriever.items():
      doc_context+="\n---\n".join([doc.page_content for doc in retrieved_docs if doc.page_content.strip()])
    prompt = PromptTemplate(template="""
        You are an intelligent chatbot answering legal document-related queries.
        Answer accurately using only the provided sub queries and their curresponding answers given as contexts  .
        If no relevant information is found, state that no relevant information is available.

        CONTEXT: {context}\nQUESTION: {question}\nFINAL ANSWER:
    """, input_variables=["context", "question"])
    chain = prompt | ChatGroq(groq_api_key=groq_api_key, model='llama3-70b-8192', temperature=0.05)
    response = chain.invoke({"context": doc_context, "question": query})
    return response.content 
