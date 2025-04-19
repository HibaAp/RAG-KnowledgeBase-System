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
         You are a system specialized in legal document analysis. Your task is to compare two answers generated from different legal documents related to vehicle regulations. Follow these steps:

    1. **Extract Relevant Content:** Answer the question using most relevent contents from the context.
    2. **Extract Numerical Values:** Compare extracted numerical values where applicable.
    3. use the entire context to answer the question.
    4. **Final Answer:** Provide a concise answer to the question based on the extracted information.
    5. In answers avoid terms like "based on the given context" or "according to the context".
    6. Give the result as an answer to the question.frmae the result in accordance with the question.  
        Do not miss out any detail regrading the query in the answer.  
        Answer accurately using only the provided sub queries and their curresponding answers given as contexts  .
        If no relevant information is found, state that no relevant information is available.

        CONTEXT: {context}\nQUESTION: {question}\nFINAL ANSWER:
    """, input_variables=["context", "question"])
    chain = prompt | ChatGroq(groq_api_key=groq_api_key, model='llama3-70b-8192', temperature=0.05)
    response = chain.invoke({"context": doc_context, "question": query})
    return response.content 
