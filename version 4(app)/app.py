import streamlit as st
from query_processing import retrieve_results
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from config import GROQ_API_KEY

# Streamlit page setup
st.set_page_config(page_title="Legal RAG Assistant", page_icon="⚖️", layout="centered")

# Hacker green theme
st.markdown("""
    <style>
        body, .main { background-color: #0a1812; font-family: 'Courier New', monospace; }
        .stTextInput>div>input, .stTextArea textarea {
            background-color: #0f2318; color: #00ff41; border: 1px solid #00ff41;
            border-radius: 0px; font-family: 'Courier New', monospace;
        }
        .stButton>button {
            background-color: #0f2318; color: #00ff41; border: 1px solid #00ff41;
            border-radius: 0px; font-family: 'Courier New', monospace;
        }
        .block-container { padding: 1rem; }
        h1, h2, h3, p, .markdown-text-container {
            color: #00ff41;
        }
    </style>
""", unsafe_allow_html=True)

st.title("⚖️ Legal RAG Assistant")

query = st.text_area("Enter your legal question:", height=100)
use_web = st.toggle("Enable Web Search", value=True)

if st.button("EXECUTE"):
    if query.strip():
        with st.spinner("Processing..."):
            try:
                results = retrieve_results(query, use_web=use_web)

                # Flatten retrieved documents
                all_docs = []
                for docs in results.values():
                    all_docs.extend([str(d) for d in docs])

                if all_docs:
                    joined_text = "\n\n".join(all_docs)[:4000]  # Simple truncation

                    summary_prompt = ChatPromptTemplate.from_messages([
                        ("system", "You are a legal assistant. Summarize the following retrieved documents to answer the user's legal question as clearly and concisely as possible."),
                        ("human", "User Question: {query}\n\nRetrieved Content:\n{context}")
                    ])

                    llm = ChatGroq(groq_api_key=GROQ_API_KEY, model="llama3-70b-8192", temperature=0.3)
                    summarizer = summary_prompt | llm | StrOutputParser()
                    final_answer = summarizer.invoke({"query": query, "context": joined_text})

                    st.markdown(final_answer)
                else:
                    st.info("No relevant information found.")
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a question.")
