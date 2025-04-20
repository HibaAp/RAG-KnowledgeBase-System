import streamlit as st
from query_processing import retrieve_results
from answer_generation import get_answer

# Streamlit page setup
st.set_page_config(page_title="Regulatory Knowledge Assistant", page_icon="⚖️", layout="centered")

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
use_web = st.toggle("Enable Web Search", value=False, label_visibility="collapsed")

if st.button("EXECUTE"):
    if query.strip():
        with st.spinner("Processing..."):
            try:
                # Step 1: Retrieve context docs
                results = retrieve_results(query, use_web=use_web)

                # Step 2: Generate final answer using DSPy-based `get_answer`
                final_answer = get_answer(query, results)

                # Display output
                st.markdown(final_answer if final_answer else "No answer generated.")
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a question.")
