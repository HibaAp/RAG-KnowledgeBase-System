import os
import streamlit as st

# Disable OneDNN optimization for TensorFlow
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Import your custom modules
from query_processing import retrieve_results
from answer_generation import get_answer
from retrieval import retrieve

# Set Streamlit page configuration
st.set_page_config(page_title="RAG Knowledge Base", layout="centered")

# App title
st.title("📚 RAG Knowledge Base System")
st.markdown("Ask domain-specific questions and get intelligent answers using a Retrieval-Augmented Generation pipeline.")

# User input box
user_query = st.text_area("🔍 Enter your query:", placeholder="E.g. How is the vertical inclination of beam verified?")

# When the user clicks the button
if st.button("Get Answer"):
    if user_query.strip():
        with st.spinner("Retrieving information and generating answer..."):
            try:
                retrieved_docs = retrieve_results(user_query, retrieve_function=retrieve)
                answer = get_answer(user_query, retrieved_docs)
                st.success("✅ Answer Generated")
                st.markdown("### 📥 Answer:")
                st.write(answer)

                # Optional: Show retrieved docs
                with st.expander("📄 Retrieved Documents"):
                    for idx, doc in enumerate(retrieved_docs, 1):
                        st.markdown(f"**Doc {idx}:** {doc}")

            except Exception as e:
                st.error(f"❌ An error occurred: {e}")
    else:
        st.warning("⚠️ Please enter a query to proceed.")
