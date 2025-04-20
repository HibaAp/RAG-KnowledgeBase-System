import streamlit as st
from query_processing import retrieve_results
from answer_generation import get_answer

st.set_page_config(page_title="Legal QA Assistant", layout="wide")

st.title("📜 Legal Document QA ")

# Input section
query = st.text_area("🔍 Enter your legal or compliance question:", height=150)

use_web = st.checkbox("Enable web search", value=False)
run_button = st.button("🧠 Process & Generate Answer")

if run_button and query:
    with st.spinner("Retrieving relevant documents..."):
        try:
            retrieved_docs = retrieve_results(query, use_web=use_web)
        except Exception as e:
            st.error(f"❌ Retrieval failed: {str(e)}")
            st.stop()

    st.success("✅ Documents retrieved!")

    with st.spinner("Generating answer using DSPy..."):
        try:
            final_answer = get_answer(query, retrieved_docs)
        except Exception as e:
            st.error(f"❌ Answer generation failed: {str(e)}")
            st.stop()

    # Output section
    st.subheader("📑 Generated Answer")
    st.markdown(final_answer)

    with st.expander("📚 Retrieved Documents (Debug Info)"):
        for sub_q_index, (sub_q, docs) in enumerate(retrieved_docs.items()):
            st.markdown(f"**Sub-query:** {sub_q}")
            for i, doc in enumerate(docs[:3]):
                doc_key = f"doc_{sub_q_index}_{i}"
                content = doc.page_content[:500] if hasattr(doc, "page_content") else str(doc)
                st.text_area(f"Doc {i+1}", content, height=120, key=doc_key)

else:
    st.info("Enter a question and click the button to start.")
