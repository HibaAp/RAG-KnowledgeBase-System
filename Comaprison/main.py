import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from query_processing import retrieve_results
from answer_generation import get_answer
from retrieval import retrieve

def main():
    user_query = input("Enter your query: ")
    pdf_path = input("Enter the path to the PDF file: ").strip().strip('"')  # ✅ Fix file path formatting

    # Pass pdf_path correctly
    retrieved_docs = retrieve_results(user_query, pdf_path, retrieve_function=retrieve)
    
    answer = get_answer(user_query, retrieved_docs)
    print("\nAnswer:", answer)

if __name__ == "__main__":
    main()

#how vertical inclination of beam is verified?
#"C:\Users\Admin\Downloads\R048r12e.pdf"

