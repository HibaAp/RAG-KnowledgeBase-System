# main.py

from config import GROQ_API_KEY
from query_processing import retrieve_results
from answer_generation import get_answer

def main():
    user_query = input("Enter your query: ")
    
    # Pass the key into your retrieval pipeline
    retrieved = retrieve_results(
        query=user_query,
        groq_api_key=GROQ_API_KEY
    )
    
    # And also into your answer generator
    answer = get_answer(
        query=user_query,
        retriever=retrieved,
        groq_api_key=GROQ_API_KEY
    )
    
    print("\nAnswer:\n", answer)

if __name__ == "__main__":
    main()


#How is the vertical inclination of the beam is verified?
