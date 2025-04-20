import os
import dspy
from typing import List, Dict
from langchain_core.documents import Document
from answer_generation import get_answer, analyze_legal_query  # Assuming the code is in legal_analysis.py
from config import GROQ_API_KEY
try:
    from config import GROQ_API_KEY
except ImportError:
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY")


# Mock retriever function to simulate document retrieval
def mock_retriever(query: str) -> Dict[str, List[Document]]:
    """
    Creates a mock retrieval result with sample legal documents
    """
    # Create sample documents
    doc1 = Document(
        page_content="""
        Section 3.1 Vehicle Registration Requirements
        All vehicles operated on public roads must be registered with the Department of Motor Vehicles.
        Registration must be renewed annually and requires:
        1. Proof of insurance
        2. Vehicle safety inspection certificate
        3. Payment of registration fees based on vehicle type and weight
        """,
        metadata={"source": "vehicle_code.pdf", "page": 12}
    )
    
    doc2 = Document(
        page_content="""
        Section 5.4 Commercial Vehicle Regulations
        Commercial vehicles exceeding 10,000 pounds require special registration.
        Operators must possess a valid Commercial Driver's License (CDL).
        Weight restrictions apply on certain roadways and bridges.
        """,
        metadata={"source": "commercial_regulations.pdf", "page": 27}
    )
    
    # Return dictionary mapping query to list of documents
    return {query: [doc1, doc2]}

# Test cases
def run_tests():
    print("Starting Legal Document Analyzer tests...\n")
    
    # Test case 1: Basic query
    query1 = "What are the requirements for vehicle registration?"
    print(f"Test 1 Query: {query1}")
    retrieval_results = mock_retriever(query1)
    
    result = get_answer(query1, retrieval_results)
    print(f"Answer: {result}\n")
    
    # Test case 2: More specific query
    query2 = "Do commercial vehicles need special registration?"
    print(f"Test 2 Query: {query2}")
    retrieval_results = mock_retriever(query2)
    
    result = get_answer(query2, retrieval_results)
    print(f"Answer: {result}\n")
    
    # Test case 3: Out of context query to test low confidence handling
    query3 = "What are motorcycle helmet laws?"
    print(f"Test 3 Query: {query3}")
    retrieval_results = mock_retriever(query3)
    
    result = get_answer(query3, retrieval_results)
    print(f"Answer: {result}\n")
    
    # Test case 4: Full analysis with detailed output
    query4 = "What documents are needed for vehicle registration?"
    print(f"Test 4 Query (Full Analysis): {query4}")
    retrieval_results = mock_retriever(query4)
    
    detailed_result = analyze_legal_query(
        question=query4,
        retriever=retrieval_results,
        confidence_threshold=75
    )
    
    print("Detailed Analysis:")
    print(f"Answer: {detailed_result.get('answer', 'No answer')}")
    print(f"Confidence: {detailed_result.get('confidence_score', 0)}%")
    print(f"Reasoning: {detailed_result.get('reasoning', 'No reasoning provided')}")
    print(f"References: {detailed_result.get('references', [])}\n")
    
    print("Tests completed!")

if __name__ == "__main__":
    # Make sure GROQ_API_KEY is set in environment or config
    if not os.environ.get("GROQ_API_KEY") and not hasattr(config, "GROQ_API_KEY"):
        print("Warning: GROQ_API_KEY not found! Set it before running tests.")
    
    # Run the tests
    run_tests()