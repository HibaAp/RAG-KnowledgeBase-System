from googlesearch import search

def web_search(query: str, max_results: int = 3) :
    """Return the first web search result."""
    try:
        results = list(search(query, num_results=max_results))
        return results
    except Exception as e:
        print(f"Error: {e}")
        return None