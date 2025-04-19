# web_search.py

from googlesearch import search

def web_search(query: str, max_results: int = 3) -> list:
    """Return up to `max_results` URLs for `query` (never None)."""
    try:
        return list(search(query, num_results=max_results))
    except Exception as e:
        print(f"Web search error: {e}")
        return []
