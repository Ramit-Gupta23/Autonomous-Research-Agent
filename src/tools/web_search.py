import os
from typing import List
from tavily import TavilyClient


def search_web(query: str, max_results: int = 4) -> List[dict]:
    """
    Tavily se recent web news aur articles dhundta hai.
    """
    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    
    response = client.search(
        query=query,
        max_results=max_results,
        include_answer=False
    )
    
    results = []
    for r in response.get("results", []):
        results.append({
            "title": r.get("title", "No title"),
            "content": r.get("content", "")[:600],
            "url": r.get("url", ""),
            "score": r.get("score", 0)
        })
    
    return results