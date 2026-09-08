import arxiv
from typing import List


def search_arxiv(query: str, max_results: int = 4) -> List[dict]:
    """
    Arxiv pe academic papers dhundta hai.
    No API key needed - completely free.
    """
    client = arxiv.Client()
    
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )
    
    results = []
    for paper in client.results(search):
        results.append({
            "title": paper.title,
            "summary": paper.summary[:600],  # First 600 chars enough
            "url": paper.pdf_url,
            "published": str(paper.published.date()),
            "authors": [str(a) for a in paper.authors[:3]]
        })
    
    return results