import arxiv
import time
from typing import List


def search_arxiv(query: str, max_results: int = 4) -> List[dict]:
    """
    Arxiv pe academic papers dhundta hai.
    Rate limit handle karta hai gracefully.
    """
    try:
        time.sleep(3)  # Arxiv ko breathe karne do
        
        client = arxiv.Client(
            num_retries=2,
            delay_seconds=5.0
        )
        
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        results = []
        for paper in client.results(search):
            results.append({
                "title": paper.title,
                "summary": paper.summary[:600],
                "url": paper.pdf_url,
                "published": str(paper.published.date()),
                "authors": [str(a) for a in paper.authors[:3]]
            })
        
        return results

    except Exception as e:
        # Arxiv down ho ya rate limit ho — empty return karo, crash mat karo
        print(f"⚠️ Arxiv unavailable: {e}")
        return []