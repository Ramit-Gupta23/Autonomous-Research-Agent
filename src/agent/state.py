from typing import TypedDict, List, Annotated
from langgraph.graph.message import add_messages


class ResearchState(TypedDict):
    topic: str                               # User ka research topic
    search_queries: List[str]                # Agent khud banata hai queries
    web_results: List[dict]                  # Tavily se aaye results
    arxiv_results: List[dict]               # Arxiv papers
    key_facts: List[str]                     # Extracted important facts
    contradictions: List[str]               # Conflicting info across sources
    knowledge_gaps: List[str]               # Kya nahi mila research mein
    final_report: str                        # Final markdown report
    current_step: str                        # UI ko dikhane ke liye
    messages: Annotated[list, add_messages]  # Chat history