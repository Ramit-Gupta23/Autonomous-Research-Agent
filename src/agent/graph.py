from langgraph.graph import StateGraph, END
from src.agent.state import ResearchState
from src.agent.nodes import (
    planner_node,
    web_search_node,
    arxiv_search_node,
    extractor_node,
    analyzer_node,
    reporter_node
)


def build_graph():
    graph = StateGraph(ResearchState)
    
    # Nodes add karo
    graph.add_node("planner", planner_node)
    graph.add_node("web_search", web_search_node)
    graph.add_node("arxiv_search", arxiv_search_node)
    graph.add_node("extractor", extractor_node)
    graph.add_node("analyzer", analyzer_node)
    graph.add_node("reporter", reporter_node)
    
    # Flow define karo — ek straight pipeline
    graph.set_entry_point("planner")
    graph.add_edge("planner", "web_search")
    graph.add_edge("web_search", "arxiv_search")
    graph.add_edge("arxiv_search", "extractor")
    graph.add_edge("extractor", "analyzer")
    graph.add_edge("analyzer", "reporter")
    graph.add_edge("reporter", END)
    
    return graph.compile()


# Import karke direct use karo
research_agent = build_graph()