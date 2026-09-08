import os
import json
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from src.agent.state import ResearchState
from src.tools.arxiv_tool import search_arxiv
from src.tools.web_search import search_web
from dotenv import load_dotenv

load_dotenv()

# Ek LLM instance - sab nodes yahi use karenge
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)


def parse_json_safely(text: str, fallback: list) -> list:
    """
    LLM kabhi kabhi JSON ke around markdown wrap karta hai.
    Ye function safely parse karta hai.
    """
    try:
        # Markdown code blocks hatao agar hain
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
        return json.loads(cleaned.strip())
    except Exception:
        return fallback


# ─────────────────────────────────────────
# NODE 1: PLANNER
# ─────────────────────────────────────────
def planner_node(state: ResearchState) -> dict:
    """
    Topic leke 5 targeted search queries banata hai.
    2 academic papers ke liye, 3 news/developments ke liye.
    """
    response = llm.invoke([
        SystemMessage(content="""You are an expert research planner.
Given a topic, generate exactly 5 search queries.
- First 2 queries: for academic/scientific papers
- Last 3 queries: for recent news and developments

Return ONLY a valid JSON array of 5 strings. No explanation. No markdown.
Example format: ["query1", "query2", "query3", "query4", "query5"]"""),
        
        HumanMessage(content=f"Research topic: {state['topic']}")
    ])
    
    fallback = [
        f"{state['topic']} research papers",
        f"{state['topic']} academic study 2024",
        f"{state['topic']} latest news 2024",
        f"{state['topic']} recent developments",
        f"{state['topic']} challenges and future"
    ]
    
    queries = parse_json_safely(response.content, fallback)
    
    return {
        "search_queries": queries,
        "current_step": f"✅ Planning done — {len(queries)} queries generated"
    }


# ─────────────────────────────────────────
# NODE 2: WEB SEARCH
# ─────────────────────────────────────────
def web_search_node(state: ResearchState) -> dict:
    """
    Last 3 queries use karke Tavily se web search karta hai.
    """
    all_results = []
    
    for query in state["search_queries"][2:]:  # Last 3 = news queries
        results = search_web(query, max_results=3)
        all_results.extend(results)
    
    return {
        "web_results": all_results,
        "current_step": f"🌐 Web search done — {len(all_results)} sources found"
    }


# ─────────────────────────────────────────
# NODE 3: ARXIV SEARCH
# ─────────────────────────────────────────
def arxiv_search_node(state: ResearchState) -> dict:
    """
    First 2 queries use karke Arxiv se academic papers dhundta hai.
    """
    all_results = []
    
    for query in state["search_queries"][:2]:  # First 2 = academic queries
        results = search_arxiv(query, max_results=3)
        all_results.extend(results)
    
    return {
        "arxiv_results": all_results,
        "current_step": f"📚 Arxiv search done — {len(all_results)} papers found"
    }


# ─────────────────────────────────────────
# NODE 4: EXTRACTOR
# ─────────────────────────────────────────
def extractor_node(state: ResearchState) -> dict:
    """
    Sab sources padhke 10-15 key facts extract karta hai.
    """
    # Web content prepare karo
    web_text = "\n\n".join([
        f"[WEB] {r['title']}\n{r['content']}"
        for r in state["web_results"][:5]
    ])
    
    # Arxiv content prepare karo
    arxiv_text = "\n\n".join([
        f"[PAPER] {r['title']} ({r['published']})\n{r['summary']}"
        for r in state["arxiv_results"][:4]
    ])
    
    combined = f"{web_text}\n\n{arxiv_text}"
    
    response = llm.invoke([
        SystemMessage(content="""You are a research analyst.
Extract 10-15 key facts from the provided sources.
Each fact must be:
- Specific (not vague)
- Include the type of source [WEB] or [PAPER]
- Be a complete, standalone statement

Return ONLY a valid JSON array of strings. No markdown. No explanation.
Example: ["Fact one with source info", "Fact two...", ...]"""),
        
        HumanMessage(content=f"Topic: {state['topic']}\n\nSources:\n{combined[:3000]}")
    ])
    
    fallback = ["Could not extract structured facts from sources"]
    facts = parse_json_safely(response.content, fallback)
    
    return {
        "key_facts": facts,
        "current_step": f"💡 Extraction done — {len(facts)} key facts found"
    }


# ─────────────────────────────────────────
# NODE 5: ANALYZER
# ─────────────────────────────────────────
def analyzer_node(state: ResearchState) -> dict:
    """
    Key facts mein contradictions aur knowledge gaps dhundta hai.
    Ye cheez ise normal RAG se alag banati hai.
    """
    facts_text = "\n".join([f"- {f}" for f in state["key_facts"]])
    
    response = llm.invoke([
        SystemMessage(content="""You are a critical research analyst.
Analyze these facts and find:
1. CONTRADICTIONS: Where sources disagree or conflict with each other
2. KNOWLEDGE GAPS: Important questions NOT answered by these sources

Return ONLY valid JSON in this exact format. No markdown:
{
    "contradictions": ["contradiction 1", "contradiction 2"],
    "gaps": ["gap 1", "gap 2", "gap 3"]
}"""),
        
        HumanMessage(content=f"Topic: {state['topic']}\n\nFacts:\n{facts_text}")
    ])
    
    try:
        cleaned = response.content.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
        analysis = json.loads(cleaned.strip())
        contradictions = analysis.get("contradictions", [])
        gaps = analysis.get("gaps", [])
    except Exception:
        contradictions = []
        gaps = []
    
    return {
        "contradictions": contradictions,
        "knowledge_gaps": gaps,
        "current_step": f"🔬 Analysis done — {len(contradictions)} contradictions, {len(gaps)} gaps found"
    }


# ─────────────────────────────────────────
# NODE 6: REPORTER
# ─────────────────────────────────────────
def reporter_node(state: ResearchState) -> dict:
    """
    Sab information leke final structured markdown report banata hai.
    """
    facts_text = "\n".join([f"- {f}" for f in state["key_facts"]])
    contradictions_text = "\n".join([f"- {c}" for c in state["contradictions"]]) or "None identified"
    gaps_text = "\n".join([f"- {g}" for g in state["knowledge_gaps"]]) or "None identified"
    
    total_sources = len(state["web_results"]) + len(state["arxiv_results"])
    
    response = llm.invoke([
        SystemMessage(content="""You are a professional research report writer.
Write a comprehensive, well-structured markdown research report.

REQUIRED SECTIONS (use these exact headings):
# [Topic] — Research Report
## Executive Summary
## Key Findings
## Contradictions in Current Research  
## Knowledge Gaps & Future Directions
## Conclusion

Make it specific, professional, and insightful. Use bullet points where appropriate."""),
        
        HumanMessage(content=f"""
Topic: {state['topic']}
Total Sources Analyzed: {total_sources}

Key Facts:
{facts_text}

Contradictions Found:
{contradictions_text}

Knowledge Gaps:
{gaps_text}
""")
    ])
    
    return {
        "final_report": response.content,
        "current_step": "✅ Report generation complete!"
    }