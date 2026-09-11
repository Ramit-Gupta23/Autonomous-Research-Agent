import os
import json
import time
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from src.agent.state import ResearchState
from src.tools.arxiv_tool import search_arxiv
from src.tools.web_search import search_web
from dotenv import load_dotenv
import src.cache.chroma_cache as cache
load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)


def parse_json_safely(text: str, fallback: list) -> list:
    try:
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
    try:
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

    except Exception as e:
        print(f"⚠️ Planner failed: {e}")
        fallback = [
            f"{state['topic']} overview 2024",
            f"{state['topic']} research paper",
            f"{state['topic']} latest news",
            f"{state['topic']} developments",
            f"{state['topic']} future"
        ]
        return {
            "search_queries": fallback,
            "current_step": "⚠️ Planning failed — using fallback queries"
        }


# ─────────────────────────────────────────
# NODE 2: WEB SEARCH
# ─────────────────────────────────────────
def web_search_node(state: ResearchState) -> dict:
    try:
        if not state.get("search_queries"):
            return {
                "web_results": [],
                "current_step": "⚠️ Web search skipped — no queries"
            }

        # ── Cache check ──
        cached = cache.get_web_results(state["topic"])
        if cached:
            return {
                "web_results": cached,
                "current_step": f"📦 Web results from cache — {len(cached)} sources"
            }

        # ── Cache miss — actual search karo ──
        all_results = []
        for query in state["search_queries"][2:]:
            try:
                results = search_web(query, max_results=3)
                all_results.extend(results)
                time.sleep(1)
            except Exception as e:
                print(f"⚠️ Web search failed for '{query}': {e}")
                continue

        # ── Results cache mein save karo ──
        if all_results:
            cache.save_web_results(state["topic"], all_results)

        return {
            "web_results": all_results,
            "current_step": f"🌐 Web search done — {len(all_results)} sources found"
        }

    except Exception as e:
        print(f"⚠️ Web search node failed: {e}")
        return {
            "web_results": [],
            "current_step": "⚠️ Web search failed"
        }


# ─────────────────────────────────────────
# NODE 3: ARXIV SEARCH
# ─────────────────────────────────────────
def arxiv_search_node(state: ResearchState) -> dict:
    try:
        if not state.get("search_queries"):
            return {
                "arxiv_results": [],
                "current_step": "⚠️ Arxiv skipped — no queries"
            }

        # ── Cache check ──
        cached = cache.get_arxiv_results(state["topic"])
        if cached:
            return {
                "arxiv_results": cached,
                "current_step": f"📦 Arxiv results from cache — {len(cached)} papers"
            }

        # ── Cache miss — actual search karo ──
        all_results = []
        for query in state["search_queries"][:2]:
            try:
                results = search_arxiv(query, max_results=3)
                all_results.extend(results)
                time.sleep(3)
            except Exception as e:
                print(f"⚠️ Arxiv failed for '{query}': {e}")
                continue

        # ── Results cache mein save karo ──
        if all_results:
            cache.save_arxiv_results(state["topic"], all_results)

        return {
            "arxiv_results": all_results,
            "current_step": f"📚 Arxiv done — {len(all_results)} papers found"
        }

    except Exception as e:
        print(f"⚠️ Arxiv node failed: {e}")
        return {
            "arxiv_results": [],
            "current_step": "⚠️ Arxiv failed"
        }
# ─────────────────────────────────────────
# NODE 4: EXTRACTOR
# ─────────────────────────────────────────
def extractor_node(state: ResearchState) -> dict:
    try:
        web_results  = state.get("web_results", [])
        arxiv_results = state.get("arxiv_results", [])

        # Dono empty hain toh kya extract karein?
        if not web_results and not arxiv_results:
            return {
                "key_facts": ["No sources were available to extract facts from."],
                "current_step": "⚠️ Extraction skipped — no sources found"
            }

        web_text = "\n\n".join([
            f"[WEB] {r['title']}\n{r['content']}"
            for r in web_results[:5]
        ])

        arxiv_text = "\n\n".join([
            f"[PAPER] {r['title']} ({r['published']})\n{r['summary']}"
            for r in arxiv_results[:4]
        ])

        combined = f"{web_text}\n\n{arxiv_text}".strip()

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

    except Exception as e:
        print(f"⚠️ Extractor failed: {e}")
        return {
            "key_facts": [f"Extraction failed due to error: {str(e)}"],
            "current_step": "⚠️ Extraction failed — using placeholder"
        }


# ─────────────────────────────────────────
# NODE 5: ANALYZER
# ─────────────────────────────────────────
def analyzer_node(state: ResearchState) -> dict:
    try:
        key_facts = state.get("key_facts", [])

        if not key_facts or key_facts == ["No sources were available to extract facts from."]:
            return {
                "contradictions": [],
                "knowledge_gaps": ["No facts available to analyze"],
                "current_step": "⚠️ Analysis skipped — no facts to analyze"
            }

        facts_text = "\n".join([f"- {f}" for f in key_facts])

        response = llm.invoke([
            SystemMessage(content="""You are a critical research analyst.
Analyze these facts and find:
1. CONTRADICTIONS: Where sources disagree or conflict
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
            gaps = ["Could not parse analysis results"]

        return {
            "contradictions": contradictions,
            "knowledge_gaps": gaps,
            "current_step": f"🔬 Analysis done — {len(contradictions)} contradictions, {len(gaps)} gaps found"
        }

    except Exception as e:
        print(f"⚠️ Analyzer failed: {e}")
        return {
            "contradictions": [],
            "knowledge_gaps": ["Analysis failed — could not identify gaps"],
            "current_step": "⚠️ Analysis failed — continuing to report"
        }


# ─────────────────────────────────────────
# NODE 6: REPORTER
# ─────────────────────────────────────────
def reporter_node(state: ResearchState) -> dict:
    try:
        key_facts     = state.get("key_facts", [])
        contradictions = state.get("contradictions", [])
        gaps          = state.get("knowledge_gaps", [])
        web_results   = state.get("web_results", [])
        arxiv_results = state.get("arxiv_results", [])

        # Kuch bhi nahi mila toh bhi ek basic report banao
        if not key_facts:
            return {
                "final_report": f"# {state['topic']} — Research Report\n\nNo data could be retrieved for this topic. Please try again.",
                "current_step": "⚠️ Report generation failed — no data available"
            }

        facts_text         = "\n".join([f"- {f}" for f in key_facts])
        contradictions_text = "\n".join([f"- {c}" for c in contradictions]) or "None identified"
        gaps_text          = "\n".join([f"- {g}" for g in gaps]) or "None identified"
        total_sources      = len(web_results) + len(arxiv_results)

        response = llm.invoke([
            SystemMessage(content="""You are a professional research report writer.
Write a comprehensive, well-structured markdown research report.

REQUIRED SECTIONS (use these exact headings):
# [Topic] - Research Report
## Executive Summary
## Key Findings
## Contradictions in Current Research
## Knowledge Gaps & Future Directions
## Conclusion

Make it specific, professional, and insightful."""),
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

    except Exception as e:
        print(f"⚠️ Reporter failed: {e}")
        # Last resort — basic report manually banao
        basic_report = f"""# {state['topic']} - Research Report

## Executive Summary
Research was conducted on {state['topic']} but report generation encountered an error.

## Key Facts Found
{chr(10).join([f"- {f}" for f in state.get('key_facts', ['No facts available'])])}

## Error
Reporter node failed: {str(e)}
"""
        return {
            "final_report": basic_report,
            "current_step": "⚠️ Report used fallback template due to error"
        }