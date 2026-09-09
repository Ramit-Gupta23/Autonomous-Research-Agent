import streamlit as st
from dotenv import load_dotenv
from src.agent.graph import research_agent
from src.agent.state import ResearchState

load_dotenv()

st.set_page_config(
    page_title="Autonomous Research Agent",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 Autonomous Research Agent")
st.markdown("AI agent which research on basis of  papers, news, analysis and report.")
st.divider()

topic = st.text_input("Research Topic", placeholder="e.g., AI in healthcare, Quantum computing")

if st.button("🚀 Start Research", type="primary", key="main_search_btn"):
    if not topic:
        st.error("⚠️ Topic daalo pehle!")
    else:
        initial_state = ResearchState(
            topic=topic,
            search_queries=[],
            web_results=[],
            arxiv_results=[],
            key_facts=[],
            contradictions=[],
            knowledge_gaps=[],
            final_report="",
            current_step="Starting...",
            messages=[]
        )

        status = st.status("Researching...", expanded=True)

        with status:
            for step in research_agent.stream(initial_state):
                for node_name, state_update in step.items():
                    msg = state_update.get("current_step", node_name)
                    st.write(msg)

        status.update(label="Research Complete! ✅", state="complete")

        final = research_agent.invoke(initial_state)

        tab1, tab2, tab3, tab4 = st.tabs([
            "📋 Full Report",
            "💡 Key Facts",
            "⚠️ Contradictions & Gaps",
            "📚 Sources"
        ])

        with tab1:
            st.markdown(final.get("final_report", "No report generated"))
            st.download_button(
                label="📥 Download Report",
                data=final.get("final_report", ""),
                file_name=f"research_{topic[:20]}.md",
                mime="text/markdown",
                key="download_btn"
            )

        with tab2:
            facts = final.get("key_facts", [])
            if facts:
                for i, fact in enumerate(facts, 1):
                    st.write(f"**{i}.** {fact}")
            else:
                st.info("No facts extracted")

        with tab3:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("⚠️ Contradictions")
                contradictions = final.get("contradictions", [])
                if contradictions:
                    for c in contradictions:
                        st.warning(c)
                else:
                    st.info("No contradictions found")

            with col2:
                st.subheader("🔍 Knowledge Gaps")
                gaps = final.get("knowledge_gaps", [])
                if gaps:
                    for g in gaps:
                        st.info(g)
                else:
                    st.info("No gaps identified")

        with tab4:
            st.subheader("🌐 Web Sources")
            for r in final.get("web_results", [])[:6]:
                st.markdown(f"- [{r['title']}]({r['url']})")

            st.subheader("📚 Academic Papers")
            papers = final.get("arxiv_results", [])
            if papers:
                for r in papers[:5]:
                    st.markdown(f"- [{r['title']}]({r['url']})")
            else:
                st.info("No papers retrieved")