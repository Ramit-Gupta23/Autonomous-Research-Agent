import os
from dotenv import load_dotenv
from src.agent.graph import research_agent
from src.agent.state import ResearchState

load_dotenv()


def run(topic: str):
    print(f"\n🔬 Researching: {topic}")
    print("─" * 50)
    
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
    
    # Step by step output dikhao
    for step in research_agent.stream(initial_state):
        for node_name, state_update in step.items():
            print(f"→ {state_update.get('current_step', node_name)}")
    
    # Final result
    final = research_agent.invoke(initial_state)
    
    print("\n" + "═" * 50)
    print("📋 FINAL REPORT:")
    print("═" * 50)
    print(final["final_report"])
    
    # Save to file
    with open("report.md", "w", encoding="utf-8") as f:
        f.write(final["final_report"])
    print("\n✅ Report saved to report.md")


if __name__ == "__main__":
    topic = input("\nEnter research topic: ")
    run(topic)