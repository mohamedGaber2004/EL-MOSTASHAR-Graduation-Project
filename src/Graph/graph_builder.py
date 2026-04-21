import logging
from typing import Optional

from langgraph.graph import END, START, StateGraph
from src.Graph.state import AgentState

from src.agents import (
    DataIngestionAgent,
    ProceduralAuditorAgent,
    LegalResearcherAgent,
    EvidenceAnalystAgent,
    DefenseAnalystAgent,
    JudgeAgent
)

# Instantiate agent objects
_data_ingestion_agent = DataIngestionAgent()
_procedural_auditor_agent = ProceduralAuditorAgent()
_legal_researcher_agent = LegalResearcherAgent()
_evidence_analyst_agent = EvidenceAnalystAgent()
_defense_analyst_agent = DefenseAnalystAgent()
_judge_agent = JudgeAgent()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# =============================================================================


def build_legal_graph() -> StateGraph:
    """
    Builds and compiles the Legal Multi-Agent LangGraph (OOP version).
    """
    builder = StateGraph(AgentState)
    builder.add_node("data_ingestion",     _data_ingestion_agent.run)
    builder.add_node("procedural_auditor", _procedural_auditor_agent.run)
    builder.add_node("legal_researcher",   _legal_researcher_agent.run)
    builder.add_node("evidence_analyst",   _evidence_analyst_agent.run)
    builder.add_node("defense_analyst",    _defense_analyst_agent.run)
    builder.add_node("judge",              _judge_agent.run)
    builder.add_edge(START,                "data_ingestion")
    builder.add_edge("data_ingestion",     "procedural_auditor")
    builder.add_edge("procedural_auditor", "legal_researcher")
    builder.add_edge("legal_researcher",   "evidence_analyst")
    builder.add_edge("evidence_analyst",   "defense_analyst")
    builder.add_edge("defense_analyst",    "judge")
    builder.add_edge("judge",              END)
    return builder.compile()


def get_graph_visualization(graph: Optional[StateGraph] = None) -> bytes:
    """Returns Mermaid diagram PNG bytes of the graph.

    If no graph is provided, a fresh graph is built. This avoids import-time
    side effects in environments without IPython display.
    """
    g = graph or build_legal_graph()
    return g.get_graph().draw_mermaid_png()


def run_case(state: AgentState) -> AgentState:
    logger.info("🚀 Starting legal pipeline for case: %s", state.case_id)

    # Build graph at runtime to avoid import-time IO/display side effects.
    graph = build_legal_graph()

    final_state = graph.invoke({
        "case_id": state.case_id,
        "source_documents": state.source_documents,
    })

    logger.info("🏁 Pipeline complete — Verdict: %s", final_state.get("suggested_verdict"))
    return final_state