import logging

from IPython.display import display , Image
from langgraph.graph import END, START, StateGraph
from src.Graph.state import AgentState

from src.Nodes  import (
    data_ingestion_node,
    procedural_auditor_node,
    legal_researcher_node,
    evidence_analyst_node,
    defense_analyst_node,
    judge_node
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# =============================================================================


def build_legal_graph() -> StateGraph:
    """
    Builds and compiles the Legal Multi-Agent LangGraph.

    Flow:
    START
      └─► orchestrator
            └─► data_ingestion
                  └─► procedural_auditor
                        └─► legal_researcher
                              └─► evidence_analyst
                                    └─► defense_analyst
                                          └─► judge
                                                └─► END
    """
    builder = StateGraph(AgentState)

    # ── Register Nodes ────────────────────────────────────────────────────────
    builder.add_node("data_ingestion",     data_ingestion_node)
    builder.add_node("procedural_auditor", procedural_auditor_node)
    builder.add_node("legal_researcher",   legal_researcher_node)
    builder.add_node("evidence_analyst",   evidence_analyst_node)
    builder.add_node("defense_analyst",    defense_analyst_node)
    builder.add_node("judge",              judge_node)

    # ── Register Edges ────────────────────────────────────────────────────────
    builder.add_edge(START,                "data_ingestion")
    builder.add_edge("data_ingestion",     "procedural_auditor")
    builder.add_edge("procedural_auditor", "legal_researcher")
    builder.add_edge("legal_researcher",   "evidence_analyst")
    builder.add_edge("evidence_analyst",   "defense_analyst")
    builder.add_edge("defense_analyst",    "judge")
    builder.add_edge("judge",              END)

    return builder.compile()


legal_graph = build_legal_graph()

def get_graph_visualization() -> bytes:
    """Returns Mermaid diagram PNG bytes of the graph."""
    return legal_graph.get_graph().draw_mermaid_png()

graph_image = display(Image(get_graph_visualization()))

def run_case(state: AgentState) -> AgentState:
    logger.info("🚀 Starting legal pipeline for case: %s", state.case_id)

    final_state = legal_graph.invoke({
        "case_id": state.case_id,
        "source_documents": state.source_documents,
    })

    logger.info("🏁 Pipeline complete — Verdict: %s", final_state.get("suggested_verdict"))
    return final_state