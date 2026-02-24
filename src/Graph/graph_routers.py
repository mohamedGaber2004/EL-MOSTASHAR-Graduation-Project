import logging 
from  src.Graph import AgentState
from typing import Literal
from langgraph.graph import END

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def route_after_orchestrator(
    state: AgentState,
) -> Literal["data_ingestion"]:
    """Orchestrator always routes to DataIngestion first."""
    return "data_ingestion"


def route_after_ingestion(
    state: AgentState,
) -> Literal["procedural_auditor", "legal_researcher"]:
    """
    After ingestion: fan out to ProceduralAuditor first (blocking).
    LegalResearcher runs in parallel conceptually, but LangGraph needs
    explicit edges — both get edges, execution order handled by graph.
    Returns procedural_auditor as primary (legal_researcher added via add_edge).
    """
    return "procedural_auditor"


def route_after_procedural(
    state: AgentState,
) -> Literal["legal_researcher"]:
    return "legal_researcher"


def route_after_legal_research(
    state: AgentState,
) -> Literal["evidence_analyst"]:
    return "evidence_analyst"


def route_after_evidence(
    state: AgentState,
) -> Literal["defense_analyst"]:
    return "defense_analyst"


def route_after_defense(
    state: AgentState,
) -> Literal["judge"]:
    return "judge"


def route_after_judge(
    state: AgentState,
) -> Literal["__end__"]:
    logger.info("✅ Case %s completed — Verdict: %s | Confidence: %.2f",
                state.case_id,
                state.suggested_verdict,
                state.confidence_score or 0.0)
    return END