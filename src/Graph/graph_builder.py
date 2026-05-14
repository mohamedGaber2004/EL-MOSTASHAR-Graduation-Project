from __future__ import annotations

import logging
from functools import lru_cache
from typing import Literal, Union

from langgraph.graph import END, START, StateGraph

from src.Graph.states_and_schemas.state import AgentState
from src.Graph.shared_resources import get_kg, get_vector_store

from src.agents import (
    DataIngestionAgent,
    ProceduralAuditorAgent,
    LegalResearcherAgent,
    EvidenceAnalystAgent,
    ConfessionValidityAgent,
    WitnessCredibilityAgent,
    ProsecutionAnalystAgent,
    DefenseAnalystAgent,
    SentencingAgent,
    JudgeAgent,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Error wrapper  (بدون agent_outputs)
# ─────────────────────────────────────────────────────────────────────────────

def _safe_run(agent_key: str, agent):
    def _wrapped(state: AgentState) -> AgentState:
        try:
            return agent.run(state)
        except Exception as e:
            logger.error("❌ Agent '%s' فشل: %s", agent_key, e, exc_info=True)
            return state.model_copy(update={
                "errors":           state.errors + [f"[{agent_key}] {e}"],
                "completed_agents": state.completed_agents + [agent_key],
                "current_agent":    agent_key,
            })
    return _wrapped


# ─────────────────────────────────────────────────────────────────────────────
# Conditional routing
# ─────────────────────────────────────────────────────────────────────────────

def _route_after_procedural_audit(state: AgentState) -> Literal["legal_researcher", "judge"]:
    """
    إذا اكتشف المدقق الإجرائي مشكلة قاتلة (سقوط بالتقادم / بطلان مطلق /
    عدم اختصاص) تجاوز بقية العملاء وأرسل القضية مباشرة إلى القاضي.
    """
    fatal_keywords = {
        "سقوط", "تقادم", "بطلان مطلق", "عدم اختصاص",
        "انقضاء الدعوى", "وفاة المتهم",
    }
    for issue in state.procedural_issues:
        desc = getattr(issue, "issue_description", "") or ""
        if any(kw in desc for kw in fatal_keywords):
            logger.warning(
                "⚡ مشكلة إجرائية قاتلة — تجاوز مباشر إلى القاضي: %s", desc[:80]
            )
            return "judge"
    return "legal_researcher"


def _route_after_parallel_analysis(state: AgentState) -> Literal["prosecution_analyst"]:
    """
    بعد اكتمال المرحلة الموازية (evidence + confession + witness)
    يُحال الملف دائماً إلى محلل الاتهام.
    الدالة موجودة للتوسعة المستقبلية.
    """
    return "prosecution_analyst"

# ─────────────────────────────────────────────────────────────────────────────
# Graph builder
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def build_legal_graph():
    kg = get_kg()
    vs = get_vector_store()

    agents = {
        "data_ingestion":     DataIngestionAgent(),
        "procedural_auditor": ProceduralAuditorAgent(kg=kg),
        "legal_researcher":   LegalResearcherAgent(kg=kg, vector_store=vs),
        "evidence_analyst":   EvidenceAnalystAgent(),
        "defense_analyst":    DefenseAnalystAgent(),
        "confession_validity":  ConfessionValidityAgent(),
        "witness_credibility":  WitnessCredibilityAgent(),
        "prosecution_analyst":  ProsecutionAnalystAgent(),
        "sentencing":           SentencingAgent(),
        "judge":              JudgeAgent(),
    }

    builder = StateGraph(AgentState)

    # ── The Nodes ───────────────────────────────────────────────
    for name, agent in agents.items():
        builder.add_node(name, _safe_run(name, agent))

    # ── The Edges ────────────────────────────────────────────────────

    builder.add_edge(START, "data_ingestion")

    builder.add_edge("data_ingestion", "procedural_auditor")

    builder.add_conditional_edges(
        "procedural_auditor",
        _route_after_procedural_audit,
        {
            "legal_researcher": "legal_researcher",
            "judge":            "judge",
        },
    )

    builder.add_edge("legal_researcher", "evidence_analyst")
    builder.add_edge("legal_researcher", "confession_validity")
    builder.add_edge("legal_researcher", "witness_credibility")
    builder.add_edge("evidence_analyst",  "prosecution_analyst")
    builder.add_edge("confession_validity", "prosecution_analyst")
    builder.add_edge("witness_credibility", "prosecution_analyst")
    builder.add_edge("prosecution_analyst", "defense_analyst")
    builder.add_edge("defense_analyst", "sentencing")
    builder.add_edge("sentencing", "judge")
    builder.add_edge("judge", END)

    logger.info("✅ Legal graph built — 10 agents | conditional routing active")
    return builder.compile()


def get_graph_visualization():
    g = build_legal_graph()
    return g, g.get_graph().draw_mermaid_png()


# ─────────────────────────────────────────────────────────────────────────────
# run_case
# ─────────────────────────────────────────────────────────────────────────────

def run_case(state: AgentState) -> AgentState:
    logger.info("🚀 بدء معالجة القضية: %s", state.case_id)

    graph, _ = get_graph_visualization()

    try:
        final_state = graph.invoke(state.model_dump())
    except Exception as e:
        logger.error("❌ Pipeline فشل كلياً: %s", e, exc_info=True)
        raise

    verdict = _extract_verdict(final_state)
    errors  = _extract_errors(final_state)

    logger.info("🏁 Pipeline اكتمل — الحكم: %s", verdict)
    if errors:
        logger.warning("⚠️ أخطاء أثناء التنفيذ (%d): %s", len(errors), errors)

    return final_state


def _extract_verdict(final_state: Union[AgentState, dict]) -> str:
    if isinstance(final_state, dict):
        return str(final_state.get("suggested_verdict", "غير محدد"))
    return str(getattr(final_state, "suggested_verdict", "غير محدد"))


def _extract_errors(final_state: Union[AgentState, dict]) -> list:
    if isinstance(final_state, dict):
        return final_state.get("errors", [])
    return getattr(final_state, "errors", [])
