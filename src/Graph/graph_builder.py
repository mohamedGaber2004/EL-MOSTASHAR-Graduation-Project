from __future__ import annotations

import logging
from functools import lru_cache
from typing import Literal, Union
import functools

from langgraph.graph import END, START, StateGraph

from src.Graph.states_and_schemas.state import AgentState
from src.Graph.shared_resources import get_kg, get_vector_store
from src.Config.config import get_settings
from langchain_huggingface import HuggingFaceEmbeddings


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
# Error wrapper 
# ─────────────────────────────────────────────────────────────────────────────
def _safe_run(agent_key: str, agent):
    @functools.wraps(agent.run)          # ← preserves __name__, __doc__, etc.
    def _wrapped(state):
        try:
            result = agent.run(state)
            if isinstance(result, dict):
                _PARALLEL_FORBIDDEN_KEYS = {
                    "case_id", "case_number", "court", "court_level",
                    "jurisdiction", "filing_date", "referral_date",
                    "prosecutor_name", "referral_order_text",
                    "current_agent", "last_updated",
                    "extraction_notes", "source_documents",
                }
                return {k: v for k, v in result.items() if k not in _PARALLEL_FORBIDDEN_KEYS}
            return result
        except Exception as e:
            logger.error("❌ Agent '%s' فشل: %s", agent_key, e, exc_info=True)
            return {
                "completed_agents": [],
                "errors":           [f"{agent_key}: {str(e)[:200]}"],
            }

    _wrapped.__name__ = agent_key       # ← unique key LangGraph uses as node ID
    _wrapped.__qualname__ = agent_key
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

    audit = getattr(state, "procedural_audit", None)
    if not audit:
        return "legal_researcher"

    # fast path: check critical_nullities list first (agent already flagged these)
    for nullity in (audit.critical_nullities or []):
        if any(kw in (nullity or "") for kw in fatal_keywords):
            logger.warning(
                "⚡ بطلان مطلق حرج — تجاوز مباشر إلى القاضي: %s", nullity[:80]
            )
            return "judge"

    # slow path: scan every violation's description and nullity_type
    for issue in (audit.violations or []):
        desc        = getattr(issue, "issue_description", "") or ""
        nullity_type = getattr(issue, "nullity_type", "") or ""

        if any(kw in desc or kw in nullity_type for kw in fatal_keywords):
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
    emb= HuggingFaceEmbeddings(model_name=get_settings().ARABIC_NATIVE_EMBEDDING_MODEL)

    agents = {
        "data_ingestion":     DataIngestionAgent(),
        "procedural_auditor": ProceduralAuditorAgent(kg=kg, embeddings=emb, vector_store=vs),
        "legal_researcher":   LegalResearcherAgent(kg=kg, embeddings=emb, vector_store=vs),
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

def run_case(state: AgentState) -> dict:
    logger.info("🚀 بدء معالجة القضية: %s", state.case_id)

    graph, _ = get_graph_visualization()

    try:
        raw = graph.invoke(state.model_dump())
    except Exception as e:
        logger.error("❌ Pipeline فشل كلياً: %s", e, exc_info=True)
        raise

    # graph.invoke returns a merged dict — reconstruct typed state for safety
    if isinstance(raw, AgentState):
        final_state = raw
    elif isinstance(raw, dict):
        try:
            final_state = AgentState(**raw)
        except Exception as e:
            logger.error(
                "AgentState reconstruction failed: %s — returning raw dict", e
            )
            final_state = raw
    else:
        logger.error("Unexpected graph output type: %s", type(raw))
        final_state = raw

    verdict = _extract_verdict(final_state)
    errors  = _extract_errors(final_state)

    logger.info("🏁 Pipeline اكتمل — الحكم: %s", verdict)
    if errors:
        logger.warning("⚠️ أخطاء أثناء التنفيذ (%d): %s", len(errors), errors)

    return final_state


def _extract_verdict(final_state) -> str:
    if isinstance(final_state, dict):
        return str(final_state.get("suggested_verdict", "غير محدد"))
    return str(getattr(final_state, "suggested_verdict", "غير محدد"))


def _extract_errors(final_state) -> list:
    if isinstance(final_state, dict):
        return final_state.get("errors", [])
    return getattr(final_state, "errors", [])


def _extract_verdict(final_state: Union[AgentState, dict]) -> str:
    if isinstance(final_state, dict):
        return str(final_state.get("suggested_verdict", "غير محدد"))
    return str(getattr(final_state, "suggested_verdict", "غير محدد"))


def _extract_errors(final_state: Union[AgentState, dict]) -> list:
    if isinstance(final_state, dict):
        return final_state.get("errors", [])
    return getattr(final_state, "errors", [])
