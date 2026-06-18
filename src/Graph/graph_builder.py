from __future__ import annotations

import logging
from functools import lru_cache
from typing import Literal, Union
import functools

from langgraph.graph import END, START, StateGraph

from src.Graph.state import AgentState
from src.Graph.shared_resources import get_kg, get_vector_store
from src.Config.config import get_settings
from langchain_huggingface import HuggingFaceEmbeddings


from src.agents.data_ingestion_agent.data_ingestion_agent import DataIngestionAgent
from src.agents.procedural_auditor_agent.procedural_auditor_agent import ProceduralAuditorAgent
from src.agents.procedural_auditor_agent.PA_enums import PipelineFatalityLevel
from src.agents.legal_research_agent.legal_researcher_agent import LegalResearcherAgent
from src.agents.evidence_analyst_agent.evidence_analyst_agent import EvidenceAnalystAgent
from src.agents.defense_analyst_agent.defense_analyst_agent import DefenseAnalystAgent
from src.agents.confessoin_validity_agent.confession_validity_agent import ConfessionValidityAgent
from src.agents.witness_credibility_agent.witness_credibility_agent import WitnessCredibilityAgent
from src.agents.prosecution_analyst_agent.prosecution_analyst_agent import ProsecutionAnalystAgent
from src.agents.sentencing_agent.sentencing_agent import SentencingAgent
from src.agents.judge_agent.judge_agent import JudgeAgent

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Error wrapper 
# ─────────────────────────────────────────────────────────────────────────────
def _safe_run(agent_key: str, agent):
    @functools.wraps(agent.run)
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

    _wrapped.__name__ = agent_key
    _wrapped.__qualname__ = agent_key
    return _wrapped

# ─────────────────────────────────────────────────────────────────────────────
# Conditional routing
# ─────────────────────────────────────────────────────────────────────────────

def _route_after_procedural_audit(
    state: AgentState,
) -> Literal["legal_researcher", "judge"]:

    audit = getattr(state, "procedural_audit", None)

    if audit and audit.pipeline_fatality == PipelineFatalityLevel.FATAL:
        logger.warning(
            "⚡ procedural_auditor: fatal — تجاوز مباشر إلى القاضي | السبب: %s",
            getattr(audit, "fatality_reason", "غير محدد"),
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
    kg  = get_kg()
    vs  = get_vector_store()
    emb = HuggingFaceEmbeddings(
        model_name=get_settings().ARABIC_NATIVE_EMBEDDING_MODEL
    )

    agents = {
        "data_ingestion":      DataIngestionAgent(),
        "procedural_auditor":  ProceduralAuditorAgent(kg=kg, embeddings=emb, vector_store=vs),
        "legal_researcher":    LegalResearcherAgent(kg=kg, embeddings=emb, vector_store=vs),
        "evidence_analyst":    EvidenceAnalystAgent(),
        "confession_validity": ConfessionValidityAgent(),
        "witness_credibility": WitnessCredibilityAgent(),
        "prosecution_analyst": ProsecutionAnalystAgent(),
        "defense_analyst":     DefenseAnalystAgent(),
        "sentencing":          SentencingAgent(),
        "judge":               JudgeAgent(),
    }

    builder = StateGraph(AgentState)

    for name, agent in agents.items():
        builder.add_node(name, _safe_run(name, agent))

    # ── Edges ──────────────────────────────────────────────────
    builder.add_edge(START, "data_ingestion")

    builder.add_edge("data_ingestion", "procedural_auditor")

    builder.add_conditional_edges(
        "procedural_auditor",
        _route_after_procedural_audit,
        {
            "legal_researcher": "legal_researcher",
            "judge":            "judge",
        }
    )
    builder.add_edge("legal_researcher", "evidence_analyst")
    builder.add_edge("legal_researcher", "confession_validity")
    builder.add_edge("legal_researcher", "witness_credibility")

    builder.add_edge("evidence_analyst",    "prosecution_analyst")
    builder.add_edge("confession_validity", "prosecution_analyst")
    builder.add_edge("witness_credibility", "prosecution_analyst")

    builder.add_edge("prosecution_analyst", "defense_analyst")
    builder.add_edge("defense_analyst",     "sentencing")
    builder.add_edge("sentencing",          "judge")
    builder.add_edge("judge",               END)

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
