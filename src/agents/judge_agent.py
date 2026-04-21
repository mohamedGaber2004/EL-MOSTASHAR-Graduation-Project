import json
from pathlib import Path
 
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings
 
from .agent_base import AgentBase
from src.LLMs import get_llm_qwn, get_llm_oss, get_llm_llama
from src.Graph.graph_helpers import (
    _retrieve_for_charge, _build_fallback_package, _parse_llm_json,
    _apply_extracted_to_state, _agent_context, _now,
)
from src.Graphstore.KG_builder import LegalKnowledgeGraph
from src.Vectorstore.vector_store_builder import load_vector_store
from src.Utils.agent_output_utils import clean_agent_output
from src.Utils import VerdictType
from src.Prompts import (
    LEGAL_RESEARCHER_AGENT_PROMPT,
    DEFENSE_ANALYSIS_AGENT_PROMPT,
    EVIDENCE_ANALYST_AGENT_PROMPT,
    JUDGE_AGENT_PROMPT,
    PROCEDURAL_AUDITOR_AGENT_PROMPT,
)

# ─────────────────────────────────────────────
# القاضي
# ─────────────────────────────────────────────
class JudgeAgent(AgentBase):
    def __init__(self):
        super().__init__("JUDGE_MODEL", "JUDGE_TEMP", JUDGE_AGENT_PROMPT)
 
    def run(self, state):
        self.log(f"👨‍⚖️ القاضي — القضية {state.case_id}")
        prompt = (
            f"أصدر مسودة حكم مسببة:\n{_agent_context(state, 'judge')}\n\n"
            "ناقش كل دليل، رد على كل دفع، أشر لأي شك أو نقص. الشك للمتهم."
        )
        response = self._llm_invoke_with_retries(
            get_llm_llama(self.model_name, self.temperature),
            [SystemMessage(content=self.prompt), HumanMessage(content=prompt)],
        )
        judgment = _parse_llm_json(response.content)
        verdict_text = judgment.get("verdict", "")
        verdict = VerdictType(verdict_text) if verdict_text else VerdictType.ACQUITTAL
 
        return state.model_copy(update={
            "agent_outputs":   {**state.agent_outputs, "judge": judgment},
            "suggested_verdict": verdict,
            "confidence_score":  judgment.get("confidence_score", 0.0),
            "reasoning_trace":   [{"agent": "judge", "reasoning": judgment.get("verdict_reasoning", "")}],
            "completed_agents":  state.completed_agents + ["judge"],
            "current_agent":     "judge",
            "last_updated":      _now(),
        })
