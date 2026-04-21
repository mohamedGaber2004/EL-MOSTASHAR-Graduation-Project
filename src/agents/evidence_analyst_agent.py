from langchain_core.messages import HumanMessage, SystemMessage
from .agent_base import AgentBase
from src.LLMs import get_llm_oss
from src.Graph.graph_helpers import (
    _parse_llm_json, _agent_context
)
from src.Utils.agent_output_utils import clean_agent_output
from src.Prompts import (
    EVIDENCE_ANALYST_AGENT_PROMPT
)

# ─────────────────────────────────────────────
# محلل الأدلة
# ─────────────────────────────────────────────

class EvidenceAnalystAgent(AgentBase):
    def __init__(self):
        super().__init__("EVIDENCE_ANALYST_MODEL", "EVIDENCE_ANALYST_TEMP", EVIDENCE_ANALYST_AGENT_PROMPT)
 
    def run(self, state):
        self.log(f"🔍 محلل الأدلة — {len(state.evidences)} دليل")
        invalidated_ids = [
            eid
            for r in state.agent_outputs.get("procedural_auditor", {}).get("audit_results", [])
            if r.get("status") == "باطل"
            for eid in r.get("affected_evidence_ids", [])
        ]
        prompt = (
            f"حلل الأدلة وأصدر مصفوفة الإثبات:\n{_agent_context(state, 'evidence_analyst')}\n\n"
            f"الأدلة الباطلة (استبعدها): {invalidated_ids}\n"
            "لكل واقعة: الأدلة الداعمة وقوتها، التناقضات، درجة الإثبات."
        )
        response = self._llm_invoke_with_retries(
            get_llm_oss(self.model_name, self.temperature),
            [SystemMessage(content=self.prompt), HumanMessage(content=prompt)],
        )
        evidence_matrix = clean_agent_output(_parse_llm_json(response.content))
        return self._empty_update(state, "evidence_analyst", evidence_matrix)
