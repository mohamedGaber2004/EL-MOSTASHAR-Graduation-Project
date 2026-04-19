from .agent_base import AgentBase
from src.LLMs import get_llm_oss
from langchain_core.messages import HumanMessage, SystemMessage
from src.Prompts.evidence_analyst_agent import EVIDENCE_ANALYST_AGENT_PROMPT
from src.Graph.graph_helpers import _parse_llm_json, _now, _agent_context

class EvidenceAnalystAgent(AgentBase):
    def __init__(self):
        super().__init__("EVIDENCE_ANALYST_MODEL", "EVIDENCE_ANALYST_TEMP", EVIDENCE_ANALYST_AGENT_PROMPT)

    def run(self, state):
        self.log(f"🔍 EvidenceAnalyst — {len(state.evidences)} evidences, {len(state.witness_statements)} witnesses, {len(state.confessions)} confessions")
        audit = state.agent_outputs.get("procedural_auditor", {})
        invalidated_ids = [
            eid
            for result in audit.get("audit_results", [])
            if result.get("status") == "باطل"
            for eid in result.get("affected_evidence_ids", [])
        ]
        prompt = f"""
حلل الأدلة التالية وأصدر مصفوفة الإثبات:

{_agent_context(state, "evidence_analyst")}

الأدلة الباطلة إجرائيًا (يجب استبعادها): {invalidated_ids}

لكل واقعة:
1. حدد الأدلة الداعمة وقوتها.
2. حدد التناقضات والتعارضات.
3. استبعد الأدلة الباطلة إجرائيًا.
4. احسب درجة الإثبات الإجمالية.
"""
        response = self._llm_invoke_with_retries(
            get_llm_oss(self.model_name, self.temperature),
            [SystemMessage(content=self.prompt), HumanMessage(content=prompt)]
        )
        evidence_matrix = _parse_llm_json(response.content)
        return state.model_copy(update={
            "agent_outputs": {**state.agent_outputs, "evidence_analyst": evidence_matrix},
            "completed_agents": state.completed_agents + ["evidence_analyst"],
            "current_agent": "evidence_analyst",
            "last_updated": _now(),
        })
