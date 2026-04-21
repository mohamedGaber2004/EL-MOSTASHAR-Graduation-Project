from langchain_core.messages import HumanMessage, SystemMessage
from .agent_base import AgentBase
from src.LLMs import get_llm_qwn
from src.Graph.graph_helpers import (
    _agent_context
)
from src.Prompts import (
    DEFENSE_ANALYSIS_AGENT_PROMPT
)

# ─────────────────────────────────────────────
# محلل الدفوع
# ─────────────────────────────────────────────

class DefenseAnalystAgent(AgentBase):
    def __init__(self):
        super().__init__("DEFENSE_ANALYST_MODEL", "DEFENSE_ANALYST_TEMP", DEFENSE_ANALYSIS_AGENT_PROMPT)
 
    def run(self, state):
        self.log(f"🛡️ محلل الدفوع — {len(state.defense_documents)} مستند")
        audit_nullities = state.agent_outputs.get("procedural_auditor", {}).get("critical_nullities", [])
        all_formal      = [d for doc in state.defense_documents for d in doc.formal_defenses]
        all_substantive = [d for doc in state.defense_documents for d in doc.substantive_defenses]
        all_principles  = [p for doc in state.defense_documents for p in doc.supporting_principles]
        any_alibi       = any(doc.alibi_claimed for doc in state.defense_documents)
 
        prompt = (
            f"حلل دفوع المتهم:\n{_agent_context(state, 'defense_analyst')}\n\n"
            "لكل دفع: صنّفه، حدد سنده القانوني، قيّم قوته، واربطه بأي بطلان إجرائي."
        )
        self._llm_invoke_with_retries(
            get_llm_qwn(self.model_name, self.temperature),
            [SystemMessage(content=self.prompt), HumanMessage(content=prompt)],
        )
 
        defense_analysis = {
            "formal_defenses": [
                {"defense": d, "legal_basis": "قانون الإجراءات الجنائية", "strength": "متوسط", "notes": ""}
                for d in all_formal
            ] + [
                {"defense": f"بطلان إجرائي — {n}", "legal_basis": "م 45 إجراءات", "strength": "قوي", "notes": ""}
                for n in audit_nullities
            ],
            "substantive_defenses": [
                {"defense": d, "legal_basis": "قانون العقوبات", "strength": "متوسط", "notes": ""}
                for d in all_substantive
            ],
            "alibi_analysis": {
                "claimed": any_alibi,
                "supported_by_evidence": False,
                "notes": "يستلزم تحقيقًا في أدلة الحضور" if any_alibi else "",
            },
            "supporting_principles":    all_principles,
            "overall_defense_strength": "قوي" if audit_nullities else ("متوسط" if (all_formal or all_substantive) else "ضعيف"),
            "critical_defense_points":  audit_nullities + all_formal[:2],
        }
        return self._empty_update(state, "defense_analyst", defense_analysis)
