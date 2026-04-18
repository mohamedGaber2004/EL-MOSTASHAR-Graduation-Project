from .agent_base import AgentBase
from src.LLMs import get_llm_qwn
from langchain_core.messages import HumanMessage, SystemMessage
from src.Prompts.defense_analysis_agent import DEFENSE_ANALYSIS_AGENT_PROMPT
from src.Graph.graph_helpers import _parse_llm_json, _now, _agent_context

class DefenseAnalystAgent(AgentBase):
    def __init__(self):
        super().__init__("DEFENSE_ANALYST_MODEL", "DEFENSE_ANALYST_TEMP", DEFENSE_ANALYSIS_AGENT_PROMPT)

    def run(self, state):
        self.log(f"🛡️  DefenseAnalyst — {len(state.defense_documents)} defense documents")
        prompt = f"""
حلل دفوع المتهم التالية:

{_agent_context(state, "defense_analyst")}

لكل دفع:
1. صنّفه (شكلي / موضوعي).
2. حدد سنده القانوني.
3. قيّم قوته في مواجهة الأدلة المتاحة.
4. ارتباطه بأي بطلان إجرائي.
"""
        self._llm_invoke_with_retries(
            get_llm_qwn(self.model_name, self.temperature),
            [SystemMessage(content=self.prompt), HumanMessage(content=prompt)]
        )
        # ...existing code for building defense_analysis...
        audit_nullities  = state.agent_outputs.get("procedural_auditor", {}).get("critical_nullities", [])
        all_formal       = [d for doc in state.defense_documents for d in doc.formal_defenses]
        all_substantive  = [d for doc in state.defense_documents for d in doc.substantive_defenses]
        all_principles   = [p for doc in state.defense_documents for p in doc.supporting_principles]
        any_alibi        = any(doc.alibi_claimed for doc in state.defense_documents)
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
                "claimed":               any_alibi,
                "supported_by_evidence": False,
                "notes":                 "يستلزم تحقيقًا في أدلة دفاع الحضور" if any_alibi else "",
            },
            "supporting_principles":    all_principles,
            "overall_defense_strength": "قوي" if audit_nullities else "متوسط" if (all_formal or all_substantive) else "ضعيف",
            "critical_defense_points":  audit_nullities + all_formal[:2],
        }
        return state.model_copy(update={
            "agent_outputs": {**state.agent_outputs, "defense_analyst": defense_analysis},
            "completed_agents": state.completed_agents + ["defense_analyst"],
            "current_agent": "defense_analyst",
            "last_updated": _now(),
        })
