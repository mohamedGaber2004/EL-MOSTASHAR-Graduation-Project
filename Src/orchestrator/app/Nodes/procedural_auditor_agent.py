from .agent_base import AgentBase
from app.llms import get_llm_llama
from langchain_core.messages import HumanMessage, SystemMessage
from app.Prompts.procedural_auditor_agent import PROCEDURAL_AUDITOR_AGENT_PROMPT
from app.Graph.graph_helpers import _parse_llm_json, _now

class ProceduralAuditorAgent(AgentBase):
    def __init__(self):
        super().__init__("PROCEDURAL_AUDITOR_MODEL", "PROCEDURAL_AUDITOR_TEMP", PROCEDURAL_AUDITOR_AGENT_PROMPT)

    def run(self, state):
        self.log(f"⚖️  ProceduralAuditor — Reviewing {len(state.procedural_issues)} procedural issues")
        prompt = f"""
راجع الإجراءات التالية وأصدر تقرير البطلان:

{state.agent_outputs.get('data_ingestion', {})}

تذكر:
- م 41 دستور: القبض يستلزم أمرًا قضائيًا أو تلبسًا.
- م 45 إجراءات: التفتيش يستلزم إذنًا مسببًا.
- م 135 إجراءات: الاستجواب يستلزم إخطار النيابة خلال 24 ساعة.
- الاعتراف الباطل لا يُعتد به حتى لو صادقه المتهم لاحقًا.
"""
        response = self._llm_invoke_with_retries(
            get_llm_llama(self.model_name, self.temperature),
            [SystemMessage(content=self.prompt), HumanMessage(content=prompt)]
        )
        audit_report = _parse_llm_json(response.content)
        return state.model_copy(update={
            "agent_outputs": {**state.agent_outputs, "procedural_auditor": audit_report},
            "completed_agents": state.completed_agents + ["procedural_auditor"],
            "current_agent": "procedural_auditor",
            "last_updated": _now(),
        })
