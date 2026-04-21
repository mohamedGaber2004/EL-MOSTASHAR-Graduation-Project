from langchain_core.messages import HumanMessage, SystemMessage
from .agent_base import AgentBase
from src.LLMs import get_llm_llama
from src.Graph.graph_helpers import (
    _parse_llm_json
)

from src.Prompts import (
    PROCEDURAL_AUDITOR_AGENT_PROMPT
)

# ─────────────────────────────────────────────
# المدقق الإجرائي
# ─────────────────────────────────────────────
class ProceduralAuditorAgent(AgentBase):
    def __init__(self):
        super().__init__("PROCEDURAL_AUDITOR_MODEL", "PROCEDURAL_AUDITOR_TEMP", PROCEDURAL_AUDITOR_AGENT_PROMPT)
 
    def run(self, state):
        self.log(f"⚖️ المدقق الإجرائي — {len(state.procedural_issues)} مسألة")
        prompt = (
            f"راجع الإجراءات وأصدر تقرير البطلان:\n"
            f"{state.agent_outputs.get('data_ingestion', {})}\n\n"
            "م41 دستور: القبض يستلزم أمرًا أو تلبسًا.\n"
            "م45 إجراءات: التفتيش يستلزم إذنًا مسببًا.\n"
            "م135 إجراءات: الاستجواب يستلزم إخطار النيابة خلال 24 ساعة.\n"
            "الاعتراف الباطل لا يُعتد به حتى لو صادقه المتهم لاحقًا."
        )
        response = self._llm_invoke_with_retries(
            get_llm_llama(self.model_name, self.temperature),
            [SystemMessage(content=self.prompt), HumanMessage(content=prompt)],
        )
        audit_report = _parse_llm_json(response.content)
        return self._empty_update(state, "procedural_auditor", audit_report)
