from .agent_base import AgentBase
from src.LLMs import get_llm_llama
from langchain_core.messages import HumanMessage, SystemMessage
from src.Prompts.judge_agent import JUDGE_AGENT_PROMPT
from src.Graph.graph_helpers import _parse_llm_json, _now, _agent_context
from src.Utils import VerdictType

class JudgeAgent(AgentBase):
    def __init__(self):
        super().__init__("JUDGE_MODEL", "JUDGE_TEMP", JUDGE_AGENT_PROMPT)

    def run(self, state):
        self.log(f"👨‍⚖️  JudgeAgent — Deliberating on case {state.case_id}")
        prompt = f"""
أصدر مسودة حكم مسببة في القضية التالية:

{_agent_context(state, "judge")}

تأكد من:
1. مناقشة كل دليل على حدة.
2. الرد الصريح على كل دفع من دفوع الدفاع.
3. الإشارة الصريحة لأي نقص أو شك أو قصور.
4. عدم الإدانة إلا باليقين — الشك يُفسَّر لصالح المتهم.
"""
        response = self._llm_invoke_with_retries(
            get_llm_llama(self.model_name, self.temperature),
            [SystemMessage(content=self.prompt), HumanMessage(content=prompt)]
        )
        judgment = _parse_llm_json(response.content)
        verdict_text = judgment.get("verdict", "")
        verdict = VerdictType(verdict_text) if verdict_text else VerdictType.ACQUITTAL
        return state.model_copy(update={
            "agent_outputs": {**state.agent_outputs, "judge": judgment},
            "suggested_verdict": verdict,
            "confidence_score": judgment.get("confidence_score", 0.0),
            "reasoning_trace": [{"agent": "judge", "reasoning": judgment.get("verdict_reasoning", "")}],
            "completed_agents": state.completed_agents + ["judge"],
            "current_agent": "judge",
            "last_updated": _now(),
        })
