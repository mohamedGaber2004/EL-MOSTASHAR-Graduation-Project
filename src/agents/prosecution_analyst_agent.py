from __future__ import annotations

import json
import logging
from typing import Any

from src.Graph.states_and_schemas.state import AgentState
from src.agents.agent_base import AgentBase
from src.Prompts.prosecution_analyst_agent import PROSECUTION_ANALYST_PROMPT
from src.Graph.states_and_schemas.Agents_output_models import (
    ProsecutionNarrative,
    ProsecutionArgument,
    ProsecutionArgumentStrength,
)


logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Agent
# ─────────────────────────────────────────────────────────────────────────────

class ProsecutionAnalystAgent(AgentBase):
    """
    يحلل ملف القضية من منظور الاتهام ويُنتج:
        - state.prosecution_theory   : ProsecutionNarrative
        - state.prosecution_arguments: List[ProsecutionArgument]
    """

    AGENT_KEY = "prosecution_analyst"

    def __init__(self) -> None:
        super().__init__(
            model_config_key="PROSECUTION_ANALYST_MODEL",
            temp_config_key="PROSECUTION_ANALYST_TEMP",
            prompt=PROSECUTION_ANALYST_PROMPT,
        )

    # ── public entry point ────────────────────────────────────────

    def run(self, state: AgentState) -> AgentState:
        logger.info("▶ [%s] بدء تحليل ملف الاتهام — القضية: %s", self.AGENT_KEY, state.case_id)

        context = self._build_prosecution_context(state)
        raw_output = self._call_llm(context)
        parsed = self._parse_llm_json(raw_output)

        narrative, arguments = self._extract_outputs(parsed)

        updates: dict[str, Any] = {
            "prosecution_theory":    narrative,
            "prosecution_arguments": arguments,
        }

        logger.info(
            "✔ [%s] اكتمل — قوة الاتهام: %s | حجج: %d",
            self.AGENT_KEY,
            narrative.overall_strength if narrative else "—",
            len(arguments),
        )

        base_state = self._empty_update(state, self.AGENT_KEY, updates)
        return base_state.model_copy(update=updates)

    # ── private helpers ───────────────────────────────────────────

    def _build_prosecution_context(self, state: AgentState) -> str:
        """
        يبني السياق المرسَل إلى النموذج مُضمِّناً جميع العناصر
        التي تحتاجها نظرية الاتهام.
        """
        ctx: dict[str, Any] = {
            "case_id":     state.case_id,
            "case_number": state.case_number,
            "court":       state.court,
            "defendants": [self._safe_dump(d) for d in state.defendants],
            "charges":    [self._safe_dump(c) for c in state.charges],
            "incidents":  [self._safe_dump(i) for i in state.incidents],
            "evidences":  [self._safe_dump(e) for e in state.evidences],
            "witness_statements": [self._safe_dump(w) for w in state.witness_statements],
            "confessions":        [self._safe_dump(c) for c in state.confessions],
            "lab_reports":        [self._safe_dump(r) for r in state.lab_reports],
            "criminal_records":   [self._safe_dump(r) for r in state.criminal_records],
            # نتائج العملاء السابقين إن وُجدت
            "procedural_issues":  [self._safe_dump(p) for p in state.procedural_issues],
            "evidence_scores":    [self._safe_dump(s) for s in state.evidence_scores],
            "applied_principles": [self._safe_dump(p) for p in state.applied_principles],
        }
        return self.prompt.format(context=json.dumps(ctx, ensure_ascii=False, indent=2))

    def _call_llm(self, prompt_text: str) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [
            SystemMessage(content=(
                "أنت محلل اتهامي قانوني متخصص في القانون الجنائي المصري. "
                "تجيب بـ JSON فقط بلا أي نص خارجه."
            )),
            HumanMessage(content=prompt_text),
        ]
        result = self._llm_invoke_with_retries(self._llm, messages)

        # LangChain models return an AIMessage; raw string content is in .content
        if hasattr(result, "content"):
            return result.content
        return str(result)

    def _extract_outputs(
        self,
        parsed: dict,
    ) -> tuple[ProsecutionNarrative | None, list[ProsecutionArgument]]:
        """
        يحوّل الـ dict المُحلَّل إلى Pydantic models.
        يُعيد (None, []) عند الفشل لتجنب توقف الـ pipeline.
        """
        narrative: ProsecutionNarrative | None = None
        arguments: list[ProsecutionArgument] = []

        # ── ProsecutionNarrative ──────────────────────────────────
        raw_narrative = parsed.get("prosecution_narrative") or {}
        if raw_narrative:
            try:
                # normalise overall_strength to enum
                strength_raw = raw_narrative.get("overall_strength", "متوسطة")
                strength_map = {
                    "قوية":   ProsecutionArgumentStrength.STRONG,
                    "متوسطة": ProsecutionArgumentStrength.MODERATE,
                    "ضعيفة":  ProsecutionArgumentStrength.WEAK,
                }
                raw_narrative["overall_strength"] = strength_map.get(
                    strength_raw, ProsecutionArgumentStrength.MODERATE
                )
                narrative = ProsecutionNarrative(**{
                    k: v for k, v in raw_narrative.items() if v is not None
                })
            except Exception as exc:
                logger.warning("[%s] فشل بناء ProsecutionNarrative: %s", self.AGENT_KEY, exc)

        # ── ProsecutionArguments ──────────────────────────────────
        raw_args = parsed.get("prosecution_arguments") or []
        strength_map = {
            "قوية":   ProsecutionArgumentStrength.STRONG,
            "متوسطة": ProsecutionArgumentStrength.MODERATE,
            "ضعيفة":  ProsecutionArgumentStrength.WEAK,
        }
        for raw_arg in raw_args:
            if not isinstance(raw_arg, dict):
                continue
            try:
                raw_arg["strength"] = strength_map.get(
                    raw_arg.get("strength", "متوسطة"),
                    ProsecutionArgumentStrength.MODERATE,
                )
                arguments.append(ProsecutionArgument(**{
                    k: v for k, v in raw_arg.items() if v is not None
                }))
            except Exception as exc:
                logger.warning(
                    "[%s] فشل بناء ProsecutionArgument: %s | raw: %s",
                    self.AGENT_KEY, exc, str(raw_arg)[:120],
                )

        return narrative, arguments