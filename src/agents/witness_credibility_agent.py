from __future__ import annotations

import json
import logging
from typing import Any

from src.Graph.states_and_schemas.state import AgentState
from src.Prompts.witness_credibility_agent import WITNESS_CREDIBILITY_PROMPT
from src.Graph.states_and_schemas.Agents_output_models import (
    WitnessCredibility,
    WitnessReliabilityLevel,
)
from src.agents.agent_base import AgentBase

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Agent
# ─────────────────────────────────────────────────────────────────────────────

class WitnessCredibilityAgent(AgentBase):
    """
    يُقيّم موثوقية الشهود ويُنتج:
        - state.witness_credibility_scores: List[WitnessCredibility]
    """

    AGENT_KEY = "witness_credibility"

    _RELIABILITY_MAP: dict[str, WitnessReliabilityLevel] = {
        "عالية":    WitnessReliabilityLevel.HIGH,
        "متوسطة":   WitnessReliabilityLevel.MEDIUM,
        "منخفضة":   WitnessReliabilityLevel.LOW,
        "مستبعد":   WitnessReliabilityLevel.EXCLUDED,
    }

    def __init__(self) -> None:
        super().__init__(
            model_config_key="WITNESS_CREDIBILITY_MODEL",
            temp_config_key="WITNESS_CREDIBILITY_TEMP",
            prompt=WITNESS_CREDIBILITY_PROMPT,
        )

    # ── public entry point ────────────────────────────────────────

    def run(self, state: AgentState) -> AgentState:
        logger.info(
            "▶ [%s] بدء تقييم الشهود — القضية: %s | عدد الشهود: %d",
            self.AGENT_KEY, state.case_id, len(state.witness_statements),
        )

        if not state.witness_statements:
            logger.info("[%s] لا توجد شهادات — تخطٍّ.", self.AGENT_KEY)
            return self._empty_update(state, self.AGENT_KEY, {})

        context = self._build_context(state)
        raw_output = self._call_llm(context)
        parsed = self._parse_llm_json(raw_output)

        scores = self._extract_scores(parsed)

        updates: dict[str, Any] = {"witness_credibility_scores": scores}

        logger.info(
            "✔ [%s] اكتمل — %d شاهد مقيَّم", self.AGENT_KEY, len(scores)
        )

        base_state = self._empty_update(state, self.AGENT_KEY, updates)
        return base_state.model_copy(update=updates)

    # ── private helpers ───────────────────────────────────────────

    def _build_context(self, state: AgentState) -> str:
        ctx: dict[str, Any] = {
            "case_id":     state.case_id,
            "case_number": state.case_number,
            "defendants": [self._safe_dump(d) for d in state.defendants],
            "incidents":  [self._safe_dump(i) for i in state.incidents],
            "witness_statements": [self._safe_dump(w) for w in state.witness_statements],
            # مقارنة الشهادات بالأدلة المادية
            "evidences":   [self._safe_dump(e) for e in state.evidences],
            "lab_reports": [self._safe_dump(r) for r in state.lab_reports],
            # الاعترافات لكشف التعارض مع أقوال الشهود
            "confessions": [self._safe_dump(c) for c in state.confessions],
            # نتائج تقييم الأدلة إن سبقت
            "evidence_scores": [self._safe_dump(s) for s in state.evidence_scores],
        }
        return self.prompt.format(context=json.dumps(ctx, ensure_ascii=False, indent=2))

    def _call_llm(self, prompt_text: str) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [
            SystemMessage(content=(
                "أنت خبير في تقييم الشهادات وفقاً للقانون الجنائي المصري. "
                "تجيب بـ JSON فقط بلا أي نص خارجه."
            )),
            HumanMessage(content=prompt_text),
        ]
        result = self._llm_invoke_with_retries(self._llm, messages)
        return result.content if hasattr(result, "content") else str(result)

    def _extract_scores(self, parsed: dict) -> list[WitnessCredibility]:
        raw_list = parsed.get("witness_credibility_scores") or []
        scores: list[WitnessCredibility] = []

        for raw in raw_list:
            if not isinstance(raw, dict):
                continue
            try:
                raw["reliability_level"] = self._RELIABILITY_MAP.get(
                    raw.get("reliability_level", "متوسطة"),
                    WitnessReliabilityLevel.MEDIUM,
                )
                scores.append(WitnessCredibility(**{
                    k: v for k, v in raw.items() if v is not None
                }))
            except Exception as exc:
                logger.warning(
                    "[%s] فشل بناء WitnessCredibility: %s | raw: %s",
                    self.AGENT_KEY, exc, str(raw)[:150],
                )

        return scores