from __future__ import annotations

import json
import logging
from typing import Any

from src.Graph.state import AgentState
from src.agents.witness_credibility_agent.witness_credibility_prompt import WITNESS_CREDIBILITY_PROMPT
from src.agents.witness_credibility_agent.witness_credibility_output_model import (
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

    # ── private helpers ───────────────────────────────────────────

    def _build_context(self, state: AgentState) -> str:
        ctx: dict[str, Any] = {
            "case_id":     state.case_id,
            "case_number": state.case_number,
            "incidents": [
                {
                    "description": getattr(i, "incident_description", None),
                    "date":        getattr(i, "incident_date", None),
                    "location":    getattr(i, "location", None),
                }
                for i in (state.incidents or [])
            ],
            "witness_statements": [self._safe_dump(w) for w in state.witness_statements],
            "evidence_scores":    [self._safe_dump(s) for s in state.evidence_scores],
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
        try:
            raw_output = self._call_llm(context)
        except Exception as e:
            logger.error("WitnessCredibilityAgent LLM failed: %s ", e)
            raw_output = {}

        parsed = self._parse_agent_json(raw_output)
        scores = self._extract_scores(parsed)
        updates: dict[str, Any] = {"witness_credibility_scores": scores}

        logger.info(
            "✔ [%s] اكتمل — %d شاهد مقيَّم", self.AGENT_KEY, len(scores)
        )

        return self._empty_update(state, self.AGENT_KEY, updates)