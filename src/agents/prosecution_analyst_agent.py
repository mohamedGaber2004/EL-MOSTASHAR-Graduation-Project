from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.Graph.states_and_schemas.state import AgentState
from src.agents.agent_base import AgentBase
from src.Prompts.prosecution_analyst_agent import PROSECUTION_ANALYST_PROMPT
from src.Graph.states_and_schemas.Agents_output_models import (
    ProsecutionNarrative,
    ProsecutionArgument,
    ProsecutionArgumentStrength,
)

logger = logging.getLogger(__name__)


class ProsecutionAnalystAgent(AgentBase):

    AGENT_KEY = "prosecution_analyst"

    def __init__(self) -> None:
        super().__init__(
            model_config_key="PROSECUTION_ANALYST_MODEL",
            temp_config_key="PROSECUTION_ANALYST_TEMP",
            prompt=PROSECUTION_ANALYST_PROMPT,
        )

    # ── context builder ───────────────────────────────────────────

    def _build_prosecution_context(self, state: AgentState) -> str:
        audit = state.procedural_audit
        ctx: dict[str, Any] = {
            "case_id":     state.case_id,
            "case_number": state.case_number,
            "court":       state.court,
            "defendants": [
                {"name": getattr(d, "name", None), "id": getattr(d, "id", None)}
                for d in (state.defendants or [])
            ],
            "charges":   [self._safe_dump(c) for c in state.charges],
            "incidents": [self._safe_dump(i) for i in state.incidents],
            "evidence_scores":            [self._safe_dump(s) for s in state.evidence_scores],
            "chain_of_custody_issues":    state.chain_of_custody_issues or [],
            "confession_assessments":     [self._safe_dump(c) for c in state.confession_assessments],
            "witness_credibility_scores": [self._safe_dump(w) for w in state.witness_credibility_scores],
            "case_articles": [
                a.get("article_number", str(a)) if isinstance(a, dict) else a
                for a in (state.case_articles or [])
            ],
            "applied_principles":       [self._safe_dump(p) for p in state.applied_principles],
            "overall_audit_assessment": audit.overall_assessment,
            "critical_nullities":       audit.critical_nullities or [],
        }
        return self.prompt.format(context=json.dumps(ctx, ensure_ascii=False, indent=2))

    # ── LLM call ──────────────────────────────────────────────────

    def _call_llm(self, prompt_text: str) -> str:
        messages = [
            SystemMessage(content=(
                "أنت محلل اتهامي قانوني متخصص في القانون الجنائي المصري. "
                "تجيب بـ JSON فقط. الجواب يبدأ بـ { مباشرة بلا أي نص قبله."
            )),
            HumanMessage(content=prompt_text),
        ]
        result = self._llm_invoke_with_retries(self._llm, messages)
        return result.content if hasattr(result, "content") else str(result)

    # ── JSON normalization ────────────────────────────────────────
    def _normalize_parsed(self, parsed) -> dict:
        """
        Handles all the ways LLMs return malformed structure:
        - list containing the correct dict  → unwrap
        - list of arguments directly        → wrap
        - non-dict entirely                 → empty fallback
        """
        if isinstance(parsed, dict):
            return parsed

        if isinstance(parsed, list):
            # Case 1: [{ "prosecution_narrative": ..., "prosecution_arguments": ... }]
            if (
                len(parsed) == 1
                and isinstance(parsed[0], dict)
                and (
                    "prosecution_narrative" in parsed[0]
                    or "prosecution_arguments" in parsed[0]
                )
            ):
                logger.info("[%s] unwrapped single-element list", self.AGENT_KEY)
                return parsed[0]

            # Case 2: list of argument dicts directly
            logger.warning("[%s] LLM returned bare list — treating as arguments", self.AGENT_KEY)
            return {
                "prosecution_narrative": {},
                "prosecution_arguments": parsed,
            }

        logger.error("[%s] unparseable output type=%s — empty fallback", self.AGENT_KEY, type(parsed))
        return {}

    # ── output extraction ─────────────────────────────────────────

    def _extract_outputs(
        self, parsed: dict
    ) -> tuple[ProsecutionNarrative | None, list[ProsecutionArgument]]:

        narrative: ProsecutionNarrative | None = None
        arguments: list[ProsecutionArgument]   = []

        strength_map = {
            "قوية":   ProsecutionArgumentStrength.STRONG,
            "متوسطة": ProsecutionArgumentStrength.MODERATE,
            "ضعيفة":  ProsecutionArgumentStrength.WEAK,
        }

        # ── narrative ─────────────────────────────────────────────
        raw_narrative = parsed.get("prosecution_narrative") or {}
        if raw_narrative and isinstance(raw_narrative, dict):
            try:
                raw_narrative["overall_strength"] = strength_map.get(
                    raw_narrative.get("overall_strength", "متوسطة"),
                    ProsecutionArgumentStrength.MODERATE,
                )
                narrative = ProsecutionNarrative(**{
                    k: v for k, v in raw_narrative.items() if v is not None
                })
            except Exception as exc:
                logger.warning("[%s] فشل بناء ProsecutionNarrative: %s", self.AGENT_KEY, exc)

        # ── arguments ─────────────────────────────────────────────
        for raw_arg in (parsed.get("prosecution_arguments") or []):
            if not isinstance(raw_arg, dict):
                continue
            # skip if this is actually the full response dict mistakenly placed here
            if "prosecution_narrative" in raw_arg:
                logger.warning("[%s] skipping nested full-response dict in arguments", self.AGENT_KEY)
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

    # ── entry point ───────────────────────────────────────────────

    def run(self, state: AgentState) -> AgentState:
        logger.info("▶ [%s] بدء تحليل ملف الاتهام — القضية: %s", self.AGENT_KEY, state.case_id)

        context    = self._build_prosecution_context(state)
        raw_output = self._call_llm(context)
        parsed     = self._normalize_parsed(self._parse_agent_json(raw_output))

        narrative, arguments = self._extract_outputs(parsed)

        logger.info(
            "✔ [%s] اكتمل — قوة الاتهام: %s | حجج: %d",
            self.AGENT_KEY,
            narrative.overall_strength if narrative else "—",
            len(arguments),
        )

        return self._empty_update(state, self.AGENT_KEY, {
            "prosecution_theory":    narrative,
            "prosecution_arguments": arguments,
        })