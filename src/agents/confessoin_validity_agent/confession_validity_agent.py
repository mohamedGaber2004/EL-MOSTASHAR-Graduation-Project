from __future__ import annotations

import json
import logging
from typing import Any

from src.Graph.state import AgentState
from src.agents.confessoin_validity_agent.confession_validity_prompt import CONFESSION_VALIDITY_PROMPT
from src.agents.confessoin_validity_agent.confession_validity_output_model import ConfessionAssessment
from src.agents.confessoin_validity_agent.CV_enums import CV_Enums, CoercionType, ConfessionAdmissibilityStatus
from src.agents.agent_base.agent_base import AgentBase

logger = logging.getLogger(__name__)




# ─────────────────────────────────────────────────────────────────────────────
# Agent
# ─────────────────────────────────────────────────────────────────────────────

class ConfessionValidityAgent(AgentBase):
    """
    يُقيّم مشروعية وحجية كل اعتراف ويُنتج:
        - state.confession_assessments: List[ConfessionAssessment]
    """

    def __init__(self) -> None:
        super().__init__(
            model_config_key="CONFESSION_VALIDITY_MODEL",
            temp_config_key="CONFESSION_VALIDITY_TEMP",
            prompt=CONFESSION_VALIDITY_PROMPT,
        )
    
    # ── private helpers ───────────────────────────────────────────
    def _build_context(self, state: AgentState) -> str:
        audit      = state.procedural_audit
        violations = audit.violations if audit else []

        # ── confession-related violations only ────────────────────────────────
        confession_violations = [
            {
                "procedure_type":    p.procedure_type,
                "issue_description": p.issue_description,
                "nullity_type":      p.nullity_type,
                "nullity_effect":    p.nullity_effect,
            }
            for p in violations
            if p.procedure_type and "استجواب" in (p.procedure_type or "")
            or "اعتراف" in (p.issue_description or "")
        ]

        ctx: dict[str, Any] = {
            "case_id":              state.case_id,
            "case_number":          state.case_number,
            "defendants": [
                {"name": getattr(d, "name", None), "id": getattr(d, "id", None)}
                for d in (state.defendants or [])
            ],
            "confessions":          [self._safe_dump(c) for c in state.confessions],
            "criminal_proceedings": [self._safe_dump(p) for p in state.criminal_proceedings],
            "confession_related_violations": confession_violations,
        }
        return self.prompt.format(context=json.dumps(ctx, ensure_ascii=False, indent=2))

    def _call_llm(self, prompt_text: str) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [
            SystemMessage(content=(
                "أنت خبير قانوني في الإجراءات الجنائية المصرية. "
                "تجيب بـ JSON فقط بلا أي نص خارجه."
            )),
            HumanMessage(content=prompt_text),
        ]
        result = self._llm_invoke_with_retries(self._llm, messages)
        return result.content if hasattr(result, "content") else str(result)

    def _extract_assessments(self, parsed: dict) -> list[ConfessionAssessment]:
        raw_list = parsed.get("confession_assessments") or []
        assessments: list[ConfessionAssessment] = []

        for raw in raw_list:
            if not isinstance(raw, dict):
                continue
            try:
                # normalise enum fields
                raw["coercion_type"] = CV_Enums._COERCION_MAP.value.get(
                    raw.get("coercion_type", "غير محدد"),
                    CoercionType.UNKNOWN,
                )
                raw["admissibility_status"] = CV_Enums._ADMISSIBILITY_MAP.value.get(
                    raw.get("admissibility_status", "غير محدد"),
                    ConfessionAdmissibilityStatus.REQUIRES_CORROBORATION,
                )
                assessments.append(ConfessionAssessment(**{
                    k: v for k, v in raw.items() if v is not None
                }))
            except Exception as exc:
                logger.warning(
                    "[%s] فشل بناء ConfessionAssessment: %s | raw: %s",
                    self.AGENT_KEY, exc, str(raw)[:150],
                )

        return assessments

    # ── public entry point ────────────────────────────────────────
    def run(self, state: AgentState) -> AgentState:
        logger.info(
            "▶ [%s] بدء تقييم الاعترافات — القضية: %s | عدد الاعترافات: %d",
            CV_Enums.AGENT_KEY.value, state.case_id, len(state.confessions),
        )

        if not state.confessions:
            logger.info("[%s] لا توجد اعترافات — تخطٍّ.", CV_Enums.AGENT_KEY.value)
            return self._empty_update(state, CV_Enums.AGENT_KEY.value, {})

        context = self._build_context(state)
        try:
            raw_output = self._call_llm(context)
        except Exception as e:
            logger.error("ConfessionValidityAgent LLM failed: %s — using fallback", e)
            raw_output = {}

        if not isinstance(raw_output, str) or not raw_output.strip():
            logger.warning("ConfessionValidityAgent: empty/invalid output — using fallback")
            raw_output = ""

        parsed = self._parse_agent_json(raw_output)

        assessments = self._extract_assessments(parsed)

        updates: dict[str, Any] = {"confession_assessments": assessments}

        logger.info(
            "✔ [%s] اكتمل — %d اعتراف مقيَّم", CV_Enums.AGENT_KEY.value, len(assessments)
        )

        return self._empty_update(state, CV_Enums.AGENT_KEY.value, updates)



