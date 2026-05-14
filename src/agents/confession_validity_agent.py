from __future__ import annotations

import json
import logging
from typing import Any

from src.Graph.states_and_schemas.state import AgentState
from src.Graph.states_and_schemas.Agents_output_models import (
    ConfessionAssessment,
    ConfessionAdmissibilityStatus,
    CoercionType,
)
from src.agents.agent_base import AgentBase

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Prompt
# ─────────────────────────────────────────────────────────────────────────────

CONFESSION_VALIDITY_PROMPT = """
أنت خبير قانوني متخصص في قانون الإجراءات الجنائية المصري، مهمتك تقييم مشروعية
وحجية كل اعتراف وارد في ملف القضية، وفقاً للقانون والمبادئ القضائية المستقرة.

## الأطر القانونية الواجبة التطبيق
1. المادة 302 إ.ج: لا يجوز الحكم بالإدانة استناداً إلى الاعتراف وحده
   إذا كان منفرداً بلا قرائن أو أدلة مادية مؤيدة.
2. المادة 40 إ.ج: حق المتهم في الصمت والاستعانة بمحامٍ.
3. الاعتراف المنتزَع بالإكراه أو التهديد باطل ولا يُعتدّ به.
4. يُشترط أن يؤخذ الاعتراف أمام جهة مختصة (نيابة عامة أو قاضٍ).
5. يُراعى الوقت المنقضي بين القبض والاعتراف كمؤشر على الإكراه.

## سياق القضية
{context}

## التعليمات
لكل اعتراف وارد في قائمة confessions أعلاه، أنتج تقييماً مستقلاً.
أرجع JSON واحداً بالهيكل الآتي بالضبط:

```json
{{
  "confession_assessments": [
    {{
      "defendant_name": "اسم المتهم",
      "confession_date": "تاريخ الاعتراف أو null",
      "confession_text_summary": "ملخص مضمون الاعتراف",
      "coercion_alleged": true,
      "coercion_type": "إكراه جسدي | إكراه نفسي | قبض غير مشروع | احتجاز مطوّل | لا يوجد | غير محدد",
      "coercion_evidence": "وصف الدليل على الإكراه أو null",
      "taken_before_competent_authority": true,
      "miranda_rights_observed": true,
      "time_since_arrest_hours": 5.0,
      "procedural_notes": "ملاحظات إجرائية أو null",
      "standalone_confession": false,
      "corroborating_evidence": ["دليل مادي مؤيد 1"],
      "admissibility_status": "مقبول | مرفوض — إكراه | مرفوض — خلل إجرائي | مرفوض — دليل وحيد بلا سند مادي | مقبول بشروط | يحتاج دليلاً تعزيزياً",
      "legal_reasoning": "التسبيب القانوني التفصيلي لقرار القبول أو الرفض",
      "impact_on_case": "يُعزز الإدانة | يُضعف الادعاء | يستوجب الاستبعاد"
    }}
  ],
  "overall_confession_impact": "ملخص الأثر الكلي للاعترافات على سير القضية"
}}
```

## قواعد صارمة
- أجب بـ JSON فقط بلا أي نص خارجه.
- استخدم اللغة العربية في جميع القيم النصية.
- إذا لم تتوفر اعترافات، أرجع confession_assessments كقائمة فارغة [].
- تحقق من كل اعتراف على حدة — لا تُعمّم.
- coercion_type يجب أن يكون إحدى القيم المحددة أعلاه بالضبط.
- admissibility_status يجب أن يكون إحدى القيم المحددة أعلاه بالضبط.
""".strip()


# ─────────────────────────────────────────────────────────────────────────────
# Agent
# ─────────────────────────────────────────────────────────────────────────────

class ConfessionValidityAgent(AgentBase):
    """
    يُقيّم مشروعية وحجية كل اعتراف ويُنتج:
        - state.confession_assessments: List[ConfessionAssessment]
    """

    AGENT_KEY = "confession_validity"

    # خريطة ترجمة القيم النصية إلى Enums
    _COERCION_MAP: dict[str, CoercionType] = {
        "إكراه جسدي":       CoercionType.PHYSICAL,
        "إكراه نفسي":       CoercionType.PSYCHOLOGICAL,
        "قبض غير مشروع":   CoercionType.ILLEGAL_ARREST,
        "احتجاز مطوّل":    CoercionType.PROLONGED_DETENTION,
        "لا يوجد":          CoercionType.NONE,
        "غير محدد":         CoercionType.UNKNOWN,
    }

    _ADMISSIBILITY_MAP: dict[str, ConfessionAdmissibilityStatus] = {
        "مقبول":                            ConfessionAdmissibilityStatus.ADMISSIBLE,
        "مرفوض — إكراه":                  ConfessionAdmissibilityStatus.INADMISSIBLE_COERCED,
        "مرفوض — خلل إجرائي":            ConfessionAdmissibilityStatus.INADMISSIBLE_PROCEDURAL,
        "مرفوض — دليل وحيد بلا سند مادي": ConfessionAdmissibilityStatus.INADMISSIBLE_SOLE_EVIDENCE,
        "مقبول بشروط":                     ConfessionAdmissibilityStatus.CONDITIONALLY_ADMISSIBLE,
        "يحتاج دليلاً تعزيزياً":          ConfessionAdmissibilityStatus.REQUIRES_CORROBORATION,
    }

    def __init__(self) -> None:
        super().__init__(
            model_config_key="CONFESSION_VALIDITY_MODEL",
            temp_config_key="CONFESSION_VALIDITY_TEMP",
            prompt=CONFESSION_VALIDITY_PROMPT,
        )

    # ── public entry point ────────────────────────────────────────

    def run(self, state: AgentState) -> AgentState:
        logger.info(
            "▶ [%s] بدء تقييم الاعترافات — القضية: %s | عدد الاعترافات: %d",
            self.AGENT_KEY, state.case_id, len(state.confessions),
        )

        # لا توجد اعترافات → تخطٍّ آمن
        if not state.confessions:
            logger.info("[%s] لا توجد اعترافات — تخطٍّ.", self.AGENT_KEY)
            return self._empty_update(state, self.AGENT_KEY, {})

        context = self._build_context(state)
        raw_output = self._call_llm(context)
        parsed = self._parse_llm_json(raw_output)

        assessments = self._extract_assessments(parsed)

        updates: dict[str, Any] = {"confession_assessments": assessments}

        logger.info(
            "✔ [%s] اكتمل — %d اعتراف مقيَّم", self.AGENT_KEY, len(assessments)
        )

        base_state = self._empty_update(state, self.AGENT_KEY, updates)
        return base_state.model_copy(update=updates)

    # ── private helpers ───────────────────────────────────────────

    def _build_context(self, state: AgentState) -> str:
        ctx: dict[str, Any] = {
            "case_id":     state.case_id,
            "case_number": state.case_number,
            "defendants": [self._safe_dump(d) for d in state.defendants],
            "confessions": [self._safe_dump(c) for c in state.confessions],
            # الأدلة المادية لتقييم standalone_confession
            "evidences":  [self._safe_dump(e) for e in state.evidences],
            "lab_reports": [self._safe_dump(r) for r in state.lab_reports],
            # المشكلات الإجرائية قد تشمل خلل أخذ الاعتراف
            "procedural_issues": [self._safe_dump(p) for p in state.procedural_issues],
            # نتائج تحليل الأدلة إن وُجدت
            "evidence_scores": [self._safe_dump(s) for s in state.evidence_scores],
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
                raw["coercion_type"] = self._COERCION_MAP.get(
                    raw.get("coercion_type", "غير محدد"),
                    CoercionType.UNKNOWN,
                )
                raw["admissibility_status"] = self._ADMISSIBILITY_MAP.get(
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