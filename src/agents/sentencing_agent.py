from __future__ import annotations

import json
import logging
from typing import Any

from src.Graph.states_and_schemas.state import AgentState
from src.Graph.states_and_schemas.Agents_output_models import CivilClaim, CivilClaimStatus
from src.agents.agent_base import AgentBase

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Prompt
# ─────────────────────────────────────────────────────────────────────────────

SENTENCING_PROMPT = """
أنت مستشار قانوني متخصص في تقدير العقوبات وفقاً للقانون الجنائي المصري.
مهمتك تحليل ملف القضية وتقديم توصيات تقديرية مفصّلة للقاضي قبل إصدار الحكم.

## مهامك
1. تحديد الظروف المشددة (قانونية وقضائية) وأثرها.
2. تحديد الظروف المخففة وأثرها، بما فيها المادة 17 عقوبات.
3. تقييم تأثير السوابق الجنائية على مقدار العقوبة.
4. تحديد كيفية التعامل مع تعدد الجرائم (المادة 32 عقوبات).
5. إعداد توصية بخصوص الدعوى المدنية التبعية وتقدير التعويض.
6. بناء خريطة واضحة: لكل تهمة → "إدانة" أو "براءة" بناءً على تحليل سابقيك.

## سياق القضية الكامل
{context}

## التعليمات
أرجع JSON واحداً بالهيكل الآتي بالضبط:

```json
{{
  "aggravating_factors": [
    "ظرف مشدد 1 مع المرجع القانوني",
    "ظرف مشدد 2"
  ],
  "mitigating_factors": [
    "ظرف مخفف 1",
    "ظرف مخفف 2"
  ],
  "applicable_article_17": true,
  "article_17_reasoning": "سبب تطبيق أو عدم تطبيق المادة 17 عقوبات",
  "recidivism_impact": "أثر السوابق الجنائية على تقدير العقوبة أو null",
  "multiple_charges_treatment": "كيفية التعامل مع تعدد الجرائم وفق المادة 32 أو null",
  "execution_suspension_eligible": false,
  "execution_suspension_reasoning": "مبرر وقف التنفيذ أو null",
  "charge_conviction_map": {{
    "وصف التهمة 1": "إدانة",
    "وصف التهمة 2": "براءة"
  }},
  "civil_claim": {{
    "plaintiff_name": "اسم المدّعي المدني",
    "plaintiff_capacity": "صفة المدّعي أو null",
    "compensation_requested": 50000.0,
    "compensation_basis": "أضرار مادية ومعنوية",
    "material_damages": 30000.0,
    "moral_damages": 20000.0,
    "supporting_documents": ["وثيقة 1", "وثيقة 2"],
    "status": "مقامة | لم تُقَم | محجوزة للفصل | مقضي بها | مرفوضة",
    "suggested_award": 40000.0,
    "award_reasoning": "أسباب التقدير المقترح",
    "award_against_defendant": "اسم المتهم الملزَم أو null",
    "solidarity_liability": false
  }},
  "sentencing_recommendation": "التوصية التقديرية الإجمالية للقاضي في فقرة واحدة",
  "suggested_penalty_range": "نطاق العقوبة المقترح لكل تهمة ثبتت إدانتها"
}}
```

## قواعد صارمة
- أجب بـ JSON فقط بلا أي نص خارجه.
- استخدم اللغة العربية في جميع القيم النصية.
- civil_claim يكون null إذا لم تُقَم دعوى مدنية أصلاً.
- applicable_article_17 يكون true فقط إذا توفرت أسباب قانونية واضحة.
- charge_conviction_map يشمل جميع التهم الواردة في ملف القضية.
- لا تخترع وقائع — استند فقط إلى ما في السياق.
""".strip()


# ─────────────────────────────────────────────────────────────────────────────
# Agent
# ─────────────────────────────────────────────────────────────────────────────

class SentencingAgent(AgentBase):
    """
    يُنتج توصية تقديرية شاملة ويُحدّث AgentState بـ:
        - state.aggravating_factors
        - state.mitigating_factors
        - state.applicable_article_17
        - state.charge_conviction_map
        - state.civil_claim
        - state.civil_award_suggested
    """

    AGENT_KEY = "sentencing"

    def __init__(self) -> None:
        super().__init__(
            model_config_key="SENTENCING_MODEL",
            temp_config_key="SENTENCING_TEMP",
            prompt=SENTENCING_PROMPT,
        )

    # ── public entry point ────────────────────────────────────────

    def run(self, state: AgentState) -> AgentState:
        logger.info(
            "▶ [%s] بدء التقدير — القضية: %s | تهم: %d",
            self.AGENT_KEY, state.case_id, len(state.charges),
        )

        context = self._build_context(state)
        raw_output = self._call_llm(context)
        parsed = self._parse_llm_json(raw_output)

        updates = self._extract_updates(parsed, state)

        logger.info(
            "✔ [%s] اكتمل — مشددات: %d | مخففات: %d | م.17: %s",
            self.AGENT_KEY,
            len(updates.get("aggravating_factors", [])),
            len(updates.get("mitigating_factors", [])),
            updates.get("applicable_article_17", False),
        )

        base_state = self._empty_update(state, self.AGENT_KEY, updates)
        return base_state.model_copy(update=updates)

    # ── private helpers ───────────────────────────────────────────

    def _build_context(self, state: AgentState) -> str:
        ctx: dict[str, Any] = {
            "case_id":     state.case_id,
            "case_number": state.case_number,
            "court":       state.court,
            "court_level": state.court_level,
            "defendants":  [self._safe_dump(d) for d in state.defendants],
            "charges":     [self._safe_dump(c) for c in state.charges],
            "incidents":   [self._safe_dump(i) for i in state.incidents],
            "criminal_records": [self._safe_dump(r) for r in state.criminal_records],
            # مخرجات العملاء السابقين
            "evidence_scores":         [self._safe_dump(s) for s in state.evidence_scores],
            "defense_arguments":        [self._safe_dump(a) for a in state.defense_arguments],
            "prosecution_arguments":    [self._safe_dump(a) for a in state.prosecution_arguments],
            "confession_assessments":   [self._safe_dump(a) for a in state.confession_assessments],
            "witness_credibility_scores": [self._safe_dump(w) for w in state.witness_credibility_scores],
            "procedural_issues":         [self._safe_dump(p) for p in state.procedural_issues],
            "applied_principles":        [self._safe_dump(p) for p in state.applied_principles],
            "case_articles":             state.case_articles,
        }
        return self.prompt.format(context=json.dumps(ctx, ensure_ascii=False, indent=2))

    def _call_llm(self, prompt_text: str) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [
            SystemMessage(content=(
                "أنت مستشار قانوني متخصص في تقدير العقوبات المصرية. "
                "تجيب بـ JSON فقط بلا أي نص خارجه."
            )),
            HumanMessage(content=prompt_text),
        ]
        result = self._llm_invoke_with_retries(self._llm, messages)
        return result.content if hasattr(result, "content") else str(result)

    def _extract_updates(self, parsed: dict, state: AgentState) -> dict[str, Any]:
        updates: dict[str, Any] = {}

        updates["aggravating_factors"] = parsed.get("aggravating_factors") or []
        updates["mitigating_factors"]  = parsed.get("mitigating_factors")  or []
        updates["applicable_article_17"] = bool(parsed.get("applicable_article_17", False))
        updates["charge_conviction_map"] = parsed.get("charge_conviction_map") or {}

        # ── civil_claim ───────────────────────────────────────────
        raw_civil = parsed.get("civil_claim")
        if raw_civil and isinstance(raw_civil, dict):
            try:
                status_map: dict[str, CivilClaimStatus] = {
                    "مقامة":            CivilClaimStatus.FILED,
                    "لم تُقَم":         CivilClaimStatus.NOT_FILED,
                    "محجوزة للفصل":    CivilClaimStatus.RESERVED,
                    "مقضي بها":        CivilClaimStatus.AWARDED,
                    "مرفوضة":           CivilClaimStatus.REJECTED,
                }
                raw_civil["status"] = status_map.get(
                    raw_civil.get("status", "لم تُقَم"),
                    CivilClaimStatus.NOT_FILED,
                )
                civil = CivilClaim(**{k: v for k, v in raw_civil.items() if v is not None})
                updates["civil_claim"] = civil
                if civil.suggested_award is not None:
                    updates["civil_award_suggested"] = str(civil.suggested_award)
            except Exception as exc:
                logger.warning("[%s] فشل بناء CivilClaim: %s", self.AGENT_KEY, exc)

        return updates