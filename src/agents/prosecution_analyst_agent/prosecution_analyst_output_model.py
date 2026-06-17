from typing import Optional, List, Any
from pydantic import BaseModel, Field, field_validator

from src.agents.agent_base.agents_enums import (
    ProsecutionArgumentStrength
)

# ── shared coercions (paste these or import from your shared helpers) ──────────

def _require_str(field_name: str):
    """
    Returns a field_validator that coerces None / non-string values on a
    required str field into a non-empty fallback string, so Pydantic never
    raises 'Field required' when the LLM emits null.
    The fallback is logged-friendly Arabic so downstream readers understand
    the model failed to populate the field.
    """
    FALLBACKS = {
        "charge_description": "وصف التهمة غير متوفر",
        "argument_text":      "نص الحجة غير متوفر",
    }
    fallback = FALLBACKS.get(field_name, f"{field_name} غير متوفر")

    @field_validator(field_name, mode="before")
    @classmethod
    def _coerce(cls, v: Any) -> str:
        if v is None:
            return fallback
        if isinstance(v, str):
            return v.strip() or fallback
        if isinstance(v, (list, dict)):
            import json
            return json.dumps(v, ensure_ascii=False)
        return str(v)

    return _coerce


class ProsecutionNarrative(BaseModel):
    # ── وقائع الاتهام ─────────────────────────────────────────────
    summary:                 str           = Field(default="غير متوفر", description="ملخص موجز لوقائع الاتهام كما يراها المدّعي العام (3-5 جمل)")
    incident_reconstruction: str           = Field(default="لم يقم النموذج بإعادة بناء التسلسل", description="إعادة تسلسل وقائع الحادثة زمنياً من منظور النيابة")
    motive:                  Optional[str] = Field(default=None, description="الدافع المزعوم للجريمة إن ظهر من الأوراق")
 
    # ── ربط الأدلة بالاتهام ───────────────────────────────────────
    evidence_chain:     List[str]     = Field(default_factory=list, description="سلسلة الأدلة المادية الداعمة للاتهام مرتبةً حسب الأهمية")
    confession_role:    Optional[str] = Field(default=None, description="دور الاعتراف — إن وُجد — في بناء الاتهام")
    witness_summary:    Optional[str] = Field(default=None, description="ملخص إسهام شهادات الشهود في تأييد رواية النيابة")
    lab_report_summary: Optional[str] = Field(default=None, description="ملخص دور التقارير الفنية والمعملية في إثبات الاتهام")
 
    # ── ربط التهم بعناصرها القانونية ──────────────────────────────
    charge_element_mapping: List[dict] = Field(
        default_factory=list,
        description=(
            "قائمة بكل تهمة وعناصرها القانونية المتوفرة في الأوراق. "
            "كل عنصر: {charge_description, article, elements_proven: [str], elements_missing: [str]}"
        )
    )
 
    # ── التقييم الكلي ─────────────────────────────────────────────
    overall_strength:              ProsecutionArgumentStrength = Field(description="التقدير الإجمالي لقوة ملف الاتهام")
    weaknesses:                    List[str]                   = Field(default_factory=list, description="نقاط الضعف في ملف الاتهام التي ينبغي أن يراها القاضي")
    recommended_verdict_direction: str = Field(
        default="توجيه غير محدد",
        description=(
            "توجيه الحكم المقترح من منظور الاتهام: "
            "'إدانة كاملة' / 'إدانة جزئية' / 'تخفيف' مع مبرر"
        )
    )

class ProsecutionArgument(BaseModel):
    charge_description:      str                         = Field(description="وصف التهمة التي تخصّها هذه الحجة")
    article_reference:       Optional[str]               = Field(default=None, description="المادة القانونية المستند إليها (مثال: المادة 234 عقوبات)")
    argument_text:           str                         = Field(description="نص الحجة الاتهامية كما سيُعرض على القاضي")
    supporting_evidence_ids: List[str]                   = Field(default_factory=list, description="معرّفات الأدلة الداعمة لهذه الحجة من قائمة evidences")
    strength:                ProsecutionArgumentStrength = Field(description="قوة هذه الحجة بالذات")
    rebuttal_risk:           Optional[str]               = Field(default=None, description="أبرز ما يمكن أن يرد به الدفاع على هذه الحجة")

    # ── coerce required str fields — never let null through ───────────────────
    coerce_charge_description = _require_str("charge_description")
    coerce_argument_text      = _require_str("argument_text")

    @field_validator("supporting_evidence_ids", mode="before")
    @classmethod
    def coerce_evidence_ids(cls, v: Any) -> List[str]:
        """LLMs sometimes return a list of dicts or a bare string here."""
        if v is None:
            return []
        if isinstance(v, str):
            return [v] if v.strip() else []
        if isinstance(v, list):
            result = []
            for item in v:
                if isinstance(item, str) and item.strip():
                    result.append(item)
                elif isinstance(item, dict):
                    for val in item.values():
                        if isinstance(val, str) and val.strip():
                            result.append(val)
                            break
            return result
        return []