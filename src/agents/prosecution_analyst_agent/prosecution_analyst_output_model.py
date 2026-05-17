from typing import Optional, List
from pydantic import BaseModel, Field

from src.agents.agents_enums import (
    ProsecutionArgumentStrength
)


class ProsecutionNarrative(BaseModel):
    # ── وقائع الاتهام ─────────────────────────────────────────────
    summary:                 str           = Field(description="ملخص موجز لوقائع الاتهام كما يراها المدّعي العام (3-5 جمل)")
    incident_reconstruction: str           = Field(description="إعادة تسلسل وقائع الحادثة زمنياً من منظور النيابة")
    motive:                  Optional[str] = Field(default=None,description="الدافع المزعوم للجريمة إن ظهر من الأوراق")
 
    # ── ربط الأدلة بالاتهام ───────────────────────────────────────
    evidence_chain:     List[str]     = Field(default_factory=list,description="سلسلة الأدلة المادية الداعمة للاتهام مرتبةً حسب الأهمية")
    confession_role:    Optional[str] = Field(default=None,description="دور الاعتراف — إن وُجد — في بناء الاتهام")
    witness_summary:    Optional[str] = Field(default=None,description="ملخص إسهام شهادات الشهود في تأييد رواية النيابة")
    lab_report_summary: Optional[str] = Field(default=None,description="ملخص دور التقارير الفنية والمعملية في إثبات الاتهام")
 
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
    weaknesses:                    List[str]                   = Field(default_factory=list,description="نقاط الضعف في ملف الاتهام التي ينبغي أن يراها القاضي")
    recommended_verdict_direction: str = Field(
        description=(
            "توجيه الحكم المقترح من منظور الاتهام: "
            "'إدانة كاملة' / 'إدانة جزئية' / 'تخفيف' مع مبرر"
        )
    )

class ProsecutionArgument(BaseModel):
    charge_description:      str                         = Field(description="وصف التهمة التي تخصّها هذه الحجة")
    article_reference:       Optional[str]               = Field(default=None,description="المادة القانونية المستند إليها (مثال: المادة 234 عقوبات)")
    argument_text:           str                         = Field(description="نص الحجة الاتهامية كما سيُعرض على القاضي")
    supporting_evidence_ids: List[str]                   = Field(default_factory=list,description="معرّفات الأدلة الداعمة لهذه الحجة من قائمة evidences")
    strength:                ProsecutionArgumentStrength = Field(description="قوة هذه الحجة بالذات")
    rebuttal_risk:           Optional[str]               = Field(default=None,description="أبرز ما يمكن أن يرد به الدفاع على هذه الحجة")