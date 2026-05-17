from typing import Optional, List
from pydantic import BaseModel, Field

from src.agents.agents_enums import (
    CoercionType,
    ConfessionAdmissibilityStatus,
)


class ConfessionAssessment(BaseModel):

    defendant_name:          str           = Field(description="اسم المتهم صاحب الاعتراف")
    confession_date:         Optional[str] = Field(default=None,description="تاريخ الاعتراف")
    confession_text_summary: Optional[str] = Field(default=None,description="ملخص مضمون الاعتراف")
 
    # ── الإكراه ───────────────────────────────────────────────────
    coercion_alleged:  bool          = Field(default=False,description="هل ادّعى المتهم أو الدفاع وجود إكراه؟")
    coercion_type:     CoercionType  = Field(default=CoercionType.NONE,description="نوع الإكراه المزعوم")
    coercion_evidence: Optional[str] = Field(default=None,description="الدليل على الإكراه إن وُجد (شهادة، تقرير طبي، ...)")
 
    # ── الإجراءات ─────────────────────────────────────────────────
    taken_before_competent_authority: bool            = Field(default=True,description="هل أُخذ الاعتراف أمام جهة مختصة (نيابة / قاضٍ)؟")
    miranda_rights_observed:          bool            = Field(default=True,description="هل أُحيط المتهم علماً بحقه في الصمت والاستعانة بمحامٍ؟")
    time_since_arrest_hours:          Optional[float] = Field(default=None,description="الساعات المنقضية بين القبض والاعتراف")
    procedural_notes:                 Optional[str]   = Field(default=None,description="ملاحظات إجرائية أخرى تؤثر على صحة الاعتراف")
 
    # ── المادة 302 إجراءات جنائية ─────────────────────────────────
    standalone_confession:  bool      = Field(default=False,description="هل الاعتراف هو الدليل الوحيد بلا سند مادي مستقل؟")
    corroborating_evidence: List[str] = Field(default_factory=list,description="الأدلة المادية المعززة للاعتراف إن وُجدت")
 
    # ── الحكم النهائي ─────────────────────────────────────────────
    admissibility_status: ConfessionAdmissibilityStatus = Field(description="حكم قبول الاعتراف")
    legal_reasoning:      str                           = Field(description="التسبيب القانوني لحكم القبول / الرفض")
    impact_on_case:       str                           = Field(
        description=(
            "أثر هذا التقييم على القضية: "
            "'يُعزز الإدانة' / 'يُضعف الادعاء' / 'يستوجب الاستبعاد'"
        )
    )