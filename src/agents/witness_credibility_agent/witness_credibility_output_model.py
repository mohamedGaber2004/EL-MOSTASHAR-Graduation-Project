from typing import Optional, List
from pydantic import BaseModel, Field

from src.Utils.Enums.agents_enums import (
    WitnessReliabilityLevel
)


 
class WitnessCredibility(BaseModel):

    witness_name:              str           = Field(description="اسم الشاهد")
    statement_date:            Optional[str] = Field(default=None,description="تاريخ أداء الشهادة")
    relationship_to_defendant: Optional[str] = Field(default=None,description="صلة الشاهد بالمتهم أو المجني عليه إن وُجدت")
 
    # ── اتساق الرواية ─────────────────────────────────────────────
    internal_consistency:             bool      = Field(default=True,description="هل الشهادة متسقة داخلياً بلا تناقضات ذاتية؟")
    consistent_with_prior_statements: bool      = Field(default=True,description="هل تتفق مع تصريحاته السابقة (محاضر الضبط / أقوال الاستئناف)؟")
    contradictions_found:             List[str] = Field(default_factory=list,description="التناقضات المحددة مع الروايات الأخرى أو الأدلة المادية")
 
    # ── التوافق مع الأدلة المادية ─────────────────────────────────
    corroborated_by_physical_evidence: bool          = Field(default=False,description="هل تتوافق الشهادة مع الأدلة المادية المضبوطة؟")
    corroboration_details:             Optional[str] = Field(default=None,description="تفاصيل التوافق أو التعارض مع الأدلة المادية")
 
    # ── احتمالية التحيز ───────────────────────────────────────────
    potential_bias: Optional[str] = Field(default=None,description="أي مؤشر على تحيز محتمل (عداوة / مصلحة / ضغط)")
    demeanor_notes: Optional[str] = Field(default=None,description="ملاحظات على سلوك الشاهد أثناء الأداء إن وُجدت في الأوراق")
 
    # ── الحكم النهائي ─────────────────────────────────────────────
    reliability_level:     WitnessReliabilityLevel = Field(description="درجة الموثوقية الإجمالية")
    reliability_reasoning: str                     = Field(description="التسبيب الموجز لدرجة الموثوقية المُقدَّرة")
    weight_in_case:        str                     = Field(
        description=(
            "الوزن المقترح في سلسلة الإثبات: "
            "'ركيزة أساسية' / 'مساند' / 'استئناسي' / 'يُهمَل'"
        )
    )