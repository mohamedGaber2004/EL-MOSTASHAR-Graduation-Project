from typing import Optional, List
from pydantic import BaseModel, Field



class FinalJudgment(BaseModel):
    # ── منطوق الحكم ───────────────────────────────────────────────
    verdict:             Optional[str] = Field(default=None, description="منطوق الحكم: إدانة كاملة / إدانة جزئية / براءة")
    recommended_penalty: Optional[str] = Field(default=None, description="العقوبة المقترحة محددةً بنوعها ومدتها وفق النصوص المنطبقة")

    # ── التسبيب والتفصيل ──────────────────────────────────────────
    reasoning_summary:  Optional[str] = Field(default=None, description="ملخص أسباب الحكم مستوعِباً الأدلة والدفوع وتقدير المحكمة")
    per_charge_rulings: List[dict]    = Field(
        default_factory=list,
        description=(
            "الحكم مفصَّلاً بحسب كل تهمة. "
            "كل عنصر: {charge_description, verdict, penalty, reasoning}"
        )
    )
    full_ruling_text: Optional[str] = Field(default=None, description="نص الحكم كاملاً بصياغته الرسمية كما يُنطق في الجلسة")

    # ── مؤشر الثقة ────────────────────────────────────────────────
    confidence_score: float = Field(default=0.0, ge=0, le=1, description="درجة ثقة النموذج في توصية الحكم على مقياس 0-1")
