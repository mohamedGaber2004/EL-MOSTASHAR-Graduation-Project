from typing import Optional, List
from pydantic import BaseModel, Field

# إنشاء هيكل فرعي مخصص لكل تهمة لإجبار النموذج على إدخال أرقام
class PerChargeRuling(BaseModel):
    charge_description: str = Field(default="غير محدد", description="وصف التهمة")
    verdict:            str = Field(default="غير محدد", description="منطوق الحكم لهذه التهمة (إدانة/براءة)")
    prison_years:       int = Field(default=0, description="عدد سنوات السجن أو الحبس كـ (رقم صحيح). ضع 0 إذا كانت العقوبة شهوراً فقط أو براءة.")
    prison_months:      int = Field(default=0, description="عدد أشهر الحبس (رقم صحيح). ضع 0 إذا لم يوجد.")
    fine_amount:        float = Field(default=0.0, description="قيمة الغرامة المالية إن وجدت.")
    reasoning:          str = Field(default="غير متوفر", description="التسبيب القانوني للحكم في هذه التهمة")

class FinalJudgment(BaseModel):
    # ── منطوق الحكم الإجمالي ──────────────────────────────────────
    verdict:             Optional[str] = Field(default=None, description="منطوق الحكم الإجمالي: إدانة كاملة / إدانة جزئية / براءة")
    
    # استبدال recommended_penalty بحقول رقمية محددة
    total_prison_years:  int = Field(default=0, description="إجمالي عدد سنوات العقوبة الموقعة على المتهم")
    total_prison_months: int = Field(default=0, description="إجمالي عدد أشهر العقوبة الموقعة")
    total_fine_amount:   float = Field(default=0.0, description="إجمالي الغرامة المالية الموقعة")

    # ── التفصيل ───────────────────────────────────────────────────
    # استخدام الهيكل الفرعي بدلاً من dict العادي
    per_charge_rulings: List[PerChargeRuling] = Field(
        default_factory=list,
        description="الحكم مفصَّلاً بحسب كل تهمة متضمناً العقوبة بالأرقام المحددة.",
    )
    
    operative_text: Optional[str] = Field(
        default=None,
        description="نص الحكم كاملاً بصياغته الرسمية متضمناً العقوبة بالأرقام (مثال: حكمت المحكمة حضورياً بحبس المتهم 3 سنوات...)",
    )

    # ── مؤشر الثقة ────────────────────────────────────────────────
    confidence_score: float = Field(default=0.0, ge=0, le=1, description="درجة ثقة النموذج في الحكم على مقياس 0.0–1.0")