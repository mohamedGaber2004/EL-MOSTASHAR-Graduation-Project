from typing import Optional, List
from pydantic import BaseModel, Field


class EvidenceScoring(BaseModel):
    # ── بيانات الدليل ─────────────────────────────────────────────
    evidence_id:  Optional[str] = Field(default=None, description="المعرّف الفريد للدليل كما وَرد في قائمة الأدلة")
    type:         Optional[str] = Field(default=None, description="نوع الدليل: مادي / قولي / فني / وثائقي")
    description:  Optional[str] = Field(default=None, description="وصف موجز للدليل ومصدره")

    # ── التقييم ───────────────────────────────────────────────────
    strength:        Optional[str] = Field(default=None, description="قوة الدليل: قوي / متوسط / ضعيف")
    strength_reason: Optional[str] = Field(default=None, description="مسوّغ درجة القوة المُقدَّرة")
    admissibility:   Optional[str] = Field(default=None, description="حكم القبول: مقبول / مستبعد — مستنَد إلى نتيجة فحص البطلان")
    proof_degree:    Optional[str] = Field(default=None, description="درجة الإثبات: قاطع / كافٍ / غير كافٍ / منعدم")

    # ── ملاحظات ───────────────────────────────────────────────────
    notes: Optional[str] = Field(default=None, description="ملاحظات إضافية تؤثر على تقييم الدليل")