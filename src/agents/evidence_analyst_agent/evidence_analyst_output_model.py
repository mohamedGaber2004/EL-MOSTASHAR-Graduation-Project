from typing import Optional
from pydantic import BaseModel, Field



class EvidenceScoring(BaseModel):
    # ── بيانات الدليل ─────────────────────────────────────────────
    evidence_id:     Optional[str] = Field(default=None, description="المعرّف الفريد للدليل")
    type:            Optional[str] = Field(default=None, description="نوع الدليل: مادي / قولي / فني / وثائقي")
    description:     Optional[str] = Field(default=None, description="وصف موجز للدليل")

    # ── التقييم ───────────────────────────────────────────────────
    strength:        Optional[str] = Field(default=None, description="قوة الدليل: قوي / متوسط / ضعيف")
    strength_reason: Optional[str] = Field(default=None, description="مسوّغ درجة القوة")
    admissibility:   Optional[str] = Field(default=None, description="مقبول / مستبعد")
    proof_degree:    Optional[str] = Field(default=None, description="قاطع / كافٍ / غير كافٍ / منعدم")