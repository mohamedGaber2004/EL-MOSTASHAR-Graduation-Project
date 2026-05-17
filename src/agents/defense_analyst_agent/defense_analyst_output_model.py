from typing import Optional
from pydantic import BaseModel, Field




class DefenseArgument(BaseModel):
    # ── تصنيف الدفع ───────────────────────────────────────────────
    argument_type: Optional[str] = Field(default=None, description="نوع الدفع: شكلي / موضوعي")
    description:   Optional[str] = Field(default=None, description="نص الدفع كما أبداه المحامي أو كما يستخلصه التحليل")

    # ── الأساس القانوني والتقدير ──────────────────────────────────
    legal_basis: Optional[str] = Field(default=None, description="النص القانوني أو المبدأ القضائي الذي يستند إليه الدفع")
    strength:    Optional[str] = Field(default=None, description="قوة الدفع: قوي / متوسط / ضعيف")
