from typing import Optional, List
from pydantic import BaseModel, Field


class JudicialPrinciple(BaseModel):
    # ── مضمون المبدأ ──────────────────────────────────────────────
    principle_text: Optional[str] = Field(default=None, description="نص المبدأ القضائي كما وَرد في حكم المحكمة")
    court_source:   Optional[str] = Field(default=None, description="مصدر المبدأ: اسم المحكمة والدائرة وتاريخ الطعن")

    # ── نطاق التطبيق ──────────────────────────────────────────────
    applicable_to: Optional[str] = Field(default=None, description="الوقائع أو المسائل القانونية التي ينطبق عليها هذا المبدأ في القضية الراهنة")
