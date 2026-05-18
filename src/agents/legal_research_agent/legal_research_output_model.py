from typing import Optional, List
from pydantic import BaseModel, Field


class JudicialPrinciple(BaseModel):
    # ── مضمون المبدأ ──────────────────────────────────────────────
    principle_text: Optional[str] = Field(default=None, description="نص المبدأ القضائي كما وَرد في حكم المحكمة")
    court_source:   Optional[str] = Field(default=None, description="مصدر المبدأ: اسم المحكمة والدائرة وتاريخ الطعن")

    # ── نطاق التطبيق ──────────────────────────────────────────────
    applicable_to: Optional[str] = Field(default=None, description="الوقائع أو المسائل القانونية التي ينطبق عليها هذا المبدأ في القضية الراهنة")


class CaseArticle(BaseModel):
    article_number: Optional[str] = Field(default=None, description="رقم المادة القانونية")
    law_id:         Optional[str] = Field(default=None, description="معرّف القانون (penal_code / criminal_procedure ...)")
    article_id:     Optional[str] = Field(default=None, description="معرّف المادة في قاعدة المعرفة")
    text:           Optional[str] = Field(default=None, description="نص المادة القانونية كاملاً")
    charge_ref:     Optional[str] = Field(default=None, description="التهمة التي تنتمي إليها هذه المادة")