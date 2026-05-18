from typing import List, Optional
from pydantic import BaseModel, Field


class FormalDefense(BaseModel):
    defense:            Optional[str] = Field(default=None, description="نص الدفع الشكلي")
    legal_basis:        Optional[str] = Field(default=None, description="المادة القانونية المستند إليها")
    strength:           Optional[str] = Field(default=None, description="قوي | متوسط | ضعيف")
    strength_reasoning: Optional[str] = Field(default=None, description="سبب التقييم")
    linked_nullity:     Optional[str] = Field(default=None, description="معرف البطلان الإجرائي المرتبط أو null")
    notes:              Optional[str] = Field(default=None, description="ملاحظات إضافية")


class SubstantiveDefense(BaseModel):
    defense:                 Optional[str]       = Field(default=None, description="نص الدفع الموضوعي")
    legal_basis:             Optional[str]       = Field(default=None, description="المادة القانونية")
    strength:                Optional[str]       = Field(default=None, description="قوي | متوسط | ضعيف")
    strength_reasoning:      Optional[str]       = Field(default=None, description="سبب التقييم")
    countered_by_evidence:   Optional[List[str]] = Field(default_factory=list, description="evidence_ids تدحض هذا الدفع")
    notes:                   Optional[str]       = Field(default=None, description="ملاحظات")


class AlibiAnalysis(BaseModel):
    claimed:                 Optional[bool]      = Field(default=None)
    supported_by_evidence:   Optional[bool]      = Field(default=None)
    supporting_evidence_ids: Optional[List[str]] = Field(default_factory=list)
    notes:                   Optional[str]       = Field(default=None, description="تقييم الـ alibi")

class DefenseAnalysis(BaseModel):
    """النموذج الكامل لمخرجات وكيل الدفاع — يُستخدم داخليًا قبل التحويل إلى List[DefenseArgument]."""
    formal_defenses:          List[FormalDefense]     = Field(default_factory=list)
    substantive_defenses:     List[SubstantiveDefense] = Field(default_factory=list)
    alibi_analysis:           Optional[AlibiAnalysis] = Field(default=None)
    supporting_principles:    List[str]               = Field(default_factory=list)
    overall_defense_strength: Optional[str]           = Field(default=None)
    strength_reasoning:       Optional[str]           = Field(default=None)
    critical_defense_points:  List[str]               = Field(default_factory=list)