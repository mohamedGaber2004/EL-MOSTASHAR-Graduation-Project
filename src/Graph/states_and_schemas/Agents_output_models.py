from typing import Optional
from pydantic import BaseModel, Field

# ══════════════════════════════════════════════════════
#  Analysis Output Models  (populated by downstream agents)
# ══════════════════════════════════════════════════════

class EvidenceScoring(BaseModel):
    evidence_id:   Optional[str] = None
    admissibility: Optional[str] = None
    weight:        Optional[str] = None
    notes:         Optional[str] = None


class DefenseArgument(BaseModel):
    argument_type: Optional[str] = None   # شكلي / موضوعي
    description:   Optional[str] = None
    legal_basis:   Optional[str] = None
    strength:      Optional[str] = None   # قوي / متوسط / ضعيف


class JudicialPrinciple(BaseModel):
    principle_text: Optional[str] = None
    court_source:   Optional[str] = None
    applicable_to:  Optional[str] = None


class FinalJudgment(BaseModel):
    verdict:             Optional[str] = None   # إدانة / براءة / جزئية
    reasoning_summary:   Optional[str] = None
    recommended_penalty: Optional[str] = None
    confidence_score:    float         = Field(default=0.0, ge=0, le=1)


class ProceduralIssue(BaseModel):
    """دفع إجرائي / بطلان"""
    procedure_type:     Optional[str] = None   # ضبط / قبض / استجواب / تفتيش
    issue_description:  Optional[str] = None
    warrant_present:    bool          = False
    conducting_officer: Optional[str] = None
    nullity_type:       Optional[str] = None   # بطلان مطلق / بطلان نسبي
    article_basis:      Optional[str] = None
    source_document_id: Optional[str] = None

