from __future__ import annotations
from typing import List, Optional

from pydantic import BaseModel, Field

from src.Utils.legal_enumerations import Gender ,NullityType, CourtLevel

# ══════════════════════════════════════════════════════
#  Core Entities
# ══════════════════════════════════════════════════════

class Defendant(BaseModel):
    """متهم في القضية"""
    name:                  Optional[str]      = None
    alias:                 Optional[str]      = None
    national_id:           Optional[str]      = None
    passport_number:       Optional[str]      = None
    gender:                Optional[Gender]   = None
    date_of_birth:         Optional[str]      = None
    age:                   Optional[int]      = None
    occupation:            Optional[str]      = None
    nationality:           Optional[str]      = None
    address:               Optional[str]      = None
    complicity_role:       Optional[str]      = None   # فاعل / شريك / محرّض
    source_document_id:    Optional[str]      = None


class Charge(BaseModel):
    """تهمة موجهة"""
    law_code:                  Optional[str]       = None   # قانون العقوبات / قانون المخدرات …
    article_number:            Optional[str]       = None
    description:               Optional[str]       = None
    incident_type:             Optional[str]       = None
    attempt_flag:              bool                = False
    charge_date:               Optional[str]       = None
    charge_location:           Optional[str]       = None
    linked_defendant_names:    List[str]           = Field(default_factory=list)
    complicity_role:           Optional[str]       = None
    source_document_id:        Optional[str]       = None


class CaseIncident(BaseModel):
    """واقعة / حادثة"""
    incident_type:         Optional[str]  = None
    incident_date:         Optional[str]  = None
    incident_location:     Optional[str]  = None
    incident_description:  Optional[str]  = None
    perpetrator_names:     List[str]      = Field(default_factory=list)
    victim_names:          List[str]      = Field(default_factory=list)
    source_document_id:    Optional[str]  = None


class Evidence(BaseModel):
    """دليل مادي / إجرائي"""
    evidence_type:             Optional[str]  = None   # سلاح / مخدر / مستند …
    description:               Optional[str]  = None
    seizure_date:              Optional[str]  = None
    seizure_location:          Optional[str]  = None
    seized_by:                 Optional[str]  = None
    seizure_warrant_present:   bool           = False
    linked_defendant_name:     Optional[str]  = None
    source_document_id:        Optional[str]  = None


class LabReport(BaseModel):
    """تقرير معملي / طبي شرعي / فني"""
    report_type:           Optional[str]  = None   # طبي شرعي / معملي / كيميائي
    examination_date:      Optional[str]  = None
    examiner_name:         Optional[str]  = None
    result:                Optional[str]  = None
    linked_defendant_name: Optional[str]  = None
    source_document_id:    Optional[str]  = None


class WitnessStatement(BaseModel):
    """شهادة شاهد"""
    witness_name:           Optional[str]  = None
    witness_type:           Optional[str]  = None   # عيان / خبير / ضابط …
    relation_to_defendant:  Optional[str]  = None
    statement_summary:      Optional[str]  = None
    statement_date:         Optional[str]  = None
    was_sworn_in:           bool           = False
    presence_at_scene:      bool           = False
    key_facts_mentioned:    List[str]      = Field(default_factory=list)
    source_document_id:     Optional[str]  = None


class Confession(BaseModel):
    """اعتراف / إنكار"""
    defendant_name:        Optional[str]  = None
    text:                  Optional[str]  = None
    confession_date:       Optional[str]  = None
    confession_stage:      Optional[str]  = None   # تحقيق / محكمة
    legal_counsel_present: bool           = False
    coercion_claimed:      bool           = False
    voluntary:             bool           = True
    key_admissions:        List[str]      = Field(default_factory=list)
    source_document_id:    Optional[str]  = None


class ProceduralIssue(BaseModel):
    """دفع إجرائي / بطلان"""
    procedure_type:     Optional[str]        = None   # ضبط / قبض / استجواب
    issue_description:  Optional[str]        = None
    warrant_present:    bool                 = False
    conducting_officer: Optional[str]        = None
    nullity_type:       Optional[NullityType] = None
    article_basis:      Optional[str]        = None
    source_document_id: Optional[str]        = None


class PriorJudgment(BaseModel):
    """حكم أو سابقة قضائية"""
    judgment_number:    Optional[str]  = None
    court_name:         Optional[str]  = None
    judgment_date:      Optional[str]  = None
    summary:            Optional[str]  = None
    source_document_id: Optional[str]  = None


class DefenseDocument(BaseModel):
    """مذكرة دفاع"""
    submitted_by:           Optional[str]  = None
    formal_defenses:        List[str]      = Field(default_factory=list)
    substantive_defenses:   List[str]      = Field(default_factory=list)
    supporting_principles:  List[str]      = Field(default_factory=list)
    alibi_claimed:          bool           = False
    source_document_id:     Optional[str]  = None


# ══════════════════════════════════════════════════════
#  Analysis Output Models
# ══════════════════════════════════════════════════════

class EvidenceAnalysis(BaseModel):
    evidence_id:        Optional[str]   = None
    admissibility:      Optional[str]   = None
    weight:             Optional[str]   = None
    notes:              Optional[str]   = None


class DefenseArgument(BaseModel):
    argument_type:      Optional[str]   = None   # شكلي / موضوعي
    description:        Optional[str]   = None
    legal_basis:        Optional[str]   = None
    strength:           Optional[str]   = None   # قوي / متوسط / ضعيف


class JudicialPrinciple(BaseModel):
    principle_text:     Optional[str]   = None
    court_source:       Optional[str]   = None
    applicable_to:      Optional[str]   = None


class FinalJudgment(BaseModel):
    verdict:            Optional[str]   = None   # إدانة / براءة / جزئية
    reasoning_summary:  Optional[str]   = None
    recommended_penalty:Optional[str]   = None
    confidence_score:   float           = Field(default=0.0, ge=0, le=1)


# ══════════════════════════════════════════════════════
#  Extraction Schemas  (used by instructor per doc-type)
# ══════════════════════════════════════════════════════

class CaseMeta(BaseModel):
    case_number:         Optional[str]        = None
    court:               Optional[str]        = None
    court_level:         Optional[str] = None
    jurisdiction:        Optional[str]        = None
    filing_date:         Optional[str]        = None
    referral_date:       Optional[str]        = None
    prosecutor_name:     Optional[str]        = None
    referral_order_text: Optional[str]        = None


class AmrIhalaExtraction(BaseModel):
    """أمر الإحالة"""
    case_meta:   CaseMeta        = Field(default_factory=CaseMeta)
    defendants:  List[Defendant] = Field(default_factory=list)
    charges:     List[Charge]    = Field(default_factory=list)


class MahdarDabtExtraction(BaseModel):
    """محضر الضبط والتفتيش"""
    evidences:          List[Evidence]         = Field(default_factory=list)
    procedural_issues:  List[ProceduralIssue]  = Field(default_factory=list)
    incidents:          List[CaseIncident]     = Field(default_factory=list)


class MahdarIstijwabExtraction(BaseModel):
    """محضر الاستجواب"""
    confessions: List[Confession] = Field(default_factory=list)


class AqwalShuhudExtraction(BaseModel):
    """أقوال الشهود"""
    witness_statements: List[WitnessStatement] = Field(default_factory=list)


class TaqrirTibbiExtraction(BaseModel):
    """التقرير الطبي / المعملي"""
    lab_reports: List[LabReport] = Field(default_factory=list)


class MozakaretDifaExtraction(BaseModel):
    """مذكرة الدفاع"""
    defense_documents:  List[DefenseDocument]  = Field(default_factory=list)
    procedural_issues:  List[ProceduralIssue]  = Field(default_factory=list)