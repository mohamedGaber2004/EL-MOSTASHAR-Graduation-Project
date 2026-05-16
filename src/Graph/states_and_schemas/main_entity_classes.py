from typing import List, Optional
from pydantic import BaseModel, Field
from pydantic import BaseModel, field_validator

# ══════════════════════════════════════════════════════
#  Core Entities
# ══════════════════════════════════════════════════════

class Defendant(BaseModel):
    """متهم في القضية"""
    name:               Optional[str] = None
    alias:              Optional[str] = None
    national_id:        Optional[str] = None
    gender:             Optional[str] = None   # ذكر / أنثى
    date_of_birth:      Optional[str] = None
    age:                Optional[int] = None
    occupation:         Optional[str] = None
    nationality:        Optional[str] = None
    address:            Optional[str] = None
    complicity_role:    Optional[str] = None   # فاعل أصلي / شريك / محرّض / متدخل
    source_document_id: Optional[str] = None


class CriminalRecord(BaseModel):
    """صحيفة السوابق الجنائية لمتهم"""
    defendant_name:     Optional[str] = None
    record_summary:     Optional[str] = None   # ملخص مختصر لمحتوى الصحيفة
    prior_cases:        List[str]     = Field(default_factory=list)  # قضايا سابقة إن ذُكرت
    source_document_id: Optional[str] = None


class CriminalProceedings(BaseModel):
    procedure_type:     Optional[str] = None   # ضبط / قبض / استجواب / تفتيش
    description:        Optional[str] = None
    conducting_officer: Optional[str] = None
    warrant_present:    bool          = False
    source_document_id: Optional[str] = None


class Charge(BaseModel):
    """تهمة موجهة"""
    law_code:               Optional[str] = None   # قانون العقوبات / قانون المخدرات …
    article_number:         Optional[str] = None
    description:            Optional[str] = None
    incident_type:          Optional[str] = None
    charge_classification:  Optional[str] = None   # جناية / جنحة / مخالفة
    attempt_flag:           bool          = False
    charge_date:            Optional[str] = None
    charge_location:        Optional[str] = None
    linked_defendant_names: List[str]     = Field(default_factory=list)
    complicity_role:        Optional[str] = None
    source_document_id:     Optional[str] = None


class CaseIncident(BaseModel):
    """واقعة / حادثة"""
    incident_type:        Optional[str] = None
    incident_date:        Optional[str] = None
    incident_location:    Optional[str] = None
    incident_description: Optional[str] = None
    perpetrator_names:    List[str]     = Field(default_factory=list)
    victim_names:         List[str]     = Field(default_factory=list)
    source_document_id:   Optional[str] = None
    @field_validator("incident_date", mode="before")
    @classmethod
    def coerce_date(cls, v):
        if isinstance(v, dict):
            # ✅ convert range to readable string
            start = v.get("start", "")
            end   = v.get("end", "")
            if start and end:
                return f"{start} إلى {end}"
            return start or end or None
        return v


class Evidence(BaseModel):
    """دليل مادي / إجرائي (حرز)"""
    evidence_type:           Optional[str] = None   # سلاح / مخدر / مستند / هاتف …
    description:             Optional[str] = None
    detailed_text:           Optional[str] = None   # النص الكامل للوصف كما ورد في المحضر
    seizure_date:            Optional[str] = None
    seizure_location:        Optional[str] = None
    seized_by:               Optional[str] = None   # اسم وصفة الضابط
    seizure_warrant_present: bool          = False
    linked_defendant_name:   Optional[str] = None
    source_document_id:      Optional[str] = None


class EvidenceItem(BaseModel):
    description: str
    evidence_number: Optional[str] = None

class FingerprintAnalysis(BaseModel):
    fingerprint_location: str
    fingerprint_type: str
    match_percentage: float
    match_status: str

class FacialRecognitionAnalysis(BaseModel):
    source: str
    match_percentage: float
    observations: list[str]

class WalletAnalysis(BaseModel):
    description_match: str
    contents: list[str]

class LabResult(BaseModel):
    fingerprint_analysis: Optional[FingerprintAnalysis] = None
    facial_recognition_analysis: Optional[FacialRecognitionAnalysis] = None
    wallet_analysis: Optional[WalletAnalysis] = None

class LabReport(BaseModel):
    report_type: str
    report_number: str
    examination_date: str
    examiner_name: str
    prosecutor_name: str
    items_sent_for_analysis: list[EvidenceItem]
    result: LabResult

    @field_validator("items_sent_for_analysis", mode="before")
    @classmethod
    def coerce_items(cls, v):
        coerced = []
        for item in v:
            if isinstance(item, str):
                coerced.append({"description": item})
            elif isinstance(item, dict):
                coerced.append(item)
        return coerced
    
    linked_defendant_name: Optional[str] = None
    source_document_id: Optional[str] = None


class WitnessStatement(BaseModel):
    witness_name:           Optional[str]       = None
    occupation:             Optional[str]       = None
    id_number:              Optional[str]       = None
    relation_to_defendant:  Optional[str]       = None
    statement_date:         Optional[str]       = None
    was_sworn_in:           Optional[bool]      = None
    presence_at_scene:      Optional[bool]      = None
    key_facts_mentioned:    List[str]           = []   # ← was bare List[str] with no coercion
    age:                    Optional[int]       = None
    source_document_id:     Optional[str]       = None

    @field_validator("key_facts_mentioned", mode="before")
    @classmethod
    def coerce_to_list(cls, v):
        """LLM sometimes returns a string instead of a list — wrap it."""
        if v is None:
            return []
        if isinstance(v, str):
            return [v] if v.strip() else []
        return v


class Confession(BaseModel):
    """اعتراف / إنكار في محضر الاستجواب"""
    defendant_name:        Optional[str] = None
    text:                  Optional[str] = None   # ملخص موقف المتهم
    confession_date:       Optional[str] = None
    confession_stage:      Optional[str] = None   # تحقيق / محكمة
    legal_counsel_present: bool          = False
    coercion_claimed:      bool          = False
    voluntary:             bool          = True
    key_admissions:        List[str]     = Field(default_factory=list)
    source_document_id:    Optional[str] = None


class DefenseLawyer(BaseModel):
    name: str
    bar_association: Optional[str] = None
    registration_number: Optional[str] = None

class DefenseDocument(BaseModel):
    submitted_by:          Optional[str]       = None
    defendant_name:        Optional[str]       = None
    defense_team:          List[DefenseLawyer] = Field(default_factory=list)
    formal_defenses:       List[str]           = Field(default_factory=list)
    substantive_defenses:  List[str]           = Field(default_factory=list)
    supporting_principles: List[str]           = Field(default_factory=list)
    alibi_claimed:         bool                = False
    alibi_description:     Optional[str]       = None
    source_document_id:    Optional[str]       = None

    @field_validator("defense_team", mode="before")
    @classmethod
    def coerce_defense_team(cls, v):
        if not isinstance(v, list):
            return v
        return [
            {"name": item} if isinstance(item, str) else item
            for item in v
        ]