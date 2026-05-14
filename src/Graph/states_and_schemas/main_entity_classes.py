from typing import List, Optional
from pydantic import BaseModel, Field


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


class LabReport(BaseModel):
    """تقرير معملي / طبي شرعي / فني"""
    report_type:            Optional[str] = None   # طبي شرعي / كيميائي / بلستي / رقمي / مروري
    report_number:          Optional[str] = None   # رقم تقرير المعمل
    examination_date:       Optional[str] = None
    examiner_name:          Optional[str] = None
    prosecutor_name:        Optional[str] = None   # وكيل النيابة مُرسِل التقرير
    items_sent_for_analysis: List[str]    = Field(default_factory=list)  # الاستيناء المرسلة
    result:                 Optional[str] = None
    linked_defendant_name:  Optional[str] = None
    source_document_id:     Optional[str] = None


class WitnessStatement(BaseModel):
    """شهادة شاهد"""
    witness_name:          Optional[str] = None
    witness_type:          Optional[str] = None   # عيان / خبير / ضابط / مجني عليه / شاهد نفي
    age:                   Optional[int] = None
    occupation:            Optional[str] = None
    id_number:             Optional[str] = None   # رقم الكارنيه / البطاقة
    relation_to_defendant: Optional[str] = None
    statement_summary:     Optional[str] = None
    statement_date:        Optional[str] = None
    was_sworn_in:          bool          = False
    presence_at_scene:     bool          = False
    key_facts_mentioned:   List[str]     = Field(default_factory=list)
    source_document_id:    Optional[str] = None


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


class DefenseDocument(BaseModel):
    """مذكرة دفاع"""
    submitted_by:          Optional[str] = None   # اسم المحامي
    defendant_name:        Optional[str] = None   # المتهم المدافَع عنه
    defense_team:          Optional[str] = None   # هيئة الدفاع إن تعددوا
    formal_defenses:       List[str]     = Field(default_factory=list)   # دفوع شكلية / إجرائية
    substantive_defenses:  List[str]     = Field(default_factory=list)   # دفوع موضوعية
    supporting_principles: List[str]     = Field(default_factory=list)   # مبادئ قانونية / نقض
    alibi_claimed:         bool          = False
    alibi_description:     Optional[str] = None   # وصف الألبي إن وُجد
    source_document_id:    Optional[str] = None