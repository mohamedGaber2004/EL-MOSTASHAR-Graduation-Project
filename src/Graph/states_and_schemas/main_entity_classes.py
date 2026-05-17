from typing import List, Optional
from pydantic import BaseModel, Field
from pydantic import BaseModel, field_validator

# ══════════════════════════════════════════════════════
#  Core Entities
# ══════════════════════════════════════════════════════

class Defendant(BaseModel):
    """متهم في القضية"""
    name:               Optional[str] = None   # الاسم
    alias:              Optional[str] = None   # اسم الشهره
    national_id:        Optional[str] = None   # الرقم القومي
    gender:             Optional[str] = None   # ذكر / أنثى
    date_of_birth:      Optional[str] = None   # الميلاد
    age:                Optional[int] = None   # السن
    occupation:         Optional[str] = None   # العمل
    nationality:        Optional[str] = "مصري" # الجنسيه
    address:            Optional[str] = None   # العنوان
    complicity_role:    Optional[str] = None   # فاعل أصلي / شريك / محرّض / متدخل
    source_document_id: Optional[str] = None


class Charge(BaseModel):
    """تهمة موجهة"""
    law_code:               Optional[str] = None # قانون العقوبات / قانون المخدرات …
    article_number:         Optional[str] = None # رقم الماده
    description:            Optional[str] = None # وصف التهمه
    incident_type:          Optional[str] = None # نوع الجريمة
    charge_classification:  Optional[str] = None # جناية / جنحة / مخالفة
    attempt_flag:           bool          = False # شروع ام لا
    charge_date:            Optional[str] = None # التاريخ
    charge_location:        Optional[str] = None # المكان
    linked_defendant_names: List[str]     = Field(default_factory=list) # اسم المتهم
    source_document_id:     Optional[str] = None


class CaseIncident(BaseModel):
    """واقعة / حادثة"""
    incident_type:        Optional[str] = None # نوع الجريمة
    incident_date:        Optional[str] = None # التاريخ
    incident_location:    Optional[str] = None # المكان
    incident_description: Optional[str] = None # وصف الجريمه
    perpetrator_names:    List[str]     = Field(default_factory=list) # اسماء الجناه
    victim_names:         List[str]     = Field(default_factory=list) # اسماء المجني عليهم
    source_document_id:   Optional[str] = None
    @field_validator("incident_date", mode="before")
    @classmethod
    def coerce_date(cls, v):
        if isinstance(v, dict):
            start = v.get("start", "")
            end   = v.get("end", "")
            if start and end:
                return f"{start} إلى {end}"
            return start or end or None
        return v


class Evidence(BaseModel):
    """دليل مادي / إجرائي (حرز)"""
    evidence_type:           Optional[str] = None   # سلاح / مخدر / مستند / هاتف …
    description:             Optional[str] = None   # وصف الحرز المختصر
    detailed_text:           Optional[str] = None   # النص الكامل للوصف كما ورد في المحضر
    seizure_date:            Optional[str] = None   # تاريخ ووقت الضبط
    seizure_location:        Optional[str] = None   # مكان الضبط
    seized_by:               Optional[str] = None   # اسم وصفة الضابط
    seizure_warrant_present: bool          = False  # وجود امر او اذن نيابه
    linked_defendant_name:   Optional[str] = None   # اسم المتهم الذي ضُبط الحرز بحوزته.
    source_document_id:      Optional[str] = None


class EvidenceItem(BaseModel):
    description: str                      # وصف العنصر المُرسَل للتحليل
    evidence_number: Optional[str] = None # رقم الحرز المُسجَّل في محضر الضبط

class LabReport(BaseModel):
    report_type:     str # نوع التقرير (طبي شرعي / كيميائي / بلستي / غيره)
    report_number:   str # رقم التقرير الرسمي الصادر من المعمل
    examination_date:str # تاريخ إجراء الفحص بالمعمل
    examiner_name:   str # اسم الخبير أو الطبيب الشرعي المُجري للفحص
    prosecutor_name: str # اسم عضو النيابة الآمر بالفحص 
    items_sent_for_analysis: list[EvidenceItem] = Field(default_factory=list) # قائمة العناصر المُرسَلة للمعمل للتحليل (أحراز، مستندات، تسجيلات)
    result: Optional[str] = None # نتائج التحليل المختلفة (بصمات، فيديو، فحص مادي) — null لو لم تصدر نتائج بعد
    linked_defendant_name: Optional[str] = None # اسم المتهم المرتبط بهذا التقرير لربطه بملفه في القضية
    source_document_id: Optional[str] = None
    
    @field_validator("items_sent_for_analysis", mode="before")
    @classmethod
    def coerce_items(cls, v):
        if not isinstance(v, list):
            return []
        coerced = []
        for item in v:
            if isinstance(item, str):
                coerced.append({"description": item})
            elif isinstance(item, dict):
                coerced.append(item)
        return coerced

class WitnessStatement(BaseModel):
    witness_name:           Optional[str]       = None # اسم الشاهد
    witness_type:           Optional[str]       = None # ['عيان','ضابط','مخبر']
    occupation:             Optional[str]       = None # العمل
    id_number:              Optional[str]       = None # رقم الكارنيه او البطاقه
    relation_to_defendant:  Optional[str]       = None # علاقته بالمتهم
    statement_date:         Optional[str]       = None # تاريخ الإدلاء بالشهادة.
    statement_summary:      Optional[str]       = None # ملخص كل سؤال و اجابته ليتم تقييمهم لاحقا
    was_sworn_in:           Optional[bool]      = None # إن صُرِّح بأداء اليمين.
    presence_at_scene:      Optional[bool]      = None # إن كان حاضراً في مكان الحادث.
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
    defendant_name:        Optional[str] = None  # اسم المتهم
    text:                  Optional[str] = None  # ملخص موقف المتهم
    confession_date:       Optional[str] = None  # تاريخ الاستجواب
    confession_stage:      Optional[str] = None  # تحقيق / محكمة
    legal_counsel_present: bool          = False # حضور المحامي ام لا
    coercion_claimed:      bool          = False # تعرض لتهديد او اكراه
    key_admissions:        List[str]     = Field(default_factory=list) # نقاط أقرّ بها المتهم حتى لو كان إنكاره جزئياً.
    source_document_id:    Optional[str] = None


class CriminalRecord(BaseModel):
    """صحيفة السوابق الجنائية لمتهم"""
    defendant_name:     Optional[str] = None                         # اسم المتهم
    record_summary:     Optional[str] = None                         # ملخص مختصر لمحتوى الصحيفة
    prior_cases:        List[str]     = Field(default_factory=list)  # قضايا سابقة إن ذُكرت
    source_document_id: Optional[str] = None

class CriminalProceedings(BaseModel):
    procedure_type:     Optional[str] = None  # ضبط / قبض / استجواب / تفتيش
    description:        Optional[str] = None  # الوصف
    warrant_present:    bool          = False # false إن كان الإذن غائباً
    conducting_officer: Optional[str] = None  # اسم وصفة الضابط المعني.
    source_document_id: Optional[str] = None

class DefenseDocument(BaseModel):
    submitted_by:          Optional[str]       = None # اسم المحامي مُقدِّم المذكرة.
    defendant_name:        Optional[str]       = None # اسم المتهم المدافَع عنه.
    formal_defenses:       List[str]           = Field(default_factory=list) # (تتعلق بصحة الإجراءات لا بموضوع الجريمة).
    substantive_defenses:  List[str]           = Field(default_factory=list) # الدفوع الموضوعية (تتعلق بأركان الجريمة وعدم توافرها).
    supporting_principles: List[str]           = Field(default_factory=list) # مبادئ محكمة النقض والمواد القانونية التي استند إليها الدفاع.
    alibi_claimed:         bool                = False # إن ادّعى الدفاع صراحةً وجود المتهم في مكان آخر وقت الحادث.
    alibi_description:     Optional[str]       = None # وصف إن وُجد (المكان، الزمان، الشهود الداعمون).
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