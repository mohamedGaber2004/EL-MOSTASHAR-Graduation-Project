from typing import List, Optional, Any
from pydantic import BaseModel, Field
from pydantic import BaseModel, field_validator
import json, re


# ══════════════════════════════════════════════════════
#  Shared coercion helpers (module-level, reused across models)
# ══════════════════════════════════════════════════════
def _coerce_str_list(v: Any) -> List[str]:
    """
    Normalise an LLM list-of-strings field.

    Handles:
      - None / missing           → []
      - bare string              → [string]
      - list of strings          → as-is
      - list of dicts            → extract first non-null string value per dict
      - list of mixed types      → best-effort extraction
    """
    if v is None:
        return []
    if isinstance(v, str):
        return [v] if v.strip() else []
    if isinstance(v, list):
        result: list[str] = []
        for item in v:
            if isinstance(item, str):
                if item.strip():
                    result.append(item)
            elif isinstance(item, dict):
                # pick the first non-empty string value in the dict
                for val in item.values():
                    if isinstance(val, str) and val.strip():
                        result.append(val)
                        break
        return result
    return []

def _coerce_bool_required(v: Any) -> bool:
    """
    Coerce an LLM boolean field where the default is False.
    Handles Arabic affirmatives, English strings, ints.
    """
    if isinstance(v, bool):
        return v
    if isinstance(v, int):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in {"true", "yes", "1", "نعم", "صحيح", "موجود"}
    return False

def _coerce_bool_optional(v: Any) -> Optional[bool]:
    """
    Coerce an LLM boolean field where None is semantically meaningful
    (i.e. 'we don't know').
    """
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, int):
        return bool(v)
    if isinstance(v, str):
        stripped = v.strip().lower()
        if stripped in {"true", "yes", "1", "نعم", "صحيح", "موجود"}:
            return True
        if stripped in {"false", "no", "0", "لا", "غير موجود", "لم"}:
            return False
        return None  # ambiguous string → unknown
    return None

def _coerce_name_with_title(v: Any) -> Optional[str]:
    """
    Coerce an LLM 'name + title/rank' field (officer name + rank,
    examiner name + title, etc).

    Handles:
      - plain string                       → as-is
      - dict {"name": ..., "title": ...}   → combined "name - title",
        dropping whichever side is missing/empty instead of inserting
        the literal word 'None'
      - list (defensive)                   → joined string

    Fixes the production error where examiner_name arrived as
    {'name': None, 'title': '...نائي المختص'} and Pydantic rejected it
    because the field is a plain str with no coercion.
    """
    if v is None:
        return None
    if isinstance(v, str):
        return v
    if isinstance(v, dict):
        name  = v.get("name")  or v.get("الاسم")  or ""
        title = v.get("title") or v.get("الصفة")  or v.get("role") or v.get("صفة") or ""
        parts = [str(p).strip() for p in (name, title) if p and str(p).strip()]
        return " - ".join(parts) or None
    if isinstance(v, list):
        flat = _coerce_str_list(v)
        return "، ".join(flat) or None
    return v

def _coerce_singular_name(v: Any) -> Optional[str]:
    """
    Coerce a field meant to hold ONE person's name, but which LLMs
    sometimes wrap in a list (single- or multi-element) or return as
    a dict.

    Fixes the production error where linked_defendant_name arrived as
    ['شريف محمد خل...د فرحات سعيد'] and Pydantic rejected it because
    the field is a plain str with no coercion.
    """
    if v is None:
        return None
    if isinstance(v, str):
        return v
    if isinstance(v, list):
        flat = _coerce_str_list(v)
        return "، ".join(flat) or None
    if isinstance(v, dict):
        for val in v.values():
            if isinstance(val, str) and val.strip():
                return val
        return None
    return v

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
    @field_validator("age", mode="before")
    @classmethod
    def coerce_age(cls, v: Any) -> Optional[int]:
        """
        LLMs often return age as a string ('32', 'نحو 30', '~30').
        Extract the first integer found; return None if not parsable.
        """
        if v is None:
            return None
        if isinstance(v, int):
            return v
        if isinstance(v, float):
            return int(v)
        if isinstance(v, str):
            m = re.search(r'\d+', v)
            return int(m.group()) if m else None
        return None

    @field_validator("nationality", mode="before")
    @classmethod
    def coerce_nationality(cls, v: Any) -> str:
        """
        The shared prompt rule tells the model to emit null for any
        unstated field. Egyptian criminal documents usually only state
        nationality explicitly when it's NOT Egyptian, so a literal
        null from the model would otherwise wipe out the 'مصري'
        default (Pydantic only applies a field default when the key is
        missing, not when it's present as null). Re-apply the default
        here so that behavior survives regardless of what the model
        sends.
        """
        if v is None:
            return "مصري"
        if isinstance(v, str) and v.strip():
            return v.strip()
        return "مصري"

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
    @field_validator("attempt_flag", mode="before")
    @classmethod
    def coerce_attempt_flag(cls, v: Any) -> bool:
        return _coerce_bool_required(v)

    @field_validator("linked_defendant_names", mode="before")
    @classmethod
    def coerce_linked_defendant_names(cls, v: Any) -> List[str]:
        return _coerce_str_list(v)

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
    def coerce_date(cls, v: Any) -> Optional[str]:
        """
        Normalise date fields — LLMs sometimes return a dict with
        'start' / 'end' keys for date ranges.
        """
        if isinstance(v, dict):
            start = v.get("start", "")
            end   = v.get("end", "")
            if start and end:
                return f"{start} إلى {end}"
            return start or end or None
        return v

    @field_validator("perpetrator_names", "victim_names", mode="before")
    @classmethod
    def coerce_name_lists(cls, v: Any) -> List[str]:
        return _coerce_str_list(v)

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
    @field_validator("seizure_warrant_present", mode="before")
    @classmethod
    def coerce_seizure_warrant_present(cls, v: Any) -> bool:
        return _coerce_bool_required(v)

    @field_validator("seized_by", mode="before")
    @classmethod
    def coerce_seized_by(cls, v: Any) -> Optional[str]:
        """Same 'name + rank' drift risk as examiner_name/conducting_officer."""
        return _coerce_name_with_title(v)

    @field_validator("linked_defendant_name", mode="before")
    @classmethod
    def coerce_linked_defendant_name(cls, v: Any) -> Optional[str]:
        """Same singular-name-arrives-as-list/dict drift as LabReport's."""
        return _coerce_singular_name(v)

class EvidenceItem(BaseModel):
    description: str                      # وصف العنصر المُرسَل للتحليل
    evidence_number: Optional[str] = None # رقم الحرز المُسجَّل في محضر الضبط

class LabReport(BaseModel):
    report_type:     Optional[str] = None # نوع التقرير (طبي شرعي / كيميائي / بلستي / غيره)
    report_number:   Optional[str] = None # رقم التقرير الرسمي الصادر من المعمل
    examination_date:Optional[str] = None # تاريخ إجراء الفحص بالمعمل
    examiner_name:   Optional[str] = None # اسم الخبير أو الطبيب الشرعي المُجري للفحص
    prosecutor_name: Optional[str] = None # اسم عضو النيابة الآمر بالفحص 
    items_sent_for_analysis: list[EvidenceItem] = Field(default_factory=list) # قائمة العناصر المُرسَلة للمعمل للتحليل (أحراز، مستندات، تسجيلات)
    result: Optional[str] = None # نتائج التحليل المختلفة (بصمات، فيديو، فحص مادي) — null لو لم تصدر نتائج بعد
    linked_defendant_name: Optional[str] = None # اسم المتهم المرتبط بهذا التقرير لربطه بملفه في القضية
    source_document_id: Optional[str] = None

    @field_validator("examiner_name", mode="before")
    @classmethod
    def coerce_examiner_name(cls, v: Any) -> Optional[str]:
        """
        Production-error fix: model returned
        {'name': None, 'title': '...نائي المختص'} instead of a string.
        """
        return _coerce_name_with_title(v)

    @field_validator("linked_defendant_name", mode="before")
    @classmethod
    def coerce_linked_defendant_name(cls, v: Any) -> Optional[str]:
        """
        Production-error fix: model returned
        ['شريف محمد خل...د فرحات سعيد'] instead of a string.
        """
        return _coerce_singular_name(v)

    @field_validator("items_sent_for_analysis", mode="before")
    @classmethod
    def coerce_items(cls, v: Any) -> list:
        """
        Normalise items_sent_for_analysis:
          - bare string        → [{"description": string}]
          - list of strings    → [{"description": s} for s in list]
          - list of dicts      → ensure 'description' key is always present
          - missing desc       → reconstruct from sibling keys or repr
        """
        if not isinstance(v, list):
            if isinstance(v, str) and v.strip():
                return [{"description": v}]
            return []

        coerced = []
        for item in v:
            if isinstance(item, str):
                if item.strip():
                    coerced.append({"description": item})
            elif isinstance(item, dict):
                if not item.get("description"):
                    # Try common sibling keys before falling back to repr
                    item["description"] = (
                        item.get("name")
                        or item.get("item")
                        or item.get("content")
                        or item.get("text")
                        or str(item)
                    )
                coerced.append(item)
        return coerced

    @field_validator("result", mode="before")
    @classmethod
    def coerce_result(cls, v: Any) -> Optional[str]:
        """LLMs sometimes return structured dict results — serialise to string."""
        if isinstance(v, dict):
            return json.dumps(v, ensure_ascii=False)
        if isinstance(v, list):
            return json.dumps(v, ensure_ascii=False)
        return v

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

    @field_validator("statement_summary", mode="before")
    @classmethod
    def coerce_statement_summary(cls, v: Any) -> Optional[str]:
        """
        Normalise statement_summary:
          - list of Q/A dicts → formatted string
          - list of strings   → joined string
          - plain string      → as-is
        """
        if isinstance(v, list):
            parts: list[str] = []
            for item in v:
                if isinstance(item, dict):
                    q = item.get("question") or ""
                    a = item.get("answer") or ""
                    parts.append(f"س: {q}\nج: {a}".strip() if q else a)
                elif isinstance(item, str):
                    parts.append(item)
            return "\n\n".join(filter(None, parts)) or None
        return v

    @field_validator("was_sworn_in", "presence_at_scene", mode="before")
    @classmethod
    def coerce_optional_bools(cls, v: Any) -> Optional[bool]:
        return _coerce_bool_optional(v)

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
    @field_validator("legal_counsel_present", "coercion_claimed", mode="before")
    @classmethod
    def coerce_bools(cls, v: Any) -> bool:
        return _coerce_bool_required(v)

    @field_validator("key_admissions", mode="before")
    @classmethod
    def coerce_key_admissions(cls, v: Any) -> List[str]:
        return _coerce_str_list(v)

class CriminalRecord(BaseModel):
    """صحيفة السوابق الجنائية لمتهم"""
    defendant_name:     Optional[str] = None # اسم المتهم
    record_summary:     Optional[str] = None # ملخص مختصر لمحتوى الصحيفة
    prior_cases:        List[str]     = Field(default_factory=list)  # قضايا سابقة إن ذُكرت
    source_document_id: Optional[str] = None
    @field_validator("prior_cases", mode="before")
    @classmethod
    def coerce_prior_cases(cls, v: Any) -> List[str]:
        return _coerce_str_list(v)

class CriminalProceedings(BaseModel):
    procedure_type:     Optional[str] = None  # ضبط / قبض / استجواب / تفتيش
    description:        Optional[str] = None  # الوصف
    warrant_present:    Optional[bool]= False # false إن كان الإذن غائباً
    conducting_officer: Optional[str] = None  # اسم وصفة الضابط المعني.
    source_document_id: Optional[str] = None
    @field_validator("warrant_present", mode="before")
    @classmethod
    def coerce_warrant_present(cls, v: Any) -> Optional[bool]:
        return _coerce_bool_optional(v)

    @field_validator("conducting_officer", mode="before")
    @classmethod
    def coerce_conducting_officer(cls, v: Any) -> Optional[str]:
        """Same 'name + rank' drift risk as examiner_name/seized_by."""
        return _coerce_name_with_title(v)

class DefenseDocument(BaseModel):
    submitted_by:          Optional[str]       = None # اسم المحامي مُقدِّم المذكرة.
    defendant_name:        Optional[str]       = None # اسم المتهم المدافَع عنه.
    formal_defenses:       List[str]           = Field(default_factory=list) # (تتعلق بصحة الإجراءات لا بموضوع الجريمة).
    substantive_defenses:  List[str]           = Field(default_factory=list) # الدفوع الموضوعية (تتعلق بأركان الجريمة وعدم توافرها).
    supporting_principles: List[str]           = Field(default_factory=list) # مبادئ محكمة النقض والمواد القانونية التي استند إليها الدفاع.
    alibi_claimed:         bool                = False # إن ادّعى الدفاع صراحةً وجود المتهم في مكان آخر وقت الحادث.
    alibi_description:     Optional[str]       = None  # وصف إن وُجد (المكان، الزمان، الشهود الداعمون).
    source_document_id:    Optional[str]       = None
    @field_validator("alibi_claimed", mode="before")
    @classmethod
    def coerce_alibi_claimed(cls, v: Any) -> bool:
        return _coerce_bool_required(v)

    @field_validator(
        "formal_defenses",
        "substantive_defenses",
        "supporting_principles",
        mode="before",
    )
    @classmethod
    def coerce_defense_lists(cls, v: Any) -> List[str]:
        return _coerce_str_list(v)

    @field_validator("alibi_description", mode="before")
    @classmethod
    def coerce_alibi_description(cls, v: Any) -> Optional[str]:
        """
        LLMs sometimes return alibi as a structured dict with
        time / location / supporting_witnesses keys.
        """
        if isinstance(v, dict):
            time      = v.get("time", "")
            location  = v.get("location", "")
            witnesses = "، ".join(v.get("supporting_witnesses", []))
            parts     = filter(None, [time, location, f"شهود: {witnesses}" if witnesses else ""])
            return " — ".join(parts) or None
        return v