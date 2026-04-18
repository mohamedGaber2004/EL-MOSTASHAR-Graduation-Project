from typing import List, Optional, Dict
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime

from src.Utils import (
    Gender,
    MentalState,
    LawCode,
    ComplicityRole,
    EvidenceType,
    CourtLevel,
    ValidityStatus,
    VerdictType,
    WitnessRelation,
    WitnessType,
    ProcedureType,
    NullityType,
    IncidentType
)


# =============================================================================
#  DEFENDANT
# =============================================================================

class Defendant(BaseModel):
    """
    بيانات المتهم المستخلصة من ملفات القضية.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # ── هوية ──────────────────────────────────────────────────────────────────
    name:               str
    alias:              Optional[str]       = None          # الاسم المستعار / اللقب
    national_id:        Optional[str]       = None
    passport_number:    Optional[str]       = None
    gender:             Optional[Gender]    = None
    age:                Optional[int]       = None          # ← بيأثر في التخفيف
    date_of_birth:      Optional[datetime]  = None
    nationality:        Optional[str]       = None
    occupation:         Optional[str]       = None          # ← بيأثر في ظروف التخفيف
    address:            Optional[str]       = None

    # ── حالة نفسية / قانونية ─────────────────────────────────────────────────
    mental_state:       Optional[MentalState] = None        # ← جنون / إكراه / سكر
    mental_report_id:   Optional[str]       = None          # مرجع تقرير الطب الشرعي النفسي

    # ── سوابق ──────────────────────────────────────────────────────────────────
    prior_record:       Optional[bool]      = False
    prior_crimes:       List[str]           = Field(default_factory=list)  # ← نوع السوابق
    prior_sentences:    List[str]           = Field(default_factory=list)  # ← الأحكام السابقة

    # ── دور في الجريمة ────────────────────────────────────────────────────────
    complicity_role:    Optional[ComplicityRole] = None

    # ── وضع في القضية ────────────────────────────────────────────────────────
    in_custody:         Optional[bool]      = None
    arrest_date:        Optional[datetime]  = None
    detention_order_id: Optional[str]       = None          # رقم أمر الحبس الاحتياطي

    # ── مصدر البيانات ────────────────────────────────────────────────────────
    source_document_id: Optional[str]       = None
    notes:              Optional[str]       = None



# =============================================================================
#  CHARGE
# =============================================================================

class Charge(BaseModel):
    """
    التهم الموجهة — مستخلصة من أمر الإحالة أو قرار الاتهام.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # ── تعريف التهمة ──────────────────────────────────────────────────────────
    charge_id:              Optional[str]       = None      # معرف داخلي
    statute:                Optional[str]       = None      # نص المادة
    law_code:               Optional[LawCode]   = None      # القانون المنطبق
    article_number:         Optional[str]       = None      # رقم المادة
    article_paragraph:      Optional[str]       = None      # فقرة المادة إن وجدت
    incident_type:          Optional[IncidentType] = None

    description:            Optional[str]       = None      # وصف التهمة
    charge_date:            Optional[datetime]  = None      # تاريخ ارتكاب الجريمة
    charge_location:        Optional[str]       = None      # مكان الجريمة

    # ── عناصر الجريمة ────────────────────────────────────────────────────────
    elements_required:      List[str]           = Field(default_factory=list)  # أركان الجريمة
    elements_proven:        Dict[str, bool]     = Field(default_factory=dict)  # ← ثبت / لم يثبت

    # ── درجة الجريمة ─────────────────────────────────────────────────────────
    attempt_flag:           bool                = False      # شروع أم جريمة تامة؟
    complicity_role:        Optional[ComplicityRole] = None  # فاعل / شريك / محرض

    # ── العقوبة ───────────────────────────────────────────────────────────────
    penalty_range:          Optional[str]       = None      # الحد الأدنى والأقصى
    penalty_min:            Optional[str]       = None      # الحد الأدنى
    penalty_max:            Optional[str]       = None      # الحد الأقصى

    # ── ظروف ──────────────────────────────────────────────────────────────────
    aggravating_factors:    List[str]           = Field(default_factory=list)  # ظروف مشددة
    mitigating_factors:     List[str]           = Field(default_factory=list)  # ظروف مخففة

    # ── ربط بمتهمين ───────────────────────────────────────────────────────────
    linked_defendant_names: List[str]           = Field(default_factory=list)

    # ── مصدر ──────────────────────────────────────────────────────────────────
    source_document_id:     Optional[str]       = None
    notes:                  Optional[str]       = None




# =============================================================================
#  EVIDENCE
# =============================================================================

class Evidence(BaseModel):
    """
    الأدلة المادية والمستندية المضبوطة.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # ── هوية الدليل ───────────────────────────────────────────────────────────
    evidence_id:                Optional[str]           = None
    evidence_type:              Optional[EvidenceType]  = None
    description:                Optional[str]           = None

    # ── الضبط ─────────────────────────────────────────────────────────────────
    seizure_date:               Optional[datetime]      = None   # تاريخ الضبط
    seizure_location:           Optional[str]           = None   # مكان الضبط ← مهم للتفتيش
    seized_by:                  Optional[str]           = None   # ضابط/مخبر ضبطه
    seizure_warrant_present:    Optional[bool]          = None   # هل في إذن تفتيش؟

    # ── سلسلة الحيازة ────────────────────────────────────────────────────────
    chain_of_custody_ok:        Optional[bool]          = None   # سلامة سلسلة الحيازة
    chain_of_custody_notes:     Optional[str]           = None   # ملاحظات سلسلة الحيازة
    storage_conditions_ok:      Optional[bool]          = None   # ظروف التخزين سليمة؟

    # ── الصلاحية ──────────────────────────────────────────────────────────────
    validity_status:            Optional[ValidityStatus] = None
    invalidity_reason:          Optional[str]           = None   # سبب البطلان إن وُجد

    # ── الربط ─────────────────────────────────────────────────────────────────
    linked_charge_ids:          List[str]               = Field(default_factory=list)
    linked_charge_elements:     List[str]               = Field(default_factory=list)
    linked_defendant_name:      Optional[str]           = None   # بيخص متهم معين

    # ── مصدر ──────────────────────────────────────────────────────────────────
    source_document_id:         Optional[str]           = None
    page_reference:             Optional[str]           = None   # رقم الصفحة في الملف
    notes:                      Optional[str]           = None





# =============================================================================
#  LAB REPORT
# =============================================================================

class LabReport(BaseModel):
    """
    تقارير الطب الشرعي والمعامل الجنائية.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # ── هوية التقرير ──────────────────────────────────────────────────────────
    report_id:                  Optional[str]       = None
    report_type:                Optional[str]       = None   # كيمياء / دم / بصمات / DNA / سموم
    lab_name:                   Optional[str]       = None   # اسم المعمل / الجهة
    examiner_name:              Optional[str]       = None   # اسم الخبير
    examination_date:           Optional[datetime]  = None

    # ── العينة ────────────────────────────────────────────────────────────────
    sample_id:                  Optional[str]       = None
    sample_type:                Optional[str]       = None   # دم / مسحة / بصمة / سلاح
    sample_source:              Optional[str]       = None   # من أين أُخذت العينة
    sample_chain_intact:        Optional[bool]      = None   # سلسلة حيازة العينة سليمة؟

    # ── النتيجة ───────────────────────────────────────────────────────────────
    result:                     Optional[str]       = None
    result_is_positive:         Optional[bool]      = None   # إيجابي / سلبي
    confirms_charge_element:    Optional[bool]      = None   # يثبت ركن من أركان التهمة؟
    linked_charge_element:      Optional[str]       = None   # أي ركن؟

    # ── الربط ─────────────────────────────────────────────────────────────────
    linked_evidence_id:         Optional[str]       = None
    linked_defendant_name:      Optional[str]       = None

    # ── مصدر ──────────────────────────────────────────────────────────────────
    source_document_id:         Optional[str]       = None
    notes:                      Optional[str]       = None




# =============================================================================
#  WITNESS STATEMENT
# =============================================================================

class WitnessStatement(BaseModel):
    """
    أقوال الشهود المستخلصة من محاضر التحقيق والمحاكمة.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # ── هوية الشاهد ───────────────────────────────────────────────────────────
    witness_name:               Optional[str]           = None
    witness_national_id:        Optional[str]           = None
    witness_type:               Optional[WitnessType]   = None
    witness_occupation:         Optional[str]           = None
    relation_to_defendant:      Optional[WitnessRelation] = None  # قريب / عدو / محايد / ضابط

    # ── الإجراءات ─────────────────────────────────────────────────────────────
    statement_date:             Optional[datetime]      = None
    statement_stage:            Optional[str]           = None   # تحقيق / محاكمة / نيابة
    was_sworn_in:               Optional[bool]          = None   # أدى اليمين؟ ← إجرائي مهم
    presence_at_scene:          Optional[bool]          = None   # كان موجود في مكان الجريمة؟

    # ── المضمون ───────────────────────────────────────────────────────────────
    statement_summary:          Optional[str]           = None
    key_facts_mentioned:        List[str]               = Field(default_factory=list)  # وقائع ذكرها

    # ── الاتساق ───────────────────────────────────────────────────────────────
    consistency_flag:           Optional[bool]          = None   # متسق مع باقي الأدلة؟
    contradicts_other_evidence: Optional[bool]          = None
    contradiction_details:      Optional[str]           = None   # تفاصيل التناقض
    prior_statement_exists:     Optional[bool]          = None   # أدلى بأقوال سابقة؟
    retracted_statement:        Optional[bool]          = None   # عدل عن أقواله؟

    # ── الموثوقية ─────────────────────────────────────────────────────────────
    credibility_flags:          List[str]               = Field(default_factory=list)  # عوامل تأثر الموثوقية

    # ── مصدر ──────────────────────────────────────────────────────────────────
    source_document_id:         Optional[str]           = None
    page_reference:             Optional[str]           = None
    notes:                      Optional[str]           = None




# =============================================================================
#  CONFESSION
# =============================================================================

class Confession(BaseModel):
    """
    اعترافات المتهمين — بيانات الاستخلاص الإجرائي والموضوعي.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # ── هوية ──────────────────────────────────────────────────────────────────
    confession_id:              Optional[str]       = None
    defendant_name:             str

    # ── الظروف الإجرائية ─────────────────────────────────────────────────────
    confession_date:            Optional[datetime]  = None
    confession_stage:           Optional[str]       = None   # شرطة / نيابة / محكمة
    obtained_before_arrest:     Optional[bool]      = None   # قبل القبض الرسمي؟
    obtained_after_arrest:      Optional[bool]      = None   # ← بعد القبض؟ — إجرائي حاسم
    legal_counsel_present:      Optional[bool]      = None   # ← حضر محامي؟ — حق دستوري
    informed_of_rights:         Optional[bool]      = None   # أُخبر بحقوقه؟
    coercion_claimed:           Optional[bool]      = None   # ادعى الإكراه؟
    coercion_evidence:          Optional[str]       = None   # دليل على الإكراه المزعوم

    # ── المضمون ───────────────────────────────────────────────────────────────
    text:                       Optional[str]       = None   # نص الاعتراف
    key_admissions:             List[str]           = Field(default_factory=list)  # الاعترافات الجوهرية

    # ── الموثوقية ─────────────────────────────────────────────────────────────
    voluntary:                  Optional[bool]      = None   # طوعي؟
    consistent_with_facts:      Optional[bool]      = None   # يتسق مع وقائع القضية؟
    consistent_with_evidence:   Optional[bool]      = None   # يتسق مع الأدلة المادية؟
    retracted:                  Optional[bool]      = None   # ← عدل عنه؟ — مهم جدًا
    retraction_date:            Optional[datetime]  = None
    retraction_reason:          Optional[str]       = None   # سبب العدول

    # ── مصدر ──────────────────────────────────────────────────────────────────
    source_document_id:         Optional[str]       = None
    page_reference:             Optional[str]       = None
    notes:                      Optional[str]       = None




# =============================================================================
#  PROCEDURAL ISSUE
# =============================================================================

class ProceduralIssue(BaseModel):
    """
    المسائل الإجرائية المستخلصة من ملفات القضية —
    تُمرَّر لـ Procedural Auditor Agent للتقييم القانوني.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # ── تعريف الإجراء ─────────────────────────────────────────────────────────
    issue_id:                   Optional[str]           = None
    procedure_type:             Optional[ProcedureType] = None
    procedure_date:             Optional[datetime]      = None
    conducting_officer:         Optional[str]           = None   # من أجرى الإجراء
    conducting_authority:       Optional[str]           = None   # الجهة (شرطة / نيابة / قاضي)

    # ── السلامة الإجرائية ─────────────────────────────────────────────────────
    warrant_present:            Optional[bool]          = None   # وجود إذن / أمر قضائي
    warrant_id:                 Optional[str]           = None   # رقم الإذن
    notification_to_da:         Optional[bool]          = None   # تم إخطار النيابة؟
    notification_time:          Optional[str]           = None   # وقت الإخطار
    legal_timeframes_respected: Optional[bool]          = None   # المواعيد القانونية ملتزم بها؟

    # ── المشكلة المستخلصة ─────────────────────────────────────────────────────
    issue_description:          Optional[str]           = None   # وصف الإشكالية الإجرائية
    nullity_type:               Optional[NullityType]   = None   # بطلان مطلق / نسبي / لا بطلان
    article_basis:              Optional[str]           = None   # المادة القانونية للبطلان المزعوم

    # ── الأثر ─────────────────────────────────────────────────────────────────
    affects_evidence_ids:       List[str]               = Field(default_factory=list)
    may_invalidate_case:        Optional[bool]          = None
    tainted_fruit_risk:         Optional[bool]          = None   # خطر ثمرة الشجرة المسمومة

    # ── مصدر ──────────────────────────────────────────────────────────────────
    source_document_id:         Optional[str]           = None
    page_reference:             Optional[str]           = None
    notes:                      Optional[str]           = None




# =============================================================================
#  PRIOR JUDGMENT  (مستخلص من أحكام سابقة — للمقارنة والاستدلال)
# =============================================================================

class PriorJudgment(BaseModel):
    """
    بيانات الأحكام السابقة المستخلصة — للاستدلال القانوني والمقارنة.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # ── تعريف الحكم ───────────────────────────────────────────────────────────
    judgment_id:                Optional[str]           = None
    case_reference:             Optional[str]           = None   # رقم القضية
    court_name:                 Optional[str]           = None
    court_level:                Optional[CourtLevel]    = None
    verdict_date:               Optional[datetime]      = None
    verdict_type:               Optional[VerdictType]   = None

    # ── موضوع الحكم ───────────────────────────────────────────────────────────
    charge_statutes:            List[str]               = Field(default_factory=list)
    key_legal_principles:       List[str]               = Field(default_factory=list)  # مبادئ قانونية مقررة
    penalty_imposed:            Optional[str]           = None

    # ── الاستخدام ─────────────────────────────────────────────────────────────
    relevance_to_current_case:  Optional[str]           = None   # أوجه الصلة بالقضية الحالية

    # ── مصدر ──────────────────────────────────────────────────────────────────
    source_document_id:         Optional[str]           = None
    notes:                      Optional[str]           = None




# =============================================================================
#  DEFENSE DOCUMENT  (مستندات الدفاع)
# =============================================================================

class DefenseDocument(BaseModel):
    """
    مستندات ومذكرات الدفاع المستخلصة.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # ── تعريف ─────────────────────────────────────────────────────────────────
    doc_id:                     Optional[str]           = None
    doc_type:                   Optional[str]           = None   # مذكرة / تقرير طبي / حجج دفاع
    submission_date:            Optional[datetime]      = None
    submitted_by:               Optional[str]           = None   # المحامي / المتهم

    # ── الدفوع المستخلصة ──────────────────────────────────────────────────────
    formal_defenses:            List[str]               = Field(default_factory=list)   # دفوع شكلية
    substantive_defenses:       List[str]               = Field(default_factory=list)   # دفوع موضوعية
    alibi_claimed:              Optional[bool]          = None   # ادعاء حضوري في مكان آخر؟
    alibi_details:              Optional[str]           = None

    # ── الأدلة الدفاعية ───────────────────────────────────────────────────────
    defense_evidence_ids:       List[str]               = Field(default_factory=list)
    supporting_principles:      List[str]               = Field(default_factory=list)   # مبادئ نقض مستشهد بها

    # ── مصدر ──────────────────────────────────────────────────────────────────
    source_document_id:         Optional[str]           = None
    notes:                      Optional[str]           = None




# =============================================================================
#  CASE INCIDENT  (وقائع القضية الأساسية)
# =============================================================================

class CaseIncident(BaseModel):
    """
    الوقائع الجوهرية للقضية المستخلصة من محاضر الضبط والتحقيق.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    incident_id:                Optional[str]           = None
    incident_type:              Optional[IncidentType]  = None
    incident_date:              Optional[datetime]      = None
    incident_location:          Optional[str]           = None
    incident_description:       Optional[str]           = None

    # ── الأطراف ───────────────────────────────────────────────────────────────
    perpetrator_names:          List[str]               = Field(default_factory=list)
    victim_names:               List[str]               = Field(default_factory=list)
    witness_names:              List[str]               = Field(default_factory=list)

    # ── النتيجة ───────────────────────────────────────────────────────────────
    outcome:                    Optional[str]           = None   # نتيجة الواقعة (وفاة / إصابة / خسارة مالية)
    outcome_severity:           Optional[str]           = None   # درجة الخطورة

    # ── الربط ─────────────────────────────────────────────────────────────────
    linked_evidence_ids:        List[str]               = Field(default_factory=list)
    linked_charge_ids:          List[str]               = Field(default_factory=list)

    source_document_id:         Optional[str]           = None
    notes:                      Optional[str]           = None