from typing import Optional, List
from pydantic import BaseModel, Field

from src.Utils.Enums.agents_enums import (
    CoercionType,
    ConfessionAdmissibilityStatus,
    WitnessReliabilityLevel,
    CivilClaimStatus,
    ProsecutionArgumentStrength
)

# ══════════════════════════════════════════════════════
#  Analysis Output Models  (populated by downstream agents)
# ══════════════════════════════════════════════════════
class ProsecutionNarrative(BaseModel):
    # ── وقائع الاتهام ─────────────────────────────────────────────
    summary:                 str           = Field(description="ملخص موجز لوقائع الاتهام كما يراها المدّعي العام (3-5 جمل)")
    incident_reconstruction: str           = Field(description="إعادة تسلسل وقائع الحادثة زمنياً من منظور النيابة")
    motive:                  Optional[str] = Field(default=None,description="الدافع المزعوم للجريمة إن ظهر من الأوراق")
 
    # ── ربط الأدلة بالاتهام ───────────────────────────────────────
    evidence_chain:     List[str]     = Field(default_factory=list,description="سلسلة الأدلة المادية الداعمة للاتهام مرتبةً حسب الأهمية")
    confession_role:    Optional[str] = Field(default=None,description="دور الاعتراف — إن وُجد — في بناء الاتهام")
    witness_summary:    Optional[str] = Field(default=None,description="ملخص إسهام شهادات الشهود في تأييد رواية النيابة")
    lab_report_summary: Optional[str] = Field(default=None,description="ملخص دور التقارير الفنية والمعملية في إثبات الاتهام")
 
    # ── ربط التهم بعناصرها القانونية ──────────────────────────────
    charge_element_mapping: List[dict] = Field(
        default_factory=list,
        description=(
            "قائمة بكل تهمة وعناصرها القانونية المتوفرة في الأوراق. "
            "كل عنصر: {charge_description, article, elements_proven: [str], elements_missing: [str]}"
        )
    )
 
    # ── التقييم الكلي ─────────────────────────────────────────────
    overall_strength:              ProsecutionArgumentStrength = Field(description="التقدير الإجمالي لقوة ملف الاتهام")
    weaknesses:                    List[str]                   = Field(default_factory=list,description="نقاط الضعف في ملف الاتهام التي ينبغي أن يراها القاضي")
    recommended_verdict_direction: str = Field(
        description=(
            "توجيه الحكم المقترح من منظور الاتهام: "
            "'إدانة كاملة' / 'إدانة جزئية' / 'تخفيف' مع مبرر"
        )
    )

class ProsecutionArgument(BaseModel):
    charge_description:      str                         = Field(description="وصف التهمة التي تخصّها هذه الحجة")
    article_reference:       Optional[str]               = Field(default=None,description="المادة القانونية المستند إليها (مثال: المادة 234 عقوبات)")
    argument_text:           str                         = Field(description="نص الحجة الاتهامية كما سيُعرض على القاضي")
    supporting_evidence_ids: List[str]                   = Field(default_factory=list,description="معرّفات الأدلة الداعمة لهذه الحجة من قائمة evidences")
    strength:                ProsecutionArgumentStrength = Field(description="قوة هذه الحجة بالذات")
    rebuttal_risk:           Optional[str]               = Field(default=None,description="أبرز ما يمكن أن يرد به الدفاع على هذه الحجة")
 
class ConfessionAssessment(BaseModel):

    defendant_name:          str           = Field(description="اسم المتهم صاحب الاعتراف")
    confession_date:         Optional[str] = Field(default=None,description="تاريخ الاعتراف")
    confession_text_summary: Optional[str] = Field(default=None,description="ملخص مضمون الاعتراف")
 
    # ── الإكراه ───────────────────────────────────────────────────
    coercion_alleged:  bool          = Field(default=False,description="هل ادّعى المتهم أو الدفاع وجود إكراه؟")
    coercion_type:     CoercionType  = Field(default=CoercionType.NONE,description="نوع الإكراه المزعوم")
    coercion_evidence: Optional[str] = Field(default=None,description="الدليل على الإكراه إن وُجد (شهادة، تقرير طبي، ...)")
 
    # ── الإجراءات ─────────────────────────────────────────────────
    taken_before_competent_authority: bool            = Field(default=True,description="هل أُخذ الاعتراف أمام جهة مختصة (نيابة / قاضٍ)؟")
    miranda_rights_observed:          bool            = Field(default=True,description="هل أُحيط المتهم علماً بحقه في الصمت والاستعانة بمحامٍ؟")
    time_since_arrest_hours:          Optional[float] = Field(default=None,description="الساعات المنقضية بين القبض والاعتراف")
    procedural_notes:                 Optional[str]   = Field(default=None,description="ملاحظات إجرائية أخرى تؤثر على صحة الاعتراف")
 
    # ── المادة 302 إجراءات جنائية ─────────────────────────────────
    standalone_confession:  bool      = Field(default=False,description="هل الاعتراف هو الدليل الوحيد بلا سند مادي مستقل؟")
    corroborating_evidence: List[str] = Field(default_factory=list,description="الأدلة المادية المعززة للاعتراف إن وُجدت")
 
    # ── الحكم النهائي ─────────────────────────────────────────────
    admissibility_status: ConfessionAdmissibilityStatus = Field(description="حكم قبول الاعتراف")
    legal_reasoning:      str                           = Field(description="التسبيب القانوني لحكم القبول / الرفض")
    impact_on_case:       str                           = Field(
        description=(
            "أثر هذا التقييم على القضية: "
            "'يُعزز الإدانة' / 'يُضعف الادعاء' / 'يستوجب الاستبعاد'"
        )
    )
 
class WitnessCredibility(BaseModel):

    witness_name:              str           = Field(description="اسم الشاهد")
    statement_date:            Optional[str] = Field(default=None,description="تاريخ أداء الشهادة")
    relationship_to_defendant: Optional[str] = Field(default=None,description="صلة الشاهد بالمتهم أو المجني عليه إن وُجدت")
 
    # ── اتساق الرواية ─────────────────────────────────────────────
    internal_consistency:             bool      = Field(default=True,description="هل الشهادة متسقة داخلياً بلا تناقضات ذاتية؟")
    consistent_with_prior_statements: bool      = Field(default=True,description="هل تتفق مع تصريحاته السابقة (محاضر الضبط / أقوال الاستئناف)؟")
    contradictions_found:             List[str] = Field(default_factory=list,description="التناقضات المحددة مع الروايات الأخرى أو الأدلة المادية")
 
    # ── التوافق مع الأدلة المادية ─────────────────────────────────
    corroborated_by_physical_evidence: bool          = Field(default=False,description="هل تتوافق الشهادة مع الأدلة المادية المضبوطة؟")
    corroboration_details:             Optional[str] = Field(default=None,description="تفاصيل التوافق أو التعارض مع الأدلة المادية")
 
    # ── احتمالية التحيز ───────────────────────────────────────────
    potential_bias: Optional[str] = Field(default=None,description="أي مؤشر على تحيز محتمل (عداوة / مصلحة / ضغط)")
    demeanor_notes: Optional[str] = Field(default=None,description="ملاحظات على سلوك الشاهد أثناء الأداء إن وُجدت في الأوراق")
 
    # ── الحكم النهائي ─────────────────────────────────────────────
    reliability_level:     WitnessReliabilityLevel = Field(description="درجة الموثوقية الإجمالية")
    reliability_reasoning: str                     = Field(description="التسبيب الموجز لدرجة الموثوقية المُقدَّرة")
    weight_in_case:        str                     = Field(
        description=(
            "الوزن المقترح في سلسلة الإثبات: "
            "'ركيزة أساسية' / 'مساند' / 'استئناسي' / 'يُهمَل'"
        )
    )
 
 
class CivilClaim(BaseModel):
    # ── بيانات المدّعي المدني ─────────────────────────────────────
    plaintiff_name:     Optional[str] = Field(description="اسم المدّعي المدني (المجني عليه أو ورثته)")
    plaintiff_capacity: Optional[str] = Field(default=None,description="صفة المدّعي: مجني عليه / وارث / ولي / وكيل")
 
    # ── طلبات التعويض ─────────────────────────────────────────────
    compensation_requested: Optional[float] = Field(default=None,description="مبلغ التعويض المطلوب بالجنيه المصري")
    compensation_basis:     Optional[str]   = Field(default=None,description="أساس طلب التعويض: أضرار مادية / معنوية / عدم كسب / نفقات علاج")
    material_damages:       Optional[float] = Field(default=None,description="قيمة الأضرار المادية الموثقة بالجنيه")
    moral_damages:          Optional[float] = Field(default=None,description="قيمة الأضرار الأدبية / المعنوية بالجنيه")
 
    # ── الوثائق الداعمة ───────────────────────────────────────────
    supporting_documents: List[str] = Field(default_factory=list,description="المستندات الداعمة للادعاء المدني (فواتير / تقارير طبية / ...)")
 
    # ── حالة الدعوى ───────────────────────────────────────────────
    status: CivilClaimStatus = Field(default=CivilClaimStatus.NOT_FILED,description="الحالة الراهنة للدعوى المدنية")
 
    # ── التقدير المقترح ───────────────────────────────────────────
    suggested_award:         Optional[float] = Field(default=None,description="مبلغ التعويض المقترح بناءً على تقدير وكيل التسوية")
    award_reasoning:         Optional[str] = Field(default=None,description="أسباب التقدير المقترح")
    award_against_defendant: Optional[str] = Field(default=None,description="اسم المتهم الملزَم بالتعويض إن تعدد المتهمون")
    solidarity_liability:    bool = Field(default=False,description="هل المسؤولية تضامنية بين المتهمين المتعددين؟")


class EvidenceScoring(BaseModel):
    # ── بيانات الدليل ─────────────────────────────────────────────
    evidence_id:  Optional[str] = Field(default=None, description="المعرّف الفريد للدليل كما وَرد في قائمة الأدلة")
    type:         Optional[str] = Field(default=None, description="نوع الدليل: مادي / قولي / فني / وثائقي")
    description:  Optional[str] = Field(default=None, description="وصف موجز للدليل ومصدره")

    # ── التقييم ───────────────────────────────────────────────────
    strength:        Optional[str] = Field(default=None, description="قوة الدليل: قوي / متوسط / ضعيف")
    strength_reason: Optional[str] = Field(default=None, description="مسوّغ درجة القوة المُقدَّرة")
    admissibility:   Optional[str] = Field(default=None, description="حكم القبول: مقبول / مستبعد — مستنَد إلى نتيجة فحص البطلان")
    proof_degree:    Optional[str] = Field(default=None, description="درجة الإثبات: قاطع / كافٍ / غير كافٍ / منعدم")

    # ── ملاحظات ───────────────────────────────────────────────────
    notes: Optional[str] = Field(default=None, description="ملاحظات إضافية تؤثر على تقييم الدليل")


class DefenseArgument(BaseModel):
    # ── تصنيف الدفع ───────────────────────────────────────────────
    argument_type: Optional[str] = Field(default=None, description="نوع الدفع: شكلي / موضوعي")
    description:   Optional[str] = Field(default=None, description="نص الدفع كما أبداه المحامي أو كما يستخلصه التحليل")

    # ── الأساس القانوني والتقدير ──────────────────────────────────
    legal_basis: Optional[str] = Field(default=None, description="النص القانوني أو المبدأ القضائي الذي يستند إليه الدفع")
    strength:    Optional[str] = Field(default=None, description="قوة الدفع: قوي / متوسط / ضعيف")


class JudicialPrinciple(BaseModel):
    # ── مضمون المبدأ ──────────────────────────────────────────────
    principle_text: Optional[str] = Field(default=None, description="نص المبدأ القضائي كما وَرد في حكم المحكمة")
    court_source:   Optional[str] = Field(default=None, description="مصدر المبدأ: اسم المحكمة والدائرة وتاريخ الطعن")

    # ── نطاق التطبيق ──────────────────────────────────────────────
    applicable_to: Optional[str] = Field(default=None, description="الوقائع أو المسائل القانونية التي ينطبق عليها هذا المبدأ في القضية الراهنة")


class ProceduralIssue(BaseModel):
    # ── مصدر الإجراء وطبيعته ─────────────────────────────────────
    procedure_source:   Optional[str] = Field(default=None, description="المستند أو المحضر الذي كشف عن الإشكالية الإجرائية")
    procedure_type:     Optional[str] = Field(default=None, description="نوع الإجراء محل الفحص: ضبط / قبض / استجواب / تفتيش")
    issue_description:  Optional[str] = Field(default=None, description="وصف الإشكالية الإجرائية بدقة")

    # ── ضمانات الإجراء ────────────────────────────────────────────
    warrant_present:    bool          = Field(default=False,        description="هل صدر إذن قضائي أو نيابي يُجيز الإجراء؟")
    conducting_officer: Optional[str] = Field(default=None,         description="اسم ووظيفة من أجرى الإجراء")

    # ── البطلان وأثره ─────────────────────────────────────────────
    nullity_type:   Optional[str] = Field(default=None, description="نوع البطلان: بطلان مطلق / بطلان نسبي")
    nullity_effect: Optional[str] = Field(default=None, description="أثر البطلان على الأدلة المترتبة عليه")
    article_basis:  Optional[str] = Field(default=None, description="المادة القانونية الموجِبة للبطلان")

    # ── المرجع الوثائقي ───────────────────────────────────────────
    source_document_id: Optional[str] = Field(default=None, description="معرّف المستند المصدر في ملف القضية")


class ExcludedDefenseClaim(BaseModel):
    # ── الدفع المستبعد ────────────────────────────────────────────
    claim:  Optional[str] = Field(default=None, description="ملخص الدفع الإجرائي الذي أثاره المحامي")
    reason: Optional[str] = Field(default=None, description="سبب استبعاده: عدم صحة قانونية / ثبوت خلافه من الأوراق / سبق الفصل فيه")


class ProceduralAuditResult(BaseModel):
    # ── مخرجات المراجعة ───────────────────────────────────────────
    violations:              List[ProceduralIssue]      = Field(default_factory=list, description="قائمة الانتهاكات الإجرائية المرصودة في ملف القضية")
    excluded_defense_claims: List[ExcludedDefenseClaim] = Field(default_factory=list, description="الدفوع الإجرائية التي أثارها الدفاع وجرى استبعادها مع مسوّغ كل منها")

    # ── التقييم الإجمالي ──────────────────────────────────────────
    overall_assessment: Optional[str] = Field(default=None, description="الخلاصة الكلية لسلامة الإجراءات أو اعتلالها")
    critical_nullities: List[str]     = Field(default_factory=list, description="البطلانات الجوهرية التي قد تُسقط أدلة محورية أو تُوجب إعادة التحقيق")

    # ── المراجع القانونية المستخدمة ───────────────────────────────
    kg_articles_used: List[str] = Field(default_factory=list, description="المواد القانونية المستند إليها في تقييم صحة الإجراءات")

    # ── بيانات تشغيلية ────────────────────────────────────────────
    meta: dict = Field(default_factory=dict, description="بيانات وصفية تشغيلية: الوكيل المُنفِّذ، الإصدار، الطابع الزمني")


class FinalJudgment(BaseModel):
    # ── منطوق الحكم ───────────────────────────────────────────────
    verdict:             Optional[str] = Field(default=None, description="منطوق الحكم: إدانة كاملة / إدانة جزئية / براءة")
    recommended_penalty: Optional[str] = Field(default=None, description="العقوبة المقترحة محددةً بنوعها ومدتها وفق النصوص المنطبقة")

    # ── التسبيب والتفصيل ──────────────────────────────────────────
    reasoning_summary:  Optional[str] = Field(default=None, description="ملخص أسباب الحكم مستوعِباً الأدلة والدفوع وتقدير المحكمة")
    per_charge_rulings: List[dict]    = Field(
        default_factory=list,
        description=(
            "الحكم مفصَّلاً بحسب كل تهمة. "
            "كل عنصر: {charge_description, verdict, penalty, reasoning}"
        )
    )
    full_ruling_text: Optional[str] = Field(default=None, description="نص الحكم كاملاً بصياغته الرسمية كما يُنطق في الجلسة")

    # ── مؤشر الثقة ────────────────────────────────────────────────
    confidence_score: float = Field(default=0.0, ge=0, le=1, description="درجة ثقة النموذج في توصية الحكم على مقياس 0-1")