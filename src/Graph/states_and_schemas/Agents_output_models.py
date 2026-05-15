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
    """
    النظرية الاتهامية المتكاملة كما تبنّتها النيابة العامة.
    يُنشئها ProsecutionAnalystAgent ويُخزَّن في AgentState.prosecution_theory.
    """
 
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
    """
    حجة اتهامية واحدة مرتبطة بتهمة بعينها.
    تُخزَّن في AgentState.prosecution_arguments  (List[ProsecutionArgument]).
    """
 
    charge_description:      str                         = Field(description="وصف التهمة التي تخصّها هذه الحجة")
    article_reference:       Optional[str]               = Field(default=None,description="المادة القانونية المستند إليها (مثال: المادة 234 عقوبات)")
    argument_text:           str                         = Field(description="نص الحجة الاتهامية كما سيُعرض على القاضي")
    supporting_evidence_ids: List[str]                   = Field(default_factory=list,description="معرّفات الأدلة الداعمة لهذه الحجة من قائمة evidences")
    strength:                ProsecutionArgumentStrength = Field(description="قوة هذه الحجة بالذات")
    rebuttal_risk:           Optional[str]               = Field(default=None,description="أبرز ما يمكن أن يرد به الدفاع على هذه الحجة")
 

class ConfessionAssessment(BaseModel):
    """
    تقييم مشروعية وحجية اعتراف متهم بعينه.
    يُنشئها ConfessionValidityAgent وتُخزَّن في AgentState.confession_assessments.
 
    المرجع القانوني الرئيسي:
        - المادة 302 إجراءات جنائية (لا يجوز الحكم بالإدانة بناءً على الاعتراف وحده)
        - أحكام النقض في ضمانات صحة الاعتراف
    """
 
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
    """
    تقييم موثوقية شاهد بعينه.
    يُنشئها WitnessCredibilityAgent وتُخزَّن في AgentState.witness_credibility_scores.
    """
 
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
    """
    الدعوى المدنية التبعية المرفوعة بجانب الدعوى الجنائية.
    تُخزَّن في AgentState.civil_claim.
 
    الأساس القانوني: المواد 251-255 إجراءات جنائية مصري
    """
 
    # ── بيانات المدّعي المدني ─────────────────────────────────────
    plaintiff_name:     str           = Field(description="اسم المدّعي المدني (المجني عليه أو ورثته)")
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


class ProceduralIssue(BaseModel):
    """دفع إجرائي / بطلان"""
    procedure_source:   Optional[str] = None
    procedure_type:     Optional[str] = None   # ضبط / قبض / استجواب / تفتيش
    issue_description:  Optional[str] = None
    warrant_present:    bool          = False
    conducting_officer: Optional[str] = None
    nullity_type:       Optional[str] = None   # بطلان مطلق / بطلان نسبي
    article_basis:      Optional[str] = None
    source_document_id: Optional[str] = None


class FinalJudgment(BaseModel):
    verdict:             Optional[str] = None   # إدانة / براءة / جزئية
    reasoning_summary:   Optional[str] = None
    recommended_penalty: Optional[str] = None
    confidence_score:    float         = Field(default=0.0, ge=0, le=1)