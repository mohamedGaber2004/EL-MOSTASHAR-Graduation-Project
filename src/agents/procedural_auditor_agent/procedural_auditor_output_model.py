from typing import Optional, List, Any
from pydantic import BaseModel, Field, field_validator
from src.agents.procedural_auditor_agent.PA_enums import PipelineFatalityLevel

class ProceduralIssue(BaseModel):

    # ── مصدر الإجراء وطبيعته ──────────────────────────────────────────────────
    procedure_source:   Optional[str] = Field(
        default=None,
        description="المستند أو المحضر الذي كشف عن الإشكالية الإجرائية"
    )
    procedure_type:     Optional[str] = Field(
        default=None,
        description="نوع الإجراء: ضبط / قبض / استجواب / تفتيش / تحقيق / إخطار / تكليف بالحضور"
    )
    issue_description:  Optional[str] = Field(
        default=None,
        description="وصف دقيق للمسألة الإجرائية والسبب الذي جعلها مخالفة"
    )

    # ── ضمانات الإجراء ────────────────────────────────────────────────────────
    authorized_officer: Optional[bool] = Field(
        default=None,
        description=(
            "هل صدر الإجراء من شخص مخوّل قانوناً؟ "
            "(مأمور ضبط قضائي / نيابة / قاضٍ حسب طبيعة الإجراء) "
            "true=مخوّل | false=غير مخوّل | null=غير محدد من الأوراق. "
            "ملاحظة: التلبس يُجيز القبض والتفتيش بلا إذن مسبق — لا يعني غياب الإذن بطلاناً تلقائياً."
        )
    )
    warrant_present:    Optional[bool] = Field(
        default=None,
        description=(
            "هل صدر إذن قضائي أو نيابي مسبق للإجراء؟ "
            "true=صدر إذن صريح | false=لم يصدر | null=غير محدد. "
            "غيابه في حالة التلبس لا يُولِّد بطلاناً."
        )
    )
    conducting_officer: Optional[str] = Field(
        default=None,
        description="اسم ووظيفة من أجرى الإجراء"
    )

    # ── تصنيف الجزاء الإجرائي ────────────────────────────────────────────────
    # المقال يوجب التمييز بين بطلان / انعدام / سقوط / عدم القبول
    sanction_type:      Optional[str] = Field(
        default=None,
        description=(
            "نوع الجزاء الإجرائي الصحيح وفق التوصيف القانوني الدقيق: "
            "بطلان مطلق / بطلان نسبي / انعدام / سقوط / عدم القبول. "
            "لا تستخدم 'بطلان' إذا كان الأصح 'انعدام' أو 'سقوط'."
        )
    )

    # ── اختبار الجوهرية ───────────────────────────────────────────────────────
    # الاختبار المزدوج: (1) القاعدة جوهرية؟ (2) أثّر فعلاً في حق الدفاع؟
    rule_is_substantive:   Optional[bool] = Field(
        default=None,
        description=(
            "الاختبار الأول: هل القاعدة المخالَفة جوهرية (تحمي مصلحة أساسية) "
            "لا إرشادية؟ "
            "true=جوهرية | false=إرشادية فقط (لا بطلان). "
            "القواعد الإرشادية لا تُولِّد بطلاناً ولو خُولفت."
        )
    )
    actual_harm_to_defense: Optional[bool] = Field(
        default=None,
        description=(
            "الاختبار الثاني: هل أثّرت المخالفة فعلاً في حق الدفاع أو سلامة المحاكمة؟ "
            "true=ضرر فعلي ثابت | false=لا ضرر فعلي (بطلان غير منتج). "
            "كلا الاختبارين يجب أن يكون true لإعلان البطلان."
        )
    )
    harm_description:       Optional[str] = Field(
        default=None,
        description=(
            "تحديد طبيعة الضرر الفعلي الذي لحق بحق الدفاع أو بسلامة المحاكمة. "
            "null إذا لم يثبت ضرر فعلي."
        )
    )

    # ── البطلان وأثره التتابعي ────────────────────────────────────────────────
    # المقال: الأثر لا ينسحب تلقائياً — يجب تحديد كل طبقة على حدة
    nullity_type:             Optional[str] = Field(
        default=None,
        description=(
            "نوع البطلان (إذا كان sanction_type = بطلان): "
            "بطلان مطلق: يتعلق بالنظام العام / تُقضي به المحكمة من تلقاء نفسها / لا يزول بالإجازة. "
            "بطلان نسبي: يحمي مصلحة خاصة / يتمسك به صاحب المصلحة فقط / يسقط بالتأخير أو الإجازة الضمنية."
        )
    )
    effect_on_void_act:       Optional[str] = Field(
        default=None,
        description="أثر البطلان على الإجراء الباطل ذاته — ماذا يفقد من قيمته وأثره المباشر."
    )
    effect_on_prior_acts:     Optional[str] = Field(
        default=None,
        description=(
            "أثر البطلان على الإجراءات السابقة عليه. "
            "الأصل: لا يسري عليها إذا كانت مستقلة وصحيحة."
        )
    )
    effect_on_subsequent_acts: Optional[str] = Field(
        default=None,
        description=(
            "أثر البطلان على الإجراءات اللاحقة. "
            "يمتد إليها فقط إذا كانت قائمة مباشرة على الإجراء الباطل ولا يمكن فصلها عنه. "
            "إذا كان لها أساس مستقل — تبقى صحيحة."
        )
    )
    article_basis:             Optional[str] = Field(
        default=None,
        description="المادة القانونية المنتهكة ومصدرها (دستور / قانون الإجراءات الجنائية / غيره)"
    )

    # ── مسار الإصلاح (م. 335 + م. 334) ──────────────────────────────────────
    purpose_achieved:   Optional[bool] = Field(
        default=None,
        description=(
            "استثناء م. 334: هل تحققت الغاية من الإجراء رغم العيب الشكلي؟ "
            "true → قد لا يُقضى بالبطلان. "
            "null إذا لم ينطبق الاستثناء على هذا النوع من الإجراءات."
        )
    )
    is_correctable:     Optional[bool] = Field(
        default=None,
        description=(
            "م. 335: هل يمكن تصحيح هذا الإجراء أو إعادته؟ "
            "true=قابل للتصحيح (يُقترح التصحيح لا الإلغاء) | "
            "false=غير قابل (العيب يمس أصل الإجراء أو مشروعيته) | "
            "null=غير محدد."
        )
    )
    correction_path:    Optional[str] = Field(
        default=None,
        description=(
            "إذا كان is_correctable=true: وصف طريق التصحيح الممكن "
            "(إعادة الإجراء / استكمال بيان جوهري / إعادة إعلان صحيح…). "
            "null إذا كان غير قابل للتصحيح."
        )
    )

    # ── المرجع الوثائقي ───────────────────────────────────────────────────────
    source_document_id: Optional[str] = Field(
        default=None,
        description="معرّف المستند المصدر في ملف القضية"
    )

    # ── validators ────────────────────────────────────────────────────────────
    @field_validator("authorized_officer", "warrant_present",
                     "rule_is_substantive", "actual_harm_to_defense",
                     "purpose_achieved", "is_correctable", mode="before")
    @classmethod
    def coerce_optional_bool(cls, v: Any) -> Optional[bool]:
        if v is None:
            return None
        if isinstance(v, bool):
            return v
        if isinstance(v, int):
            return bool(v)
        if isinstance(v, str):
            s = v.strip().lower()
            if s in {"true", "yes", "1", "نعم", "صحيح"}:
                return True
            if s in {"false", "no", "0", "لا", "خطأ"}:
                return False
            return None
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  ExcludedDefenseClaim
#  Defense procedural claims that were reviewed and rejected.
#  The article requires these to be LOGGED (not silently dropped) with reason.
# ══════════════════════════════════════════════════════════════════════════════
class ExcludedDefenseClaim(BaseModel):
    claim:            Optional[str] = Field(
        default=None,
        description="ملخص الدفع الإجرائي الذي أثاره المحامي"
    )
    rejection_reason: Optional[str] = Field(
        default=None,
        description=(
            "سبب الرفض وفق التصنيف الدقيق: "
            "لا سند واقعي (الوقائع تثبت خلافه) / "
            "لا سند قانوني (القاعدة إرشادية لا جوهرية) / "
            "الغاية تحققت (م. 334) / "
            "قابل للتصحيح لا للإبطال (م. 335) / "
            "سقط بالتأخير أو الإجازة الضمنية (بطلان نسبي فقط) / "
            "خلط بين بطلان وانعدام أو سقوط (تكييف خاطئ)"
        )
    )
    correct_sanction: Optional[str] = Field(
        default=None,
        description=(
            "التكييف القانوني الصحيح للمسألة إن كان الدفاع أخطأ في التوصيف: "
            "بطلان نسبي (لا مطلق) / انعدام / سقوط / عدم قبول / لا جزاء إجرائي. "
            "null إذا لم يكن ثمة عيب أصلاً."
        )
    )


# ══════════════════════════════════════════════════════════════════════════════
#  ProceduralAuditResult  —  top-level output of the agent
# ══════════════════════════════════════════════════════════════════════════════
class ProceduralAuditResult(BaseModel):

    # ── مخرجات المراجعة ───────────────────────────────────────────────────────
    violations: List[ProceduralIssue] = Field(
        default_factory=list,
        description="قائمة الانتهاكات الإجرائية المرصودة — كل منها اجتاز الاختبار المزدوج (جوهرية + ضرر فعلي)"
    )
    excluded_defense_claims: List[ExcludedDefenseClaim] = Field(
        default_factory=list,
        description=(
            "الدفوع الإجرائية التي أثارها الدفاع وجرى استبعادها مع مسوّغ كل منها. "
            "يجب تسجيل كل دفع مستبعد هنا — لا يحذف صمتاً."
        )
    )

    # ── التقييم الإجمالي ──────────────────────────────────────────────────────
    overall_assessment: Optional[str] = Field(
        default=None,
        description="الخلاصة الكلية لسلامة الإجراءات أو اعتلالها في ضوء الاختبار المزدوج"
    )
    critical_nullities: List[str] = Field(
        default_factory=list,
        description=(
            "البطلانات المطلقة فقط (النظام العام) التي قد تُسقط أدلة محورية "
            "أو توجب إعادة التحقيق — لا تُدرج هنا البطلانات النسبية"
        )
    )
    correctable_issues: List[str] = Field(
        default_factory=list,
        description=(
            "المخالفات القابلة للتصحيح (م. 335) التي تستوجب التصحيح لا الإلغاء — "
            "يُقترح على المحكمة إصدار أمر بالتصحيح بدلاً من إسقاط الإجراء"
        )
    )

    # ── قرار الـ routing ──────────────────────────────────────────────────────
    pipeline_fatality: PipelineFatalityLevel = Field(
        default=PipelineFatalityLevel.NONE,
        description=(
            "تقييم أثر المخالفات على استمرار الـ pipeline: "
            "none=كمّل عادي | "
            "partial=استبعد دليل وكمّل | "
            "fatal=الدعوى ساقطة أو المحكمة غير مختصة — أرسل للقاضي مباشرة. "
            "fatal فقط في 4 حالات: تقادم / وفاة متهم / عدم اختصاص نوعي / حكم بات سابق."
        )
    )
    fatality_reason: Optional[str] = Field(
        default=None,
        description="سبب الـ fatal إن وُجد — null إذا كان none أو partial."
    )

    # ── المراجع القانونية المستخدمة ───────────────────────────────────────────
    kg_articles_used: List[str] = Field(
        default_factory=list,
        description="المواد القانونية التي استُند إليها فعلاً في التقييم (من السياق المُعطى فقط)"
    )

    # ── بيانات تشغيلية ────────────────────────────────────────────────────────
    meta: dict = Field(
        default_factory=dict,
        description="بيانات وصفية تشغيلية: الوكيل المُنفِّذ، الإصدار، الطابع الزمني"
    )

    @field_validator("pipeline_fatality", mode="before")
    @classmethod
    def coerce_fatality(cls, v: Any) -> PipelineFatalityLevel:
        if isinstance(v, PipelineFatalityLevel):
            return v
        if isinstance(v, str):
            try:
                return PipelineFatalityLevel(v.lower().strip())
            except ValueError:
                return PipelineFatalityLevel.NONE
        return PipelineFatalityLevel.NONE