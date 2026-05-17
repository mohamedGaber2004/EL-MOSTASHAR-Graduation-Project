from typing import Optional, List
from pydantic import BaseModel, Field


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