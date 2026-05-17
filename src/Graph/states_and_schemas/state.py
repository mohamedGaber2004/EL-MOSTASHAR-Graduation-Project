from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional, Annotated
from operator import add
from pydantic import BaseModel, ConfigDict, Field
from pydantic import field_validator

from src.Graph.states_and_schemas.main_entity_classes import (
    Charge,
    CaseIncident,
    Confession,
    CriminalRecord,
    DefenseDocument,
    Defendant,
    Evidence,
    LabReport,
    WitnessStatement,
    CriminalProceedings,
)

from src.Graph.states_and_schemas.Agents_output_models import (
    DefenseArgument,
    EvidenceScoring,
    FinalJudgment,
    JudicialPrinciple,
    ProceduralAuditResult,
    ProsecutionNarrative,
    ProsecutionArgument,
    ConfessionAssessment,
    WitnessCredibility,
    CivilClaim,
)


class AgentState(BaseModel):

    # ── Case Meta ─────────────────────────────────────  (DataIngestionAgent)
    case_id:             Optional[str] = None 
    case_number:         Optional[str] = None # رقم القضية كاملاً كما ورد
    court:               Optional[str] = None # اسم المحكمة المُحال إليها كاملاً.
    court_level:         Optional[str] = None # جنح / جنايات / ابتدائي / استئناف / نقض.
    jurisdiction:        Optional[str] = None # الدائرة أو النيابة المُحيلة
    filing_date:         Optional[str] = None # تاريخ تحرير المحضر الأصلي إن وُجد.
    referral_date:       Optional[str] = None # تاريخ صدور أمر الإحالة.
    prosecutor_name:     Optional[str] = None # اسم ممثل النيابة العامة / وكيل النيابة الموقِّع.
    referral_order_text: Optional[str] = Field(default=None) # نص امر الاحاله مختصر

    # ── Extracted Entities ────────────────────────────  (DataIngestionAgent)
    defendants:           List[Defendant]           = Field(default_factory=list) # كل متهم ذُكر في محضر الضبط
    charges:              List[Charge]              = Field(default_factory=list) # تهمة مستقلة لكل مادة قانونية
    incidents:            List[CaseIncident]        = Field(default_factory=list) # وقائع الحادثة الموثقة في المحضر
    evidences:            List[Evidence]            = Field(default_factory=list) # كل حرز أو مضبوطات تم ضبطها
    witness_statements:   List[WitnessStatement]    = Field(default_factory=list) # الشهود (الاوليين + محضر الاستجواب)
    confessions:          List[Confession]          = Field(default_factory=list) # موقف كل متهم في الاستجواب (اعتراف أو إنكار)
    lab_reports:          List[LabReport]           = Field(default_factory=list) # سجل مستقل لكل تقرير أو فحص
    criminal_records:     List[CriminalRecord]      = Field(default_factory=list) # سجل مستقل لكل متهم
    criminal_proceedings: List[CriminalProceedings] = Field(default_factory=list) # كل الإجراءات الواردة في المحضر
    defense_documents:    List[DefenseDocument]     = Field(default_factory=list) # دفوع المحامي

    # ── Procedural Analysis ───────────────────────────  (ProceduralAuditorAgent — sequential)
    procedural_audit: ProceduralAuditResult = Field(default_factory=ProceduralAuditResult) # مشاكل الاجراءات الجنائيه (في محضر الضبظ+المثاره من المحامي)

    # ── Evidence Analysis ─────────────────────────────  (EvidenceAnalystAgent — parallel)
    evidence_scores:          Annotated[List[EvidenceScoring], add] = Field(default_factory=list) # قائمة بتقييم كل دليل على حدة
    chain_of_custody_issues:  Annotated[List[str], add]             = Field(default_factory=list) # قائمة بمشاكل سلسلة الحيازة، يعني إيه الأدلة اللي فيها خلل في طريقة ضبطها أو تداولها من لحظة الضبط 

    # ── Legal Research ────────────────────────────────  (LegalResearcherAgent — sequential)
    applied_principles: List[JudicialPrinciple] = Field(default_factory=list) # مبادئ محكمة النقض المنطبقه على القضيه
    case_articles:      List[dict]              = Field(default_factory=list) # المواد القانونية المرتبطة بالقضية

    # ── Confession Validity ───────────────────────────  (ConfessionValidityAgent — parallel)
    confession_assessments: Annotated[List[ConfessionAssessment], add] = Field(default_factory=list) # تقييم كل اعتراف في القضية

    # ── Witness Credibility ───────────────────────────  (WitnessCredibilityAgent — parallel)
    witness_credibility_scores: Annotated[List[WitnessCredibility], add] = Field(default_factory=list) # تقييم مصداقية كل شاهد

    # ── Prosecution Analysis ──────────────────────────  (ProsecutionAnalystAgent — sequential)
    prosecution_theory:    Optional[ProsecutionNarrative]  = None # نظرية الاتهام الكاملة
    prosecution_arguments: List[ProsecutionArgument]       = Field(default_factory=list) # قائمة بالحجج الاتهامية، كل حجة مرتبطة بتهمة معينة و معها الأدلة الداعمة ودرجة قوتها وما قد يرد به الدفاع.

    # ── Defense Analysis ──────────────────────────────  (DefenseAnalystAgent — sequential)
    defense_arguments: List[DefenseArgument] = Field(default_factory=list) # قائمة بحجج الدفاع المستخرجة من مذكرات الدفاع، كل حجة مصنفة (شكلية / موضوعية) مع تقييم قوتها وما يدحضها من أدلة.

    # ── Sentencing ────────────────────────────────────  (SentencingAgent — sequential)
    aggravating_factors:   List[str]                = Field(default_factory=list) # الظروف المشددة للعقوبة
    mitigating_factors:    List[str]                = Field(default_factory=list) # الظروف المخففة
    applicable_article_17: bool                     = Field(default=False) # هل تنطبق المادة 17 عقوبات؟ دي المادة اللي بتخلي القاضي يخفف العقوبة لما تكون الظروف تستدعي ده.
    charge_conviction_map: Dict[str, Optional[str]] = Field(default_factory=dict) # خريطة بتقول لكل تهمة هل النتيجة إدانة أو براءة، ده بيتعمل في مرحلة التقدير قبل ما القاضي يُصدر حكمه النهائي.
    civil_claim:           Optional[CivilClaim]     = None # بيانات الدعوى المدنية المضمومة للجنائية، بتشمل المدعي والمبلغ المطلوب والحالة.
    civil_award_suggested: Optional[str]            = None # المبلغ المقترح للتعويض المدني اللي وصل ليه secntencing agent .

    # ── Final Verdict ─────────────────────────────────  (JudgeAgent — sequential)
    suggested_verdict: Optional[FinalJudgment] = None # الحكم النهائي الصادر عن القاضي
    verdict_preamble:  Optional[str]           = None # ديباجة الحكم، بتتكلم عن المحكمة والدائرة وتاريخ الجلسة ورقم القضية.
    verdict_facts:     Optional[str]           = None # الوقائع الثابتة اللي اطمأنت إليها المحكمة.
    verdict_reasoning: Optional[str]           = None # الأسباب القانونية والواقعية التفصيلية للحكم، ده جوهر التسبيب.
    verdict_operative: Optional[str]           = None # منطوق الحكم، يعني الجزء اللي بيُنطق به في قاعة المحكمة:

    # ── Pipeline Status ───────────────────────────────
    # Both written by every agent in every step → must use add reducer
    completed_agents: Annotated[List[str], add] = Field(default_factory=list) #
    errors:           Annotated[List[str], add] = Field(default_factory=list) #
    current_agent:    Optional[str]             = None #
    last_updated:     Optional[datetime]        = Field(default_factory=datetime.now) #

    # ── Metadata ──────────────────────────────────────
    source_documents: List[str]          = Field(default_factory=list) #
    extraction_notes: Optional[str]      = None #

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="ignore")

    # ── Field's Validations ───────────────────────────────
    @field_validator("aggravating_factors", "mitigating_factors", mode="before")
    @classmethod
    def coerce_factors(cls, v):
        if not isinstance(v, list):
            return v
        return [
            item.get("description", str(item)) if isinstance(item, dict) else item
            for item in v
        ]

    def get(self, key: str, default=None):
        try:
            return self.model_dump().get(key, default)
        except Exception:
            return default