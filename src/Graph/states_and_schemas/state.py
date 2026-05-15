from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

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
    ProceduralIssue,
    ProsecutionNarrative,
    ProsecutionArgument,
    ConfessionAssessment,
    WitnessCredibility,
    CivilClaim,
)


class AgentState(BaseModel):

    # ── Case Meta ─────────────────────────────────────  (DataIngestionAgent)
    case_id:             Optional[str] = None
    case_number:         Optional[str] = None
    court:               Optional[str] = None
    court_level:         Optional[str] = None
    jurisdiction:        Optional[str] = None
    filing_date:         Optional[str] = None
    referral_date:       Optional[str] = None
    prosecutor_name:     Optional[str] = None
    referral_order_text: Optional[str] = Field(
        default=None,
        description="نص أمر الإحالة الحاكم لنطاق القضية",
    )

    # ── Extracted Entities ────────────────────────────  (DataIngestionAgent)
    defendants:           List[Defendant]           = Field(default_factory=list)
    charges:              List[Charge]              = Field(default_factory=list)
    incidents:            List[CaseIncident]        = Field(default_factory=list)
    evidences:            List[Evidence]            = Field(default_factory=list)
    witness_statements:   List[WitnessStatement]    = Field(default_factory=list)
    confessions:          List[Confession]          = Field(default_factory=list)
    lab_reports:          List[LabReport]           = Field(default_factory=list)
    criminal_records:     List[CriminalRecord]      = Field(default_factory=list)
    criminal_proceedings: List[CriminalProceedings] = Field(default_factory=list)
    defense_documents:    List[DefenseDocument]     = Field(default_factory=list)

    # ── Procedural Analysis ───────────────────────────  (ProceduralAuditorAgent)
    procedural_issues:         List[ProceduralIssue] = Field(default_factory=list)

    # ── Evidence Analysis ─────────────────────────────  (EvidenceAnalystAgent)
    evidence_scores: List[EvidenceScoring] = Field(default_factory=list)
    chain_of_custody_issues: List[str]     = Field(
        default_factory=list,
        description="مشكلات سلسلة حيازة الأدلة",
    )

    # ── Legal Research ────────────────────────────────  (LegalResearcherAgent)
    applied_principles: List[JudicialPrinciple] = Field(default_factory=list)
    case_articles:      List[str]               = Field(default_factory=list)

    # ── Confession Validity ───────────────────────────  (ConfessionValidityAgent)
    confession_assessments: List[ConfessionAssessment] = Field(default_factory=list)

    # ── Witness Credibility ───────────────────────────  (WitnessCredibilityAgent)
    witness_credibility_scores: List[WitnessCredibility] = Field(default_factory=list)

    # ── Prosecution Analysis ──────────────────────────  (ProsecutionAnalystAgent)
    prosecution_theory:    Optional[ProsecutionNarrative] = None
    prosecution_arguments: List[ProsecutionArgument]      = Field(default_factory=list)

    # ── Defense Analysis ──────────────────────────────  (DefenseAnalystAgent)
    defense_arguments: List[DefenseArgument] = Field(default_factory=list)

    # ── Sentencing ────────────────────────────────────  (SentencingAgent) ✦ جديد
    aggravating_factors:   List[str]                     = Field(default_factory=list)
    mitigating_factors:    List[str]                     = Field(default_factory=list)
    applicable_article_17: bool                          = Field(
        default=False,
        description="هل تنطبق المادة 17 عقوبات للتخفيف التقديري؟",
    )
    charge_conviction_map: Dict[str, Optional[str]]      = Field(
        default_factory=dict,
        description="وصف التهمة → 'إدانة' / 'براءة' / None إذا لم يُحسم",
    )
    civil_claim:           Optional[CivilClaim]          = None
    civil_award_suggested: Optional[str]                 = Field(
        default=None,
        description="مبلغ التعويض المدني المقترح بالجنيه (نص)",
    )

    # ── Final Verdict ─────────────────────────────────  (JudgeAgent)
    suggested_verdict: Optional[FinalJudgment] = None

    # ── Structured Verdict Sections ───────────────────  (JudgeAgent)
    verdict_preamble:  Optional[str] = Field(
        default=None,
        description="البيانات الأولية للحكم (المحكمة / التاريخ / الأطراف)",
    )
    verdict_facts:     Optional[str] = Field(
        default=None,
        description="وقائع القضية كما استخلصتها المحكمة",
    )
    verdict_reasoning: Optional[str] = Field(
        default=None,
        description="أسباب الحكم التفصيلية",
    )
    verdict_operative: Optional[str] = Field(
        default=None,
        description="منطوق الحكم",
    )

    # ── Pipeline Status ───────────────────────────────
    completed_agents: List[str]      = Field(default_factory=list)
    current_agent:    Optional[str]  = None
    errors:           List[str]      = Field(default_factory=list)

    # ── Metadata ──────────────────────────────────────
    source_documents: List[str]          = Field(default_factory=list)
    last_updated:     Optional[datetime] = Field(default_factory=datetime.now)
    extraction_notes: Optional[str]      = None

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="ignore")

    def get(self, key: str, default=None):
        try:
            return self.model_dump().get(key, default)
        except Exception:
            return default