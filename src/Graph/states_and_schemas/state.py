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
    case_number:         Optional[str] = None
    court:               Optional[str] = None
    court_level:         Optional[str] = None
    jurisdiction:        Optional[str] = None
    filing_date:         Optional[str] = None
    referral_date:       Optional[str] = None
    prosecutor_name:     Optional[str] = None
    referral_order_text: Optional[str] = Field(default=None)

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

    # ── Procedural Analysis ───────────────────────────  (ProceduralAuditorAgent — sequential)
    procedural_audit: ProceduralAuditResult = Field(default_factory=ProceduralAuditResult)

    # ── Evidence Analysis ─────────────────────────────  (EvidenceAnalystAgent — parallel)
    evidence_scores:          Annotated[List[EvidenceScoring], add] = Field(default_factory=list)
    chain_of_custody_issues:  Annotated[List[str], add]             = Field(default_factory=list)

    # ── Legal Research ────────────────────────────────  (LegalResearcherAgent — sequential)
    applied_principles: List[JudicialPrinciple] = Field(default_factory=list)
    case_articles:      List[dict]              = Field(default_factory=list)

    # ── Confession Validity ───────────────────────────  (ConfessionValidityAgent — parallel)
    confession_assessments: Annotated[List[ConfessionAssessment], add] = Field(default_factory=list)

    # ── Witness Credibility ───────────────────────────  (WitnessCredibilityAgent — parallel)
    witness_credibility_scores: Annotated[List[WitnessCredibility], add] = Field(default_factory=list)

    # ── Prosecution Analysis ──────────────────────────  (ProsecutionAnalystAgent — sequential)
    prosecution_theory:    Optional[ProsecutionNarrative]  = None
    prosecution_arguments: List[ProsecutionArgument]       = Field(default_factory=list)

    # ── Defense Analysis ──────────────────────────────  (DefenseAnalystAgent — sequential)
    defense_arguments: List[DefenseArgument] = Field(default_factory=list)

    # ── Sentencing ────────────────────────────────────  (SentencingAgent — sequential)
    aggravating_factors:   List[str]                = Field(default_factory=list)
    mitigating_factors:    List[str]                = Field(default_factory=list)

    @field_validator("aggravating_factors", "mitigating_factors", mode="before")
    @classmethod
    def coerce_factors(cls, v):
        if not isinstance(v, list):
            return v
        return [
            item.get("description", str(item)) if isinstance(item, dict) else item
            for item in v
        ]

    applicable_article_17: bool                     = Field(default=False)
    charge_conviction_map: Dict[str, Optional[str]] = Field(default_factory=dict)
    civil_claim:           Optional[CivilClaim]     = None
    civil_award_suggested: Optional[str]            = None

    # ── Final Verdict ─────────────────────────────────  (JudgeAgent — sequential)
    suggested_verdict: Optional[FinalJudgment] = None
    verdict_preamble:  Optional[str]           = None
    verdict_facts:     Optional[str]           = None
    verdict_reasoning: Optional[str]           = None
    verdict_operative: Optional[str]           = None

    # ── Pipeline Status ───────────────────────────────
    # Both written by every agent in every step → must use add reducer
    completed_agents: Annotated[List[str], add] = Field(default_factory=list)
    errors:           Annotated[List[str], add] = Field(default_factory=list)
    current_agent:    Optional[str]             = None
    last_updated:     Optional[datetime]        = Field(default_factory=datetime.now)

    # ── Metadata ──────────────────────────────────────
    source_documents: List[str]          = Field(default_factory=list)
    extraction_notes: Optional[str]      = None

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="ignore")

    def get(self, key: str, default=None):
        try:
            return self.model_dump().get(key, default)
        except Exception:
            return default