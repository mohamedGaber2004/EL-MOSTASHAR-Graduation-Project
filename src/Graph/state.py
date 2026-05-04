from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field

from src.Utils import (
    Charge, CaseIncident, CourtLevel, Confession, DefenseArgument,
    DefenseDocument, Defendant, Evidence, EvidenceAnalysis, FinalJudgment,
    JudicialPrinciple, LabReport, PriorJudgment, ProceduralIssue,
    WitnessStatement,
)


class AgentState(BaseModel):

    # ── Case Meta ─────────────────────────────────────
    case_id:                str
    case_number:            Optional[str]        = None
    court:                  Optional[str]        = None
    court_level:            Optional[CourtLevel] = None
    jurisdiction:           Optional[str]        = None
    filing_date:            Optional[str]        = None
    referral_date:          Optional[str]        = None
    prosecutor_name:        Optional[str]        = None
    referral_order_text:    Optional[str]        = Field(
        default=None,
        description="نص أمر الإحالة الحاكم لنطاق القضية"
    )

    # ── Extracted Entities ────────────────────────────
    defendants:             List[Defendant]         = Field(default_factory=list)
    charges:                List[Charge]            = Field(default_factory=list)
    incidents:              List[CaseIncident]      = Field(default_factory=list)
    evidences:              List[Evidence]          = Field(default_factory=list)
    lab_reports:            List[LabReport]         = Field(default_factory=list)
    witness_statements:     List[WitnessStatement]  = Field(default_factory=list)
    confessions:            List[Confession]        = Field(default_factory=list)
    procedural_issues:      List[ProceduralIssue]   = Field(default_factory=list)
    prior_judgments:        List[PriorJudgment]     = Field(default_factory=list)
    defense_documents:      List[DefenseDocument]   = Field(default_factory=list)

    # ── Structured Analysis Outputs ───────────────────
    evidence_analysis:      List[EvidenceAnalysis]  = Field(default_factory=list)
    defense_arguments:      List[DefenseArgument]   = Field(default_factory=list)
    applied_principles:     List[JudicialPrinciple] = Field(default_factory=list)
    legal_articles:         List[str]               = Field(default_factory=list)

    # ── Status ────────────────────────────────────────
    completed_agents:       List[str]               = Field(default_factory=list)
    current_agent:          Optional[str]           = None
    errors:                 List[str]               = Field(default_factory=list)
    confidence_score:       float                   = Field(default=0.0, ge=0, le=1)

    # ── Final Verdict ─────────────────────────────────
    suggested_verdict:      Optional[FinalJudgment] = None

    # ── Metadata ──────────────────────────────────────
    source_documents:       List[str]               = Field(default_factory=list)
    last_updated:           Optional[datetime]      = Field(default_factory=datetime.now)
    extraction_notes:       Optional[str]           = None

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="ignore")

    def get(self, key: str, default=None):
        try:
            return self.model_dump().get(key, default)
        except Exception:
            return default