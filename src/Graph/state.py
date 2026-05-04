from typing import List, Dict, Optional
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime
from src.Utils import (
    Defendant,
    Charge,
    CaseIncident,
    Evidence,
    LabReport,
    WitnessStatement,
    Confession,
    ProceduralIssue,
    PriorJudgment,
    DefenseDocument,
    CourtLevel,
    EvidenceAnalysis,
    DefenseArgument,
    JudicialPrinciple,
    FinalJudgment
)

class AgentState(BaseModel):
    # ── Case Meta ──────────────────
    case_id:                str 
    case_number:            Optional[str]        = Field(default=None)
    court:                  Optional[str]        = Field(default=None)
    court_level:            Optional[CourtLevel] = Field(default=None)
    jurisdiction:           Optional[str]        = Field(default=None)
    filing_date:            Optional[datetime]   = Field(default=None)
    referral_date:          Optional[datetime]   = Field(default=None)
    prosecutor_name:        Optional[str]        = Field(default=None)
    referral_order_text:    Optional[str]        = Field(default=None, description="نص أمر الإحالة الحاكم لنطاق القضية")

    # ── Extracted Entities ───────────────────────────────────────────────────
    defendants:             List[Defendant]           = Field(default_factory=list)
    charges:                List[Charge]              = Field(default_factory=list) 
    incidents:              List[CaseIncident]        = Field(default_factory=list)
    evidences:              List[Evidence]            = Field(default_factory=list)
    lab_reports:            List[LabReport]           = Field(default_factory=list)
    witness_statements:     List[WitnessStatement]    = Field(default_factory=list)
    confessions:            List[Confession]          = Field(default_factory=list)
    procedural_issues:      List[ProceduralIssue]     = Field(default_factory=list)
    prior_judgments:        List[PriorJudgment]       = Field(default_factory=list)
    defense_documents:      List[DefenseDocument]     = Field(default_factory=list)

    # ── Structured Outputs ───────────────
    evidence_analysis:      List[EvidenceAnalysis]    = Field(default_factory=list)
    defense_arguments:      List[DefenseArgument]     = Field(default_factory=list)
    applied_principles:     List[JudicialPrinciple]   = Field(default_factory=list)
    legal_articles:         List[str]                 = Field(default_factory=list)

    # ── Status ────────────────────────────────────────────────
    completed_agents:       List[str]            = Field(default_factory=list)
    errors:                 List[str]            = Field(default_factory=list)
    confidence_score:       Optional[float]      = Field(default=0.0, ge=0, le=1)

    # ── Final Verdict ─────────────────────
    suggested_verdict:      Optional[FinalJudgment] = None

    # ── Metadata ──────────────────────────────────────────────────────────────
    source_documents:       List[str]            = Field(default_factory=list)
    last_updated:           Optional[datetime]   = Field(default_factory=datetime.now)
    extraction_notes:       Optional[str]        = None

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="ignore"
    )

    def get(self, key, default=None):
        try:
            return self.model_dump().get(key, default)
        except Exception:
            return default