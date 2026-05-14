from __future__ import annotations

from datetime import datetime
from typing import List, Optional

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
    CriminalProceedings
)

from src.Graph.states_and_schemas.Agents_output_models import (
    DefenseArgument,
    EvidenceScoring,
    FinalJudgment,
    JudicialPrinciple,
    ProceduralIssue
)
class AgentState(BaseModel):

    # ── Case Meta ─────────────────────────────────────  (from Data Ingestion Agent / Amr_ihala file) 
    case_id:             str
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

    # ── Extracted Entities ────────────────────────────  (from DataIngestionAgent)
    defendants:                List[Defendant]           = Field(default_factory=list) # from mahdar_dabt file
    charges:                   List[Charge]              = Field(default_factory=list) # from Amr_ihala file
    incidents:                 List[CaseIncident]        = Field(default_factory=list) # from mahdar_dabt file
    evidences:                 List[Evidence]            = Field(default_factory=list) # from mahdar_dabt file
    witness_statements:        List[WitnessStatement]    = Field(default_factory=list) # from mahdar_dabt and aqual_elshuhud files
    confessions:               List[Confession]          = Field(default_factory=list) # from mahdar_istijwab file
    lab_reports:               List[LabReport]           = Field(default_factory=list) # from tqrir_fnny file
    criminal_records:          List[CriminalRecord]      = Field(default_factory=list) # from sgl_genaey file
    criminal_proceedings:      List[CriminalProceedings] = Field(default_factory=list) # from mahdar_dabt file
    defense_documents:         List[DefenseDocument]     = Field(default_factory=list) # from mozakeret_difa file

    # ── Structured Analysis Outputs ───────────────────  (from downstream agents)
    procedural_issues:  List[ProceduralIssue]   = Field(default_factory=list)
    evidence_scores:    List[EvidenceScoring]  = Field(default_factory=list)
    defense_arguments:  List[DefenseArgument]   = Field(default_factory=list)
    applied_principles: List[JudicialPrinciple] = Field(default_factory=list)
    case_articles:      List[str]               = Field(default_factory=list)

    # ── Final Verdict ─────────────────────────────────  (from VerdictAgent)
    suggested_verdict: Optional[FinalJudgment] = None

    # ── Pipeline Status ───────────────────────────────
    completed_agents: List[str]         = Field(default_factory=list)
    current_agent:    Optional[str]     = None
    errors:           List[str]         = Field(default_factory=list)

    # ── Metadata ──────────────────────────────────────
    source_documents: List[str]         = Field(default_factory=list)
    last_updated:     Optional[datetime] = Field(default_factory=datetime.now)
    extraction_notes: Optional[str]     = None

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="ignore")

    def get(self, key: str, default=None):
        try:
            return self.model_dump().get(key, default)
        except Exception:
            return default