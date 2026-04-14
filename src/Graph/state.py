from typing import List, Dict, Any, Optional
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
    VerdictType
)

class AgentState(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="ignore"
    )

    # ── معرّفات القضية ────────────────────────────────────────────────────────
    case_id:                str
    case_number:            Optional[str]       = None   # رقم القضية الرسمي
    court:                  Optional[str]       = None
    court_level:            Optional[CourtLevel] = None
    jurisdiction:           Optional[str]       = None   # الدائرة / المحكمة
    filing_date:            Optional[datetime]  = None
    referral_date:          Optional[datetime]  = None   # تاريخ إحالة للمحاكمة
    prosecutor_name:        Optional[str]       = None
    

    # ── الكيانات المستخلصة ───────────────────────────────────────────────────
    defendants:             List[Defendant]          = Field(default_factory=list)
    charges:                List[Charge]             = Field(default_factory=list)
    incidents:              List[CaseIncident]        = Field(default_factory=list)
    evidences:              List[Evidence]            = Field(default_factory=list)
    lab_reports:            List[LabReport]           = Field(default_factory=list)
    witness_statements:     List[WitnessStatement]    = Field(default_factory=list)
    confessions:            List[Confession]          = Field(default_factory=list)
    procedural_issues:      List[ProceduralIssue]     = Field(default_factory=list)
    prior_judgments:        List[PriorJudgment]       = Field(default_factory=list)
    defense_documents:      List[DefenseDocument]     = Field(default_factory=list)

    # ── مخرجات الـ Agents (تُملأ تدريجيًا) ──────────────────────────────────
    agent_outputs:          Dict[str, Any]       = Field(default_factory=dict)

    # ── تتبع الحالة ───────────────────────────────────────────────────────────
    completed_agents:       List[str]            = Field(default_factory=list)
    current_agent:          Optional[str]        = None
    errors:                 List[str]            = Field(default_factory=list)

    # ── الحكم النهائي (يُملأ فقط بواسطة Judge Agent) ─────────────────────────
    suggested_verdict:      Optional[VerdictType]  = None
    confidence_score:       Optional[float]        = None   # 0.0 → 1.0
    reasoning_trace:        Optional[List[Dict[str, str]]] = None

    # ── ميتاداتا ──────────────────────────────────────────────────────────────
    source_documents:       List[str]            = Field(default_factory=list)  # قائمة الملفات المدخلة
    last_updated:           Optional[datetime]   = None
    extraction_notes:       Optional[str]        = None