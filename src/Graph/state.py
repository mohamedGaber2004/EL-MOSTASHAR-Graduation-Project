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

    # ── معرّفات القضية ────────────────────────────────────────────────────────
    case_id:                str # معرف القضية
    case_number:            Optional[str]        = Field(default=None,description="رقم القضية الرسمي")
    court:                  Optional[str]        = Field(default=None,description="المحكمة")
    court_level:            Optional[CourtLevel] = Field(default=None,description="درجة المحكمة")
    jurisdiction:           Optional[str]        = Field(default=None,description="الدائرة / المحكمة")
    filing_date:            Optional[datetime]   = Field(default=None,description="تاريخ القيد")
    referral_date:          Optional[datetime]   = Field(default=None,description="تاريخ إحالة للمحاكمة")
    prosecutor_name:        Optional[str]        = Field(default=None,description="اسم وكيل النيابة")

    

    # ── الكيانات المستخلصة ───────────────────────────────────────────────────
    defendants:             List[Defendant]           = Field(default_factory=list,description="المتهمون")
    charges:                List[Charge]              = Field(default_factory=list,description="التهم") 
    incidents:              List[CaseIncident]        = Field(default_factory=list,description="الوقائع / الأحداث")
    evidences:              List[Evidence]            = Field(default_factory=list,description="الأدلة")
    lab_reports:            List[LabReport]           = Field(default_factory=list,description="التقارير المعملية")
    witness_statements:     List[WitnessStatement]    = Field(default_factory=list,description="أقوال الشهود")
    confessions:            List[Confession]          = Field(default_factory=list,description="الاعترافات")
    procedural_issues:      List[ProceduralIssue]     = Field(default_factory=list,description="الإشكالات الإجرائية")
    prior_judgments:        List[PriorJudgment]       = Field(default_factory=list,description="الأحكام السابقة")
    defense_documents:      List[DefenseDocument]     = Field(default_factory=list,description="مستندات الدفاع")

    # ── مخرجات الـ Agents (تُملأ تدريجيًا) ──────────────────────────────────
    agent_outputs:          Dict[str, Any]       = Field(default_factory=dict)

    # ── تتبع الحالة ───────────────────────────────────────────────────────────
    completed_agents:       List[str]            = Field(default_factory=list)
    errors:                 List[str]            = Field(default_factory=list)

    # ── الحكم النهائي (يُملأ فقط بواسطة Judge Agent) ─────────────────────────
    suggested_verdict:      Optional[VerdictType]  = None          # الحكم المقترح
    confidence_score:       Optional[float]        = None          # درجة الثقة (0 → 1)
    reasoning_trace:        Optional[List[Dict[str, str]]] = None  # تسلسل الاستدلال

    # ── ميتاداتا ──────────────────────────────────────────────────────────────
    source_documents:       List[str]            = Field(default_factory=list)  # قائمة الملفات المدخلة
    last_updated:           Optional[datetime]   = None
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