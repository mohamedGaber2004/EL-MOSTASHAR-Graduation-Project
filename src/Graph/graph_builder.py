import logging

from typing import List, Optional, Dict , Any
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime
from typing import Any, Dict
from IPython.display import display , Image
from langgraph.graph import END, START, StateGraph
from src.Utils import (
    Defendant,
    Charge,
    Evidence,
    LabReport,
    WitnessStatement,
    Confession,
    ProceduralIssue,
    DefenseDocument,
    PriorJudgment,
    CaseIncident
)

from src.Utils import (
    LawCode,
    EvidenceType,
    CourtLevel,
    ValidityStatus,
    VerdictType,
    WitnessType,
    ProcedureType,
    NullityType,
    IncidentType
)
from src.Utils import (
    CourtLevel,
    VerdictType
)

from src.Nodes  import (
    orchestrator_node,
    data_ingestion_node,
    procedural_auditor_node,
    legal_researcher_node,
    evidence_analyst_node,
    defense_analyst_node,
    judge_node
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")



# =============================================================================
#  AGENT STATE  (الحالة الكاملة للقضية)
# =============================================================================

class AgentState(BaseModel):
    """
    الحالة الكاملة للقضية — تجمع كل المعطيات المستخلصة من الوثائق.
    تُمرَّر بين الـ Agents عبر LangGraph.
    """
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


# =============================================================================
#  GRAPH CONSTRUCTION
# =============================================================================

def build_legal_graph() -> StateGraph:
    """
    Builds and compiles the Legal Multi-Agent LangGraph.

    Flow:
    START
      └─► orchestrator
            └─► data_ingestion
                  └─► procedural_auditor
                        └─► legal_researcher
                              └─► evidence_analyst
                                    └─► defense_analyst
                                          └─► judge
                                                └─► END
    """
    builder = StateGraph(AgentState)

    # ── Register Nodes ────────────────────────────────────────────────────────
    builder.add_node("orchestrator",       orchestrator_node)
    builder.add_node("data_ingestion",     data_ingestion_node)
    builder.add_node("procedural_auditor", procedural_auditor_node)
    builder.add_node("legal_researcher",   legal_researcher_node)
    builder.add_node("evidence_analyst",   evidence_analyst_node)
    builder.add_node("defense_analyst",    defense_analyst_node)
    builder.add_node("judge",              judge_node)

    # ── Register Edges ────────────────────────────────────────────────────────
    builder.add_edge(START, "orchestrator")
    builder.add_edge("orchestrator",       "data_ingestion")
    builder.add_edge("data_ingestion",     "procedural_auditor")
    builder.add_edge("procedural_auditor", "legal_researcher")
    builder.add_edge("legal_researcher",   "evidence_analyst")
    builder.add_edge("evidence_analyst",   "defense_analyst")
    builder.add_edge("defense_analyst",    "judge")
    builder.add_edge("judge",              END)

    return builder.compile()



def get_graph_visualization() -> bytes:
    """Returns Mermaid diagram PNG bytes of the graph."""
    return legal_graph.get_graph().draw_mermaid_png()


# =============================================================================
#  PUBLIC API
# =============================================================================

def run_case(state: AgentState) -> AgentState:
    logger.info("🚀 Starting legal pipeline for case: %s", state.case_id)
    final_state = legal_graph.invoke(state)
    logger.info("🏁 Pipeline complete — Verdict: %s", final_state.get("suggested_verdict"))
    return final_state



# =============================================================================
#  SAMPLE CASE FACTORY
# =============================================================================

def create_sample_case() -> AgentState:
    return AgentState(
        case_id      = "CASE-2024-001",
        case_number  = "1234/2024 جنايات",
        court        = "محكمة جنايات القاهرة",
        court_level  = CourtLevel.CRIMINAL,
        jurisdiction = "دائرة الجنايات الأولى",
        filing_date  = datetime(2024, 3, 15),

        source_documents = [
            "amr_ihalah.pdf",
            "mahdar_dabt.pdf",
            "taqrir_tibbi.pdf",
            "aqwal_shuhud.pdf",
            "mozakart_difa.pdf",
        ],

        defendants = [
            Defendant(
                name            = "أحمد محمد علي",
                national_id     = "29001011234567",
                age             = 30,
                gender          = "ذكر",
                occupation      = "عامل",
                prior_record    = False,
                complicity_role = "فاعل أصلي",
                arrest_date     = datetime(2024, 1, 10),
                in_custody      = True,
            )
        ],

        charges = [
            Charge(
                charge_id         = "CHG-001",
                statute           = "ضرب أفضى إلى موت",
                law_code          = LawCode.PENAL,
                article_number    = "236",
                description       = "ضرب المجني عليه بآلة راضة أفضى إلى وفاته",
                elements_required = ["فعل الاعتداء", "الوفاة", "رابطة السببية", "القصد الجنائي"],
                elements_proven   = {
                    "فعل الاعتداء":  True,
                    "الوفاة":        True,
                    "رابطة السببية":  True,
                    "القصد الجنائي": False,
                },
                penalty_range  = "السجن المشدد من 3 إلى 15 سنة",
                attempt_flag   = False,
            )
        ],

        incidents = [
            CaseIncident(
                incident_id          = "INC-001",
                incident_type        = IncidentType.ASSAULT_FATAL,
                incident_date        = datetime(2024, 1, 9, 23, 30),
                incident_location    = "شارع النيل — الجيزة",
                incident_description = "تشاجر المتهم مع المجني عليه وضربه بعصا على رأسه فأودى بحياته",
                perpetrator_names    = ["أحمد محمد علي"],
                victim_names         = ["سعيد عبد الله"],
                outcome              = "وفاة المجني عليه",
                linked_evidence_ids  = ["EV-001", "EV-002"],
            )
        ],

        evidences = [
            Evidence(
                evidence_id             = "EV-001",
                evidence_type           = EvidenceType.MATERIAL,
                description             = "عصا خشبية مضبوطة بمكان الحادث",
                seizure_date            = datetime(2024, 1, 10, 2, 0),
                seizure_location        = "شارع النيل — الجيزة",
                seized_by               = "النقيب محمود حسن",
                seizure_warrant_present = False,
                chain_of_custody_ok     = True,
                validity_status         = ValidityStatus.VALID,
                linked_charge_elements  = ["فعل الاعتداء"],
            ),
            Evidence(
                evidence_id            = "EV-002",
                evidence_type          = EvidenceType.FORENSIC,
                description            = "تقرير الصفة التشريحية يثبت الوفاة من إصابة الرأس",
                validity_status        = ValidityStatus.VALID,
                linked_charge_elements = ["الوفاة", "رابطة السببية"],
            ),
        ],

        witness_statements = [
            WitnessStatement(
                witness_name               = "محمود سعيد",
                witness_type               = WitnessType.EYEWITNESS,
                relation_to_defendant      = "غريب",
                was_sworn_in               = True,
                statement_summary          = "رأيت المتهم يضرب المجني عليه بعصا على رأسه",
                consistency_flag           = True,
                contradicts_other_evidence = False,
            ),
            WitnessStatement(
                witness_name               = "فاطمة علي",
                witness_type               = WitnessType.DEFENSE,
                relation_to_defendant      = "قريب",
                was_sworn_in               = True,
                statement_summary          = "المتهم كان في المنزل وقت الحادث",
                consistency_flag           = False,
                contradicts_other_evidence = True,
                contradiction_details      = "يتعارض مع شهادة الشاهد الأول وتقارير الكاميرات",
            ),
        ],

        confessions = [
            Confession(
                confession_id         = "CONF-001",
                defendant_name        = "أحمد محمد علي",
                confession_date       = datetime(2024, 1, 11),
                confession_stage      = "نيابة",
                obtained_after_arrest = True,
                legal_counsel_present = False,
                informed_of_rights    = False,
                coercion_claimed      = True,
                coercion_evidence     = "آثار كدمات على جسم المتهم موثقة طبيًا",
                text                  = "اعترف بضرب المجني عليه دفاعًا عن النفس",
                voluntary             = False,
                consistent_with_facts = True,
                retracted             = True,
                retraction_reason     = "ادعى الإكراه أثناء المحاكمة",
            )
        ],

        procedural_issues = [
            ProceduralIssue(
                issue_id             = "PROC-001",
                procedure_type       = ProcedureType.INTERROGATION,
                conducting_authority = "شرطة",
                warrant_present      = False,
                notification_to_da   = False,
                issue_description    = "استجواب المتهم بدون حضور محامٍ أو إخطار النيابة",
                nullity_type         = NullityType.ABSOLUTE,
                article_basis        = "م 124 إجراءات جنائية",
                affects_evidence_ids = ["CONF-001"],
                tainted_fruit_risk   = True,
                may_invalidate_case  = True,
            )
        ],

        defense_documents = [
            DefenseDocument(
                doc_id               = "DEF-001",
                doc_type             = "مذكرة دفاع",
                submitted_by         = "المحامي عمر فاروق",
                formal_defenses      = ["بطلان الاعتراف لعدم حضور محامٍ", "بطلان الاستجواب"],
                substantive_defenses = ["انتفاء القصد الجنائي", "الدفاع الشرعي"],
                alibi_claimed        = True,
                alibi_details        = "المتهم كان في منزله وقت الحادث — شهادة زوجته",
                supporting_principles= ["نقض 21/1/1982 — الاعتراف الباطل لا يعتد به"],
            )
        ],
    )


legal_graph = build_legal_graph()
graph_image = display(Image(get_graph_visualization()))