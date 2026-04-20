from .text_loader import _read_file , MultiEncodingTextLoader
from .norm_and_regu import norm_regu
from .regex_utils import (
    _to_western_digits,
    _stable_id,
    _normalize_article_no,
    reg
)
from .files_extractors import (
    AmendmentExtractor,
    LawExtractor ,
    Amendment , 
    ExtractedLaw , 
)
from .legal_enumerations import (
    Gender,
    MentalState,
    LawCode,
    ComplicityRole,
    EvidenceType,
    CourtLevel,
    ValidityStatus,
    VerdictType,
    WitnessRelation,
    WitnessType,
    ProcedureType,
    NullityType,
    IncidentType
)
from .main_entity_classes import (
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
from .api_client import ServiceCommunicator