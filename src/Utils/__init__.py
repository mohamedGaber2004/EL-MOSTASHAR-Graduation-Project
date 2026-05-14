from .file_utils.text_loader import _read_file , MultiEncodingTextLoader
from .Enums.norm_and_regu import norm_regu
from .Enums.regex_utils import (
    _to_western_digits,
    _stable_id,
    _normalize_article_no,
    reg
)
from .file_utils.files_extractors import (
    AmendmentExtractor,
    LawExtractor ,
    Amendment , 
    ExtractedLaw , 
)

from ..Graph.main_entity_classes import (
    Defendant,
    Charge,
    Evidence,
    LabReport,
    WitnessStatement,
    Confession,
    ProceduralIssue,
    DefenseDocument,
    CaseIncident,
    EvidenceAnalysis,
    DefenseArgument,
    JudicialPrinciple,
    FinalJudgment,
    CaseMeta,
    AmrIhalaExtraction,
    MahdarDabtExtraction,
    MahdarIstijwabExtraction,
    AqwalShuhudExtraction,
    TaqrirTibbiExtraction,
    MozakaretDifaExtraction,
    CriminalRecord
)