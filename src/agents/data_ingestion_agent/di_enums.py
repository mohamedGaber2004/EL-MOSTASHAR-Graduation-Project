from enum import Enum

from src.agents.data_ingestion_agent.data_ingestion_output_model import (
    Defendant, Charge, CaseIncident, Evidence, LabReport,
    WitnessStatement, Confession,DefenseDocument, CriminalRecord, CriminalProceedings,
)

class DI_LegalDocType(str, Enum):
    AMR_IHALA       = "amr_ihala"
    MAHDAR_DABT     = "mahdar_dabt"
    MAHDAR_ISTIJWAB = "mahdar_istijwab"
    AQWAL_SHUHUD    = "aqual_elshuhud"
    TAQRIR_TIBBI    = "taqrir_tibbi"
    MOZAKARET_DIFA  = "mozakeret_difa"
    SAWABIQ         = "sgl_genaey"


class DI_enums(Enum) : 
    _LIST_KEYS = {
        "defendants", "charges", "incidents", "evidences",
        "lab_reports", "witness_statements", "confessions",
        "criminal_records", "defense_documents",
        "defense_procedural_issues", "procedural_issues",
    }

    LIST_DEDUP: dict[str, tuple[str, ...]] = {
        "defendants":                ("national_id", "name"),
        "charges":                   ("law_code", "article_number", "description"),
        "incidents":                 ("incident_type", "incident_date", "incident_location"),
        "evidences":                 ("evidence_type", "description", "seizure_date"),
        "lab_reports":               ("report_type", "examiner_name", "examination_date"),
        "witness_statements":        ("witness_name", "statement_date"),
        "confessions":               ("defendant_name", "confession_date", "text"),
        "criminal_proceedings":      ("procedure_type", "description", "conducting_officer"),
        "procedural_issues":         ("procedure_type", "issue_description"),
        "defense_procedural_issues": ("procedure_type", "issue_description"),
        "criminal_records":          ("defendant_name",),
        "defense_documents":         ("submitted_by", "defendant_name"),
    }

    ENTITIES_MAPPING = {
        "defendants":                (Defendant,           "defendants"),
        "charges":                   (Charge,              "charges"),
        "incidents":                 (CaseIncident,        "incidents"),
        "evidences":                 (Evidence,            "evidences"),
        "lab_reports":               (LabReport,           "lab_reports"),
        "witness_statements":        (WitnessStatement,    "witness_statements"),
        "confessions":               (Confession,          "confessions"),
        "criminal_proceedings":      (CriminalProceedings, "criminal_proceedings"),
        "defense_documents":         (DefenseDocument,     "defense_documents"),
        "criminal_records":          (CriminalRecord,      "criminal_records"),
    }


    _ROUTE_PATTERNS: list[tuple[str, set[str]]] = [
        (
            DI_LegalDocType.AMR_IHALA.value,
            {"amr_ihala", "امر_الاحالة", "أمر_الإحالة", "ihala", "referral",
            "amr", "احالة", "إحالة"},
        ),
        (
            DI_LegalDocType.MAHDAR_DABT.value,
            {"mahdar_dabt", "محضر_الضبط", "mahdar_qesm", "قسم_الشرطة", "dabt",
            "ضبط", "تفتيش", "استدلالات", "jame3", "jame", "استدلال", "mahdar_jame"},
        ),
        (
            DI_LegalDocType.MAHDAR_ISTIJWAB.value,
            {"mahdar_istijwab", "محضر_الاستجواب", "istijwab", "استجواب",
            "tahqiq", "تحقيق", "niyaba", "نيابة"},
        ),
        (
            DI_LegalDocType.AQWAL_SHUHUD.value,
            {"aqwal_shuhud", "أقوال_الشهود", "shuhud", "شهود", "شاهد",
            "aqwal", "شهادة"},
        ),
        (
            DI_LegalDocType.TAQRIR_TIBBI.value,
            {"taqrir_tibbi", "تقرير_طبي", "tibbi", "طبي", "شرعي", "معمل",
            "lab", "taqrir", "تقرير", "forensic", "kimawi", "كيميائي",
            "balisti", "بلستي"},
        ),
        (
            DI_LegalDocType.MOZAKARET_DIFA.value,
            {"mozakeret_difa", "مذكرة_الدفاع", "difa", "دفاع", "mozakera",
            "مذكرة", "defense"},
        ),
        (
            DI_LegalDocType.SAWABIQ.value,
            {"sawabiq", "سوابق", "صحيفة_جنائية", "سجل_جنائي", "criminal_record",
            "sahifa", "صحيفة"},
        ),
    ]

    CASE_META_ATTRIBUTES = ["case_number", "court", "court_level", "jurisdiction","prosecutor_name", "referral_order_text", "filing_date", "referral_date"]
    AGENT_INGESTION_TXT_FILES_ENCODING = ("utf-8", "cp1256", "latin-1")