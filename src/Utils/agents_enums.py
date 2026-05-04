from enum import Enum
from src.Utils import VerdictType


class LegalDocType(str, Enum):
    AMR_IHALA       = "amr_ihala"
    MAHDAR_DABT     = "mahdar_dabt"
    MAHDAR_ISTIJWAB = "mahdar_istijwab"
    AQWAL_SHUHUD    = "aqwal_shuhud"
    TAQRIR_TIBBI    = "taqrir_tibbi"
    MOZAKARET_DIFA  = "mozakaret_difa"


class AgentsEnums:
    AGENT_INVOKATION_ERRORS  = ("429", "rate_limit", "rate_limit_exceeded", "tokens per minute", "please try again")
    AGENT_INGESTION_TXT_FILES_ENCODING = ("utf-8", "cp1256", "latin-1")

    VERDICT_MAP = {
        "إدانة":         VerdictType.CONVICTION,
        "براءة":         VerdictType.ACQUITTAL,
        "إدانة جزئية":  VerdictType.PARTIAL_GUILTY,
        "guilty":        VerdictType.CONVICTION,
        "acquittal":     VerdictType.ACQUITTAL,
        "partial_guilty": VerdictType.PARTIAL_GUILTY,
    }

    
    CRIMINAL_PROCEDURE_LAW_ID = "criminal_procedure"  # was "egyptian_criminal_procedure"

    @staticmethod
    def law_code_to_kg_id(law_code: str) -> str:
        mapping = {
            "قانون العقوبات":               "penal_code",
            "قانون الإجراءات الجنائية":     "criminal_procedure",
            "قانون مكافحة المخدرات":        "anti_drugs",
            "قانون مكافحة الإرهاب":         "anti_terror",
            "قانون الأسلحة والذخائر":       "weapons_ammunition",
            "قانون مكافحة جرائم المعلومات": "cybercrime",
            "قانون مكافحة غسيل الأموال":    "money_laundering",
            "قانون الطوارئ":                "emergency_law",
        }
        return mapping.get(law_code, law_code)