from enum import Enum


class LegalDocType(str, Enum):
    AMR_IHALA       = "amr_ihala"
    MAHDAR_DABT     = "mahdar_dabt"
    MAHDAR_ISTIJWAB = "mahdar_istijwab"
    AQWAL_SHUHUD    = "aqual_elshuhud"
    TAQRIR_TIBBI    = "taqrir_tibbi"
    MOZAKARET_DIFA  = "mozakeret_difa"
    SAWABIQ         = "sgl_genaey"


class AgentsEnums:
    AGENT_INVOKATION_ERRORS  = ("429", "rate_limit", "rate_limit_exceeded", "tokens per minute", "please try again")
    AGENT_INGESTION_TXT_FILES_ENCODING = ("utf-8", "cp1256", "latin-1")

    
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