from enum import Enum


class LegalDocType(str, Enum):
    AMR_IHALA       = "amr_ihala"
    MAHDAR_DABT     = "mahdar_dabt"
    MAHDAR_ISTIJWAB = "mahdar_istijwab"
    AQWAL_SHUHUD    = "aqual_elshuhud"
    TAQRIR_TIBBI    = "taqrir_tibbi"
    MOZAKARET_DIFA  = "mozakeret_difa"
    SAWABIQ         = "sgl_genaey"

class CoercionType(str, Enum):
    """نوع الإكراه المزعوم على الاعتراف."""
    PHYSICAL       = "إكراه جسدي"
    PSYCHOLOGICAL  = "إكراه نفسي"
    ILLEGAL_ARREST = "قبض غير مشروع"
    PROLONGED_DETENTION = "احتجاز مطوّل"
    NONE           = "لا يوجد"
    UNKNOWN        = "غير محدد"

class ConfessionAdmissibilityStatus(str, Enum):
    """حكم قبول الاعتراف."""
    ADMISSIBLE           = "مقبول"
    INADMISSIBLE_COERCED = "مرفوض — إكراه"
    INADMISSIBLE_PROCEDURAL = "مرفوض — خلل إجرائي"
    INADMISSIBLE_SOLE_EVIDENCE = "مرفوض — دليل وحيد بلا سند مادي"
    CONDITIONALLY_ADMISSIBLE = "مقبول بشروط"
    REQUIRES_CORROBORATION   = "يحتاج دليلاً تعزيزياً"
 
 
class WitnessReliabilityLevel(str, Enum):
    """درجة موثوقية الشاهد."""
    HIGH     = "عالية"
    MEDIUM   = "متوسطة"
    LOW      = "منخفضة"
    EXCLUDED = "مستبعد"
 
 
class CivilClaimStatus(str, Enum):
    """حالة الدعوى المدنية التبعية."""
    FILED          = "مقامة"
    NOT_FILED      = "لم تُقَم"
    RESERVED       = "محجوزة للفصل"
    AWARDED        = "مقضي بها"
    REJECTED       = "مرفوضة"
 
class ProsecutionArgumentStrength(str, Enum):
    """قوة حجة الاتهام."""
    STRONG   = "قوية"
    MODERATE = "متوسطة"
    WEAK     = "ضعيفة"

class AgentsEnums:
    AGENT_INVOKATION_ERRORS  = ("429", "rate_limit", "rate_limit_exceeded", "tokens per minute", "please try again")
    AGENT_INGESTION_TXT_FILES_ENCODING = ("utf-8", "cp1256", "latin-1")
    CRIMINAL_PROCEDURE_LAW_ID = "criminal_procedure"

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
            "الدستور الجنائي":              "criminal_constitution"
        }
        return mapping.get(law_code, law_code)
    
