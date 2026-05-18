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
    


    
