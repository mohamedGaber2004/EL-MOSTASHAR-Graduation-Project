from enum import Enum


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


class CV_Enums(Enum):
    AGENT_KEY = "confession_validity"
    
    _COERCION_MAP: dict[str, CoercionType] = {
        "إكراه جسدي":       CoercionType.PHYSICAL,
        "إكراه نفسي":       CoercionType.PSYCHOLOGICAL,
        "قبض غير مشروع":   CoercionType.ILLEGAL_ARREST,
        "احتجاز مطوّل":    CoercionType.PROLONGED_DETENTION,
        "لا يوجد":          CoercionType.NONE,
        "غير محدد":         CoercionType.UNKNOWN,
    }

    _ADMISSIBILITY_MAP: dict[str, ConfessionAdmissibilityStatus] = {
        "مقبول":                            ConfessionAdmissibilityStatus.ADMISSIBLE,
        "مرفوض — إكراه":                  ConfessionAdmissibilityStatus.INADMISSIBLE_COERCED,
        "مرفوض — خلل إجرائي":            ConfessionAdmissibilityStatus.INADMISSIBLE_PROCEDURAL,
        "مرفوض — دليل وحيد بلا سند مادي": ConfessionAdmissibilityStatus.INADMISSIBLE_SOLE_EVIDENCE,
        "مقبول بشروط":                     ConfessionAdmissibilityStatus.CONDITIONALLY_ADMISSIBLE,
        "يحتاج دليلاً تعزيزياً":          ConfessionAdmissibilityStatus.REQUIRES_CORROBORATION,
    }