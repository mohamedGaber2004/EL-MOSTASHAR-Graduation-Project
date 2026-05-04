from enum import Enum

class Gender(str, Enum):
    MALE    = "ذكر"
    FEMALE  = "أنثى"
    UNKNOWN = "غير محدد"


class MentalState(str, Enum):
    SANE        = "سليم العقل"
    INSANE      = "مجنون"
    DIMINISHED  = "ناقص الأهلية"
    COERCED     = "مكره"
    DRUNK       = "سكران"
    UNKNOWN     = "غير محدد"


class LawCode(str, Enum):
    PENAL           = "قانون العقوبات"
    ANTI_DRUGS      = "قانون مكافحة المخدرات"
    ANTI_TERROR     = "قانون مكافحة الإرهاب"
    TRAFFIC         = "قانون المرور"
    FIREARMS        = "قانون الأسلحة والذخائر"
    CHILD           = "قانون الطفل"
    CYBER           = "قانون مكافحة جرائم المعلومات"
    ANTI_CORRUPTION = "قانون مكافحة الفساد"
    MONEY_LAUNDRY   = "قانون مكافحة غسيل الأموال"
    OTHER           = "قانون آخر"


class ComplicityRole(str, Enum):
    PRINCIPAL      = "فاعل أصلي"
    CO_PRINCIPAL   = "فاعل مشارك"
    ACCOMPLICE     = "شريك"
    INSTIGATOR     = "محرض"
    FACILITATOR    = "مساعد"
    UNKNOWN        = "غير محدد"


class EvidenceType(str, Enum):
    MATERIAL       = "دليل مادي"
    DOCUMENTARY    = "دليل مستندي"
    TESTIMONIAL    = "دليل شهادة"
    DIGITAL        = "دليل رقمي"
    FORENSIC       = "دليل جنائي"
    CONFESSION     = "اعتراف"
    CIRCUMSTANTIAL = "قرينة"
    OTHER          = "أخرى"


class ValidityStatus(str, Enum):
    VALID        = "سليم"
    QUESTIONABLE = "مشكوك فيه"
    INVALID      = "باطل"
    PENDING      = "قيد الفحص"


class WitnessType(str, Enum):
    PROSECUTION  = "شاهد إثبات"
    DEFENSE      = "شاهد نفي"
    VICTIM       = "مجني عليه"
    EXPERT       = "خبير"
    INFORMANT    = "مخبر"
    EYEWITNESS   = "شاهد عيان"
    CHARACTER    = "شاهد في شخصية المتهم"


class WitnessRelation(str, Enum):
    RELATIVE     = "قريب"
    FRIEND       = "صديق"
    ENEMY        = "عدو"
    COLLEAGUE    = "زميل"
    STRANGER     = "غريب"
    OFFICER      = "ضابط"
    UNKNOWN      = "غير محدد"


class ProcedureType(str, Enum):
    ARREST             = "قبض"
    SEARCH_PERSON      = "تفتيش شخص"
    SEARCH_PLACE       = "تفتيش مسكن"
    SEARCH_VEHICLE     = "تفتيش مركبة"
    INTERROGATION      = "استجواب"
    CONFRONTATION      = "مواجهة"
    LINEUP             = "عرض على المجني عليه"
    NOTIFICATION       = "إخطار نيابة"
    PRETRIAL_DETENTION = "حبس احتياطي"
    SEIZURE            = "ضبط"
    WIRETAP            = "تسجيل مكالمات"
    SURVEILLANCE       = "مراقبة"
    OTHER              = "إجراء آخر"


class NullityType(str, Enum):
    ABSOLUTE = "بطلان مطلق"
    RELATIVE = "بطلان نسبي"
    NONE     = "لا بطلان"


class CourtLevel(str, Enum):
    MISDEMEANOR  = "محكمة الجنح"
    CRIMINAL     = "محكمة الجنايات"
    APPEALS      = "محكمة الاستئناف"
    CASSATION    = "محكمة النقض"
    SUPREME      = "المحكمة العليا"
    ECONOMIC     = "المحكمة الاقتصادية"
    MILITARY     = "المحكمة العسكرية"
    JUVENILE     = "محكمة الأحداث"


class VerdictType(str, Enum):
    CONVICTION        = "إدانة"
    ACQUITTAL         = "براءة"
    DISMISSAL         = "قضية لا وجه لإقامة الدعوى"
    INSUFFICIENT_EVD  = "عدم كفاية الأدلة"
    NO_JURISDICTION   = "عدم الاختصاص"
    MISTRIAL          = "بطلان المحاكمة"
    PARTIAL_GUILTY    = "إدانة جزئية"


class IncidentType(str, Enum):
    HOMICIDE       = "قتل عمد"
    MANSLAUGHTER   = "قتل خطأ"
    ASSAULT        = "ضرب وجرح"
    ASSAULT_FATAL  = "ضرب أفضى إلى موت"
    ROBBERY        = "سطو مسلح"
    THEFT          = "سرقة"
    DRUG_POSSESS   = "حيازة مخدرات"
    DRUG_TRAFFIC   = "اتجار في مخدرات"
    RAPE           = "اغتصاب"
    SEXUAL_ASSAULT = "تحرش"
    FRAUD          = "نصب واحتيال"
    FORGERY        = "تزوير"
    TERROR         = "إرهاب"
    OTHER          = "أخرى"


class LegalDocType(str, Enum):
    AMR_IHALA       = "amr_ihala"
    MAHDAR_DABT     = "mahdar_dabt"
    MAHDAR_ISTIJWAB = "mahdar_istijwab"
    AQWAL_SHUHUD    = "aqwal_shuhud"
    TAQRIR_TIBBI    = "taqrir_tibbi"
    MOZAKARET_DIFA  = "mozakaret_difa"