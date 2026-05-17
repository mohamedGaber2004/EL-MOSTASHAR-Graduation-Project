import re
from enum import Enum
from typing import Dict




class regex(Enum):
    # regex for chunker
    _CHAMBER_RE     = re.compile(r"(?:الدائرة|دائرة)\s+([^\n،,]+?)(?=\n|،|,|$)", re.UNICODE)
    _CASE_NUM_RE    = re.compile(r"(?:الطعن|طعن)\s+رق[مم]\s+([٠-٩0-9]+(?:\s*[,،]\s*[٠-٩0-9]+)*)\s+لسنة\s+([٠-٩0-9]+)",re.UNICODE)
    _RULING_DATE_RE = re.compile(r"جلسة\s+(?:[٠-٩0-9]+\s*/\s*[٠-٩0-9]+\s*/\s*[٠-٩0-9]{2,4}|[٠-٩0-9]+\s+(?:يناير|فبراير|مارس|أبريل|مايو|يونيو|يوليو|أغسطس|سبتمبر|أكتوبر|نوفمبر|ديسمبر)\s+[٠-٩0-9]{4})", re.UNICODE)

    _TABLE_HEADER_RE = re.compile(
        r'(?:الجدول|جدول)\s+رقم\s*\(?([0-9٠-٩]+)\)?',
        re.UNICODE,
    )

    _ARTICLE_BOUNDARY_RE = re.compile(
        r"""(?m)^\s*(?:المادة|مادة)\s*(?:\(\s*)?"""
        r"""(?P<num>[0-9٠-٩]+(?:\s*(?:مكرر(?:[اأإآى])?)?)?"""
        r"""(?:\s*\(\s*[اأإآA-Za-z]\s*\))?)(?:\s*\))?"""
        r"""\s*[:：\-–\s]""",
        re.VERBOSE,
    )

    ORIGINAL_LAW_RE = re.compile(
        r'(?:القانون|قانون)\s+رقم\s+([٠-٩0-9]+)\s+لسنة\s+([٠-٩0-9]+)',
        re.UNICODE,
    )

    _DATE_RE = re.compile(
        r"الموافق\s+(?:\w+\s+)?([0-9٠-٩]+)?\s*"
        r"(يناير|فبراير|مارس|أبريل|مايو|يونيو|يوليو|أغسطس|سبتمبر|أكتوبر|نوفمبر|ديسمبر)"
        r"\s+(?:سنة|عام)?\s*([0-9٠-٩]{4})",
        re.UNICODE,
    )

    _MONTH_MAP = {
        "يناير":"01","فبراير":"02","مارس":"03","أبريل":"04",
        "مايو":"05","يونيو":"06","يوليو":"07","أغسطس":"08",
        "سبتمبر":"09","أكتوبر":"10","نوفمبر":"11","ديسمبر":"12",
    }


class LawKeys(Enum):

    FOLDER_TO_LAW_KEY: Dict[str, str] = {
        "3okobat":         "penal_code",
        "8asl_amwal":      "money_laundering",
        "Asle7a":          "weapons_ammunition",
        "dostor_gena2y":   "criminal_constitution",
        "drugs":           "anti_drugs",
        "egra2at_gena2ya": "criminal_procedure",
        "erhab":           "anti_terror",
        "taware2":         "emergency_law",
        "technology":      "cybercrime",
    }

    LAW_KEY_TO_TITLE: Dict[str, str] = {
        "penal_code":            "قانون العقوبات",
        "money_laundering":      "قانون مكافحة غسل الأموال",
        "weapons_ammunition":    "قانون الأسلحة والذخيرة",
        "criminal_constitution": "الدستور الجنائي",
        "anti_drugs":            "قانون مكافحة المخدرات",
        "criminal_procedure":    "قانون الإجراءات الجنائية",
        "anti_terror":           "قانون مكافحة الإرهاب",
        "emergency_law":         "قانون الطوارئ",
        "cybercrime":            "قانون مكافحة جرائم تقنية المعلومات",
    }

    BOOK_TITLE_MAP: Dict[str, str] = {
        "mbade2_ultra_clean"        :"مبادئ محكمة النقض",
        "1762919079675_ultra_clean" : "مبادئ محكمة النقض - المجلد الثاني",
        "2019-2018_المستحدث"        :"المستحدث من المبادئ 2018-2019",
    }



class ChunkType(str, Enum):
    ARTICLE         = "article"
    PREAMBLE        = "preamble"
    AMENDED_ARTICLE = "amended_article"
    TABLE           = "table"
    RULING          = "ruling"
    PRINCIPLE       = "principle"


class AmendmentType(str, Enum):
    ADDITION     = "addition"
    DELETION     = "deletion"
    MODIFICATION = "modification"
    ADDITION_KEYWORDS = ["تضاف","يضاف","أضيف","إضافة","جديدة","جديد"]


class DocType(str, Enum):
    RULING    = "ruling"
    PRINCIPLE = "principle"


class Na2dOutputKey(str, Enum):
    RULINGS    = "rulings"
    PRINCIPLES = "principles"