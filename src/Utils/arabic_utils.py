import re
import hashlib
from typing import Dict




_ARTICLE_MARK_RE = re.compile(
    r"""(?m)^\s*(?:المادة|مادة)\s*(?:\(\s*)?
(?P<num>[0-9٠-٩]+(?:\s*(?:مكرر(?:[اأإآى])?)?)?(?:\s*\(\s*[اأإآA-Za-z]\s*\))?)
(?:\s*\))?\s*[:：\-–\s](?P<rest>.*)$""",
    re.VERBOSE,
)

_ANY_ARTICLE_RE = re.compile(
    r'(?:المادة|مادة)\s*([٠-٩0-9]+(?:\s*مكرر[اًَُِّْأإآى]?)?\s*'
    r'(?:/\s*[اأإآA-Za-zأ-ي])?\s*'
    r'(?:\(\s*[اأإآA-Za-zأ-ي]\s*\))?)',
    re.UNICODE,
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
    "يناير": "01", "فبراير": "02", "مارس": "03", "أبريل": "04",
    "مايو":  "05", "يونيو":  "06", "يوليو": "07", "أغسطس": "08",
    "سبتمبر":"09", "أكتوبر": "10", "نوفمبر":"11", "ديسمبر": "12",
}

REF_RE = re.compile(
    r"""(?:المادة|مادة|المواد)\s*(?:\(\s*)?(?P<num>[0-9٠-٩]+(?:\s*(?:مكرر(?:[اأإآى])?)?)?(?:\s*\(\s*[اأإآA-Za-z]\s*\))?)(?:\s*\))?(?:\s*(?:و|،|,)\s*(?:\(\s*)?(?P<num2>[0-9٠-٩]+(?:\s*مكرر(?:[اأإآى])?)?)(?:\s*\))?)?""",
    re.VERBOSE)

PENALTY_PATTERNS = {
    'سجن': re.compile(
        r'(?:السجن|بالسجن|سجن[اً]?)\s*(?:مدة|لمدة)?\s*(?:(?:من|لا\s+تقل\s+عن)\s*)?([0-9٠-٩]+)\s*(?:إلى|حتى|-)\s*([0-9٠-٩]+)\s*(سنة|سنوات|عام|أعوام)',
        re.UNICODE),
    'غرامة': re.compile(
        r'(?:غرامة|بغرامة)\s*(?:مالية)?\s*(?:لا\s+)?(?:تقل\s+عن|من)\s*([0-9٠-٩,]+)\s*(?:جنيه|دولار|ريال)?(?:\s*(?:ولا\s+تزيد\s+على|إلى|حتى|-)\s*([0-9٠-٩,]+))?',
        re.UNICODE),
    'إعدام':      re.compile(r'(?:الإعدام|بالإعدام|القتل|قتل[اً]?|إعدام[اً]?)', re.UNICODE),
    'أشغال_شاقة': re.compile(r'الأشغال\s+الشاقة', re.UNICODE),
}

DEF_RE = re.compile(
    r'(?:يُقصد|المقصود|يُراد|تعني|يعني|يُعرَّف)\s+(?:ب|من)?\s*["\']?([^"\'"]+)["\']?\s*[:،]\s*(.+?)(?:\.|$)',
    re.UNICODE)

DEFAULT_FOLDER_TO_LAW_KEY : Dict[str, str] = {
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


def _to_western_digits(s: str) -> str:
    return s.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))

def _normalize_article_no(token: str) -> str:
    token = _to_western_digits(token.strip())
    token = re.sub(r"\s+", " ", token)
    token = re.sub(r"مكر(?:ر[اأإآىًٍَُِّْ]?)", "مكرر", token)
    return token.strip()

def _stable_id(*components) -> str:
    combined = "|".join(str(c) for c in components)
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()[:16]

    

