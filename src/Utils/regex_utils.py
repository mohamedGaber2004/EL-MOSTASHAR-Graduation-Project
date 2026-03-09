from __future__ import annotations
from enum import Enum
import hashlib
import re


# =============================================================================
# SHARED UTILITIES
# =============================================================================

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


# =============================================================================
# REGEX PATTERNS
# =============================================================================

class reg(Enum) :
    PDF_UNICODES_REG = r"[\u0617-\u061A\u064B-\u0652\u0670]"
    DRUGS_ENTRIES_SECTIONS_REG = r'\n(\d+)\s*[–\-]\s*'
    DRUGS_TABLE_HEADER_NAME_REG = r'^([^(]+)(?:\(([^)]+)\))?'
    DRUGS_TABLE_CHEMICAL_NAME_REG = r'الاسم الكيميائي:\s*(.+?)(?=\n|$)'
    DRUGS_TABLE_TRADING_NAME_REG = r'الأسماء التجارية:\s*(.+?)(?=\n---|\n\d+|$)'
    WEAPON_TABLE_SCHEDUAL_ENTRIES_REG = r'^(\d+)\s*[–\-]\s*(.+)$'
    TBALE_DETECTION_REG = r'(?:الجدول|جدول)\s+رقم\s*\(?([0-9٠-٩]+)\)?'

    ORIGINAL_LAW_RE = re.compile(
        r'(?:القانون|قانون)\s+رقم\s+([٠-٩0-9]+)\s+لسنة\s+([٠-٩0-9]+)',
        re.UNICODE,
    )

    _ARTICLE_BOUNDARY_RE = re.compile(
        r"""(?m)^\s*(?:المادة|مادة)\s*(?:\(\s*)?"""
        r"""(?P<num>[0-9٠-٩]+(?:\s*(?:مكرر(?:[اأإآى])?)?)?"""
        r"""(?:\s*\(\s*[اأإآA-Za-z]\s*\))?)(?:\s*\))?"""
        r"""\s*[:：\-–\s]""",
        re.VERBOSE,
    )


    ARTICLE_1_2_ONLY_REG = r'المادة الأولى\s*[:：]?\s*(.*?)(?=المادة الثانية|$)'

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

    REF_RE = re.compile(
        r"(?:المادة|مادة|المواد)\s*(?:\(\s*)?(?P<num>[0-9٠-٩]+(?:\s*(?:مكرر(?:[اأإآى])?)?)?(?:\s*\(\s*[اأإآA-Za-z]\s*\))?)(?:\s*\))?(?:\s*(?:و|،|,)\s*(?:\(\s*)?(?P<num2>[0-9٠-٩]+(?:\s*مكرر(?:[اأإآى])?)?)(?:\s*\))?)?",
        re.VERBOSE,
    )

    DEF_RE = re.compile(
        r'(?:يُقصد|المقصود|يُراد|تعني|يعني|يُعرَّف)\s+(?:ب|من)?\s*["\']?([^"\'"]+)["\']?\s*[:،]\s*(.+?)(?:\.|$)',
        re.UNICODE,
    )

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

    _CASE_NUM_RE    = re.compile(r"(?:الطعن|طعن)\s+رق[مم]\s+([٠-٩0-9]+(?:\s*[,،]\s*[٠-٩0-9]+)*)\s+لسنة\s+([٠-٩0-9]+)",re.UNICODE)
    _RULING_DATE_RE = re.compile(r"جلسة\s+(?:[٠-٩0-9]+\s*/\s*[٠-٩0-9]+\s*/\s*[٠-٩0-9]{2,4}|[٠-٩0-9]+\s+(?:يناير|فبراير|مارس|أبريل|مايو|يونيو|يوليو|أغسطس|سبتمبر|أكتوبر|نوفمبر|ديسمبر)\s+[٠-٩0-9]{4})", re.UNICODE)
    _CHAMBER_RE     = re.compile(r"(?:الدائرة|دائرة)\s+([^\n،,]+?)(?=\n|،|,|$)", re.UNICODE)