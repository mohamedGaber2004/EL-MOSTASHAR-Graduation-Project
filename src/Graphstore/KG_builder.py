from src.Config.config import DataPath
from src.Utils.arabic_utils import (
    _stable_id,
    _to_western_digits,
    _normalize_article_no,
    _ARTICLE_MARK_RE,
    _ANY_ARTICLE_RE , 
    REF_RE ,
    DEF_RE , 
    PENALTY_PATTERNS , 
    ORIGINAL_LAW_RE , 
    DEFAULT_FOLDER_TO_LAW_KEY
)


from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import fnmatch, logging, re, traceback
import pandas as pd

from langchain_core.documents import Document
from neo4j import GraphDatabase

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# SECTION 2: HELPER FUNCTIONS
# =============================================================================

def _dedup_items(items: List[Dict[str, Any]],
                 key_fields: List[str]) -> List[Dict[str, Any]]:
    seen, out = set(), []
    for item in items:
        key = tuple(item.get(f) for f in key_fields)
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


def _normalize_arabic_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"[\u0617-\u061A\u064B-\u0652\u0670]", "", text)
    text = text.replace("\u0640", "")
    for old, new in [("أ", "ا"), ("إ", "ا"), ("آ", "ا"), ("ى", "ي"),
                     ("ئ", "ي"), ("ة", "ه"), ("ؤ", "و")]:
        text = text.replace(old, new)
    return re.sub(r"\s+", " ", text).strip()


# =============================================================================
# SECTION 3: DATA STRUCTURES
# =============================================================================

@dataclass
class ScheduleEntry:
    entry_number:    str
    arabic_name:     Optional[str]            = None
    english_name:    Optional[str]            = None
    chemical_name:   Optional[str]            = None
    description:     Optional[str]            = None
    trade_names:     Optional[List[str]]      = None
    additional_info: Optional[Dict[str, str]] = field(default_factory=dict)


@dataclass
class Schedule:
    schedule_id:     str
    schedule_number: str
    title:           str
    description:     Optional[str]       = None
    entries:         List[ScheduleEntry] = field(default_factory=list)


@dataclass
class Amendment:
    amendment_id:            str
    amendment_date:          Optional[str] = None
    amendment_law_number:    Optional[str] = None
    amendment_law_title:     Optional[str] = None
    amended_article_numbers: List[str]     = field(default_factory=list)
    amendment_type:          Optional[str] = None  # modification|addition|deletion
    description:             Optional[str] = None
    effective_date:          Optional[str] = None


@dataclass
class ExtractedLaw:
    law_meta:    Dict[str, Any]
    articles:    List[Dict[str, Any]]
    penalties:   List[Dict[str, Any]]
    definitions: List[Dict[str, Any]]
    references:  List[Dict[str, Any]]
    topics:      List[Dict[str, Any]]
    amendments:  List[Amendment] = field(default_factory=list)
    schedules:   List[Schedule]  = field(default_factory=list)


# =============================================================================
# SECTION 4: SCHEDULE EXTRACTORS
# =============================================================================

class DrugScheduleExtractor:
    SCHEDULE_HEADER_RE = re.compile(
        r'(?:الجدول|جدول)\s+رقم\s*\(?([0-9٠-٩]+)\)?', re.UNICODE)

    def __init__(self, raw_text: str):
        self.raw_text = raw_text

    def extract_schedules(self) -> List[Schedule]:
        schedules, matches = [], list(self.SCHEDULE_HEADER_RE.finditer(self.raw_text))
        if not matches:
            return []
        for i, match in enumerate(matches):
            schedule_num = _to_western_digits(match.group(1))
            start = match.end()
            end   = matches[i + 1].start() if i + 1 < len(matches) else len(self.raw_text)
            schedule_text = self.raw_text[start:end].strip()
            lines = [l.strip() for l in schedule_text.split('\n') if l.strip()]
            schedules.append(Schedule(
                schedule_id=f"drug_schedule_{schedule_num}",
                schedule_number=schedule_num,
                title=lines[0] if lines else f"جدول رقم {schedule_num}",
                entries=self._extract_entries(schedule_text),
            ))
        return schedules

    def _extract_entries(self, text: str) -> List[ScheduleEntry]:
        entries, sections = [], re.split(r'\n(\d+)\s*[–\-]\s*', text)
        for i in range(1, len(sections), 2):
            if i + 1 >= len(sections):
                break
            entry_num  = sections[i].strip()
            entry_text = sections[i + 1].strip()
            arabic_name = english_name = chemical_name = trade_names = None
            nm = re.match(r'^([^(]+)(?:\(([^)]+)\))?', entry_text.split('\n')[0])
            if nm:
                arabic_name  = nm.group(1).strip()
                english_name = nm.group(2).strip() if nm.group(2) else None
            cm = re.search(
                r'الاسم الكيميائي:\s*(.+?)(?=\n|Chemical Name:|الوصف:|الأسماء|$)',
                entry_text, re.UNICODE)
            chemical_name = cm.group(1).strip() if cm else None
            tm = re.search(
                r'الأسماء التجارية:\s*(.+?)(?=\n---|\n\d+|$)', entry_text, re.UNICODE)
            if tm:
                trade_names = [n.strip() for n in re.split(r'[،,\-–]', tm.group(1)) if n.strip()]
            entries.append(ScheduleEntry(
                entry_number=entry_num, arabic_name=arabic_name,
                english_name=english_name, chemical_name=chemical_name,
                trade_names=trade_names))
        return entries


class WeaponScheduleExtractor:
    SCHEDULE_HEADER_RE = re.compile(r'جدول\s+رقم\s*\(?([0-9٠-٩]+)\)?', re.UNICODE)

    def __init__(self, raw_text: str):
        self.raw_text = raw_text

    def extract_schedules(self) -> List[Schedule]:
        schedules, matches = [], list(self.SCHEDULE_HEADER_RE.finditer(self.raw_text))
        if not matches:
            return []
        for i, match in enumerate(matches):
            schedule_num = _to_western_digits(match.group(1))
            start = match.end()
            end   = matches[i + 1].start() if i + 1 < len(matches) else len(self.raw_text)
            schedule_text = self.raw_text[start:end].strip()
            lines = [l.strip() for l in schedule_text.split('\n') if l.strip()]
            schedules.append(Schedule(
                schedule_id=f"weapon_schedule_{schedule_num}",
                schedule_number=schedule_num,
                title=lines[0] if lines else f"جدول رقم {schedule_num}",
                entries=self._extract_entries(schedule_text),
            ))
        return schedules

    def _extract_entries(self, text: str) -> List[ScheduleEntry]:
        entries = []
        for line in text.split('\n'):
            m = re.match(r'^(\d+)\s*[–\-]\s*(.+)$', line.strip())
            if m:
                entries.append(ScheduleEntry(
                    entry_number=m.group(1),
                    arabic_name=m.group(2).strip(),
                    description=m.group(2).strip()))
        return entries


# =============================================================================
# SECTION 5: BASE EXTRACTOR
# Main law files only. Amendments come exclusively from new* files.
# =============================================================================

class EnhancedBaseLawExtractor:

    ARTICLE_MARK_RE = _ARTICLE_MARK_RE
    PENALTY_PATTERNS = PENALTY_PATTERNS
    REF_RE = REF_RE
    DEF_RE = DEF_RE

    def __init__(self, raw_text: str, law_id: str, law_name: str,
                 promulgation_date: Optional[str] = None,
                 source: Optional[str] = None):
        self.raw_text           = raw_text or ""
        self._law_id            = law_id
        self._law_name          = law_name
        self._promulgation_date = promulgation_date
        self._source            = source or "الجريدة الرسمية"
        self._validation_warnings: List[str] = []

    def _normalize_article_no(self, token: str) -> str:
        # Extends the shared _normalize_article_no with ordinal replacements
        token = _normalize_article_no(token)
        for old, new in [("أولاً", "أولا"), ("أولا", "أولا"),
                         ("ثانياً", "ثانيا"), ("ثانيا", "ثانيا")]:
            token = token.replace(old, new)
        return token.strip()

    def extract_law_meta(self) -> Dict[str, Any]:
        return {
            "law_id":              self._law_id,
            "title":               self._law_name,
            "promulgation_date":   self._promulgation_date,
            "effective_date":      self._promulgation_date,
            "source":              self._source,
            "language":            "ar",
            "text_length":         len(self.raw_text),
            "validation_warnings": self._validation_warnings,
        }

    def extract_articles(self) -> List[Dict[str, str]]:
        text    = self.raw_text
        matches = list(self.ARTICLE_MARK_RE.finditer(text))
        articles: List[Dict[str, str]] = []
        if not matches:
            return [{"article_number": None, "text": text.strip(), "language": "ar"}]
        if matches[0].start() > 0:
            preamble = text[:matches[0].start()].strip()
            if preamble:
                articles.append({"article_number": None, "text": preamble, "language": "ar"})
        for i, m in enumerate(matches):
            start      = m.end()
            end        = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body       = text[start:end].strip()
            article_no = self._normalize_article_no(m.group("num") or "")
            articles.append({"article_number": article_no, "text": body, "language": "ar"})
        return articles

    def extract_penalties(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        penalties = []
        for article in articles:
            article_no = article.get("article_number")
            if not article_no:
                continue
            text = article.get("text", "")
            for m in self.PENALTY_PATTERNS['سجن'].finditer(text):
                penalties.append({
                    "article_number": article_no, "penalty_type": "سجن",
                    "min_value": int(_to_western_digits(m.group(1))),
                    "max_value": int(_to_western_digits(m.group(2))),
                    "unit": "سنة"})
            for m in self.PENALTY_PATTERNS['غرامة'].finditer(text):
                min_v = _to_western_digits(m.group(1).replace(',', ''))
                max_v = m.group(2)
                penalties.append({
                    "article_number": article_no, "penalty_type": "غرامة",
                    "min_value": int(min_v) if min_v else None,
                    "max_value": int(_to_western_digits(max_v.replace(',', ''))) if max_v else None,
                    "unit": "جنيه"})
            if self.PENALTY_PATTERNS['إعدام'].search(text):
                penalties.append({"article_number": article_no, "penalty_type": "إعدام",
                                   "min_value": None, "max_value": None, "unit": None})
            if self.PENALTY_PATTERNS['أشغال_شاقة'].search(text):
                penalties.append({"article_number": article_no, "penalty_type": "أشغال_شاقة",
                                   "min_value": None, "max_value": None, "unit": None})
        return _dedup_items(penalties, ["article_number", "penalty_type", "min_value", "max_value"])

    def extract_definitions(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        definitions = []
        for article in articles:
            article_no = article.get("article_number")
            if not article_no:
                continue
            for m in self.DEF_RE.finditer(article.get("text", "")):
                definitions.append({
                    "term": m.group(1).strip(),
                    "definition_text": m.group(2).strip(),
                    "defined_in_article": article_no, "language": "ar"})
        return _dedup_items(definitions, ["term", "defined_in_article"])

    def extract_references(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        refs = set()
        for a in articles:
            from_no = str(a.get("article_number", "")).strip()
            if not from_no or from_no == "None":
                continue
            for m in self.REF_RE.finditer(a.get("text", "")):
                for to_no in [m.group("num"), m.group("num2")]:
                    if to_no:
                        to_no = self._normalize_article_no(to_no)
                        if to_no and to_no != from_no:
                            refs.add((from_no, to_no, "direct"))
        return [{"from_article": f, "to_article": t, "reference_type": rt}
                for f, t, rt in sorted(refs)]

    def extract_topics(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        topics_map = {
            "قتل":    ["قتل", "قاتل"],
            "سرقة":   ["سرقة", "سارق"],
            "مخدرات": ["مخدر", "مخدرات"],
            "أسلحة":  ["سلاح", "أسلحة"],
            "إرهاب":  ["إرهاب", "إرهابي"],
        }
        article_topics = []
        for article in articles:
            article_no = article.get("article_number")
            if not article_no:
                continue
            text = _normalize_arabic_text(article.get("text", ""))
            for topic, keywords in topics_map.items():
                if any(_normalize_arabic_text(kw) in text for kw in keywords):
                    article_topics.append({"article_number": article_no,
                                           "topic_name": topic, "confidence": 0.8})
                    break
        return _dedup_items(article_topics, ["article_number", "topic_name"])

    def extract_schedules(self) -> List[Schedule]:
        return []

    def extract_all(self) -> ExtractedLaw:
        meta     = self.extract_law_meta()
        articles = self.extract_articles()
        return ExtractedLaw(
            law_meta    = meta,
            articles    = articles,
            penalties   = self.extract_penalties(articles),
            definitions = self.extract_definitions(articles),
            references  = self.extract_references(articles),
            topics      = self.extract_topics(articles),
            amendments  = [],       # always empty — filled by new* files
            schedules   = self.extract_schedules(),
        )


# =============================================================================
# SECTION 6: LAW-SPECIFIC EXTRACTORS
# =============================================================================

class PenalCodeExtractor(EnhancedBaseLawExtractor):
    def __init__(self, raw_text: str):
        super().__init__(raw_text, "penal_code", "قانون العقوبات", "1937-07-31")

class MoneyLaunderingExtractor(EnhancedBaseLawExtractor):
    def __init__(self, raw_text: str):
        super().__init__(raw_text, "money_laundering", "قانون مكافحة غسل الأموال", "2002-05-15")

class WeaponsAmmunitionExtractor(EnhancedBaseLawExtractor):
    def __init__(self, raw_text: str):
        super().__init__(raw_text, "weapons_ammunition", "قانون الأسلحة والذخيرة", "1954-09-22")
    def extract_schedules(self) -> List[Schedule]:
        return WeaponScheduleExtractor(self.raw_text).extract_schedules()

class CriminalConstitutionExtractor(EnhancedBaseLawExtractor):
    def __init__(self, raw_text: str):
        super().__init__(raw_text, "criminal_constitution", "الدستور الجنائي")

class DrugsLawExtractor(EnhancedBaseLawExtractor):
    def __init__(self, raw_text: str):
        super().__init__(raw_text, "anti_drugs", "قانون مكافحة المخدرات", "1960-05-01")
    def extract_schedules(self) -> List[Schedule]:
        return DrugScheduleExtractor(self.raw_text).extract_schedules()

class CriminalProcedureExtractor(EnhancedBaseLawExtractor):
    def __init__(self, raw_text: str):
        super().__init__(raw_text, "criminal_procedure", "قانون الإجراءات الجنائية", "1950-10-18")

class AntiTerrorExtractor(EnhancedBaseLawExtractor):
    def __init__(self, raw_text: str):
        super().__init__(raw_text, "anti_terror", "قانون مكافحة الإرهاب", "2015-08-16")

class EmergencyLawExtractor(EnhancedBaseLawExtractor):
    def __init__(self, raw_text: str):
        super().__init__(raw_text, "emergency_law", "قانون الطوارئ", "1958-01-01")

class CybercrimeExtractor(EnhancedBaseLawExtractor):
    def __init__(self, raw_text: str):
        super().__init__(raw_text, "cybercrime", "قانون مكافحة جرائم تقنية المعلومات", "2018-07-17")


def get_extractor(law_key: str, raw_text: str) -> EnhancedBaseLawExtractor:
    mapping = {
        "penal_code":            PenalCodeExtractor,
        "money_laundering":      MoneyLaunderingExtractor,
        "weapons_ammunition":    WeaponsAmmunitionExtractor,
        "criminal_constitution": CriminalConstitutionExtractor,
        "anti_drugs":            DrugsLawExtractor,
        "criminal_procedure":    CriminalProcedureExtractor,
        "anti_terror":           AntiTerrorExtractor,
        "emergency_law":         EmergencyLawExtractor,
        "cybercrime":            CybercrimeExtractor,
    }
    if law_key not in mapping:
        raise KeyError(f"Unknown law_key: '{law_key}'")
    return mapping[law_key](raw_text)


# =============================================================================
# SECTION 7: AMENDMENT LAW EXTRACTOR
# Handles ONLY files whose names start with "new".
# FILENAME IS PRIMARY source for amendment law number + year.
# =============================================================================

_ANY_ARTICLE_RE = _ANY_ARTICLE_RE


class AmendmentLawExtractor:

    ORIGINAL_LAW_RE = ORIGINAL_LAW_RE

    def __init__(self, raw_text: str, filename: Optional[str] = None):
        self.raw_text = raw_text
        self.filename = filename

    def _from_filename(self) -> Tuple[Optional[str], Optional[str]]:
        if not self.filename:
            return None, None
        m = re.search(r'_num([0-9]+)_([0-9]{4})\.txt$', self.filename, re.IGNORECASE)
        if m:
            return m.group(1), m.group(2)
        m = re.search(r'_([0-9]+)_([0-9]{4})\.txt$', self.filename, re.IGNORECASE)
        if m:
            return m.group(1), m.group(2)
        years = re.findall(r'((?:19|20)\d{2})', self.filename)
        nums  = [n for n in re.findall(r'(\d+)', self.filename) if n not in years]
        return (nums[-1] if nums else None), (years[-1] if years else None)

    def _from_header_fallback(self) -> Tuple[Optional[str], Optional[str]]:
        for m in self.ORIGINAL_LAW_RE.finditer(self.raw_text):
            try:
                a_w = int(_to_western_digits(m.group(1)))
                b_w = int(_to_western_digits(m.group(2)))
                if 1800 <= a_w <= 2100:
                    year, law_num = str(a_w), str(b_w)
                else:
                    law_num, year = str(a_w), str(b_w)
                if 1800 <= int(year) <= 2100 and 0 < int(law_num) < 10000:
                    return law_num, year
            except Exception:
                continue
        return None, None

    def _extract_law_num_and_year(self) -> Tuple[Optional[str], Optional[str]]:
        law_num, year = self._from_filename()
        if law_num and year:
            return law_num, year
        return self._from_header_fallback()

    def _extract_all_articles(self) -> List[str]:
        seen:    set       = set()
        ordered: List[str] = []
        for m in _ANY_ARTICLE_RE.finditer(self.raw_text):
            norm = _normalize_article_no(m.group(1))
            if norm and norm not in seen:
                seen.add(norm)
                ordered.append(norm)
        return ordered

    def _detect_amendment_type(self) -> str:
        sample = self.raw_text[:800]
        if any(w in sample for w in ['تضاف', 'يضاف', 'أضيف', 'إضافة', 'جديدة', 'جديد']):
            return 'addition'
        if any(w in sample for w in ['يلغى', 'ألغي', 'يحذف', 'حذف', 'إلغاء']):
            return 'deletion'
        return 'modification'

    def _extract_amendment_date(self, year: str) -> str:
        month_map = {
            'يناير': '01', 'فبراير': '02', 'مارس': '03', 'أبريل': '04',
            'مايو': '05', 'يونيو': '06', 'يوليو': '07', 'أغسطس': '08',
            'سبتمبر': '09', 'أكتوبر': '10', 'نوفمبر': '11', 'ديسمبر': '12',
        }
        m = re.search(
            r'الموافق\s+(?:\w+\s+)?([0-9٠-٩]+)?\s*'
            r'(يناير|فبراير|مارس|أبريل|مايو|يونيو|يوليو|أغسطس|سبتمبر|أكتوبر|نوفمبر|ديسمبر)'
            r'\s+(?:سنة|عام)?\s*([0-9٠-٩]{4})',
            self.raw_text, re.UNICODE)
        if m:
            day   = _to_western_digits(m.group(1) or '01').zfill(2)
            month = month_map.get(m.group(2), '01')
            yr    = _to_western_digits(m.group(3))
            return f"{yr}-{month}-{day}"
        return f"{year}-01-01"

    def extract_amendment(self, target_law_id: str) -> Optional[Amendment]:
        law_num, year = self._extract_law_num_and_year()
        if not law_num or not year:
            logger.warning(f"[SKIP] No amendment law number/year | {self.filename}")
            return None
        all_articles = self._extract_all_articles()
        if not all_articles:
            logger.warning(f"[SKIP] No article numbers found | {self.filename}")
            return None
        amendment_type = self._detect_amendment_type()
        amendment_date = self._extract_amendment_date(year)
        amendment_id   = _stable_id(target_law_id, law_num, year)
        desc_m = re.search(r'المادة الأولى\s*[:：]?\s*(.*?)(?=المادة الثانية|$)',
                           self.raw_text, re.DOTALL)
        description = (desc_m.group(1).strip()[:500] if desc_m else self.raw_text[:500])
        logger.info(f"[OK] {self.filename} → amendment law={law_num}/{year} "
                    f"type={amendment_type} articles={all_articles}")
        return Amendment(
            amendment_id=amendment_id,
            amendment_law_number=law_num,
            amendment_date=amendment_date,
            amendment_law_title=f"قانون رقم {law_num} لسنة {year}",
            amended_article_numbers=all_articles,
            amendment_type=amendment_type,
            description=description,
            effective_date=amendment_date,
        )


# =============================================================================
# SECTION 8: FILE READING UTILITIES
# =============================================================================

def _safe_read_text_file(fp: Path,
                         encodings: Sequence[str],
                         max_size_mb: Optional[int] = 50) -> Optional[str]:
    try:
        if max_size_mb and fp.stat().st_size / (1024 * 1024) > max_size_mb:
            return None
    except Exception:
        pass
    for enc in encodings:
        try:
            return fp.read_text(encoding=enc)
        except Exception:
            continue
    try:
        return fp.read_bytes().decode("utf-8", errors="replace")
    except Exception:
        return None


def read_all_txt_in_folder(folder: Path,
                           encoding_list: Optional[Sequence[str]] = None,
                           recursive: bool = False,
                           file_pattern: str = "*.txt",
                           max_file_size_mb: Optional[int] = 50,
                           exclude_pattern: Optional[str] = None) -> str:
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    encodings = list(encoding_list) if encoding_list else ["utf-8", "utf-16", "cp1256"]
    globber   = folder.rglob if recursive else folder.glob
    txt_files = sorted(p for p in globber(file_pattern) if p.is_file())
    if exclude_pattern:
        txt_files = [f for f in txt_files
                     if not fnmatch.fnmatch(f.name.lower(), exclude_pattern.lower())]
    parts = []
    for fp in txt_files:
        text = _safe_read_text_file(fp, encodings, max_file_size_mb)
        if text:
            parts.append(text)
    return "\n\n".join(parts).strip()


def ingest_dataset(root_dir: str,
                   folder_to_law_key: Dict[str, str]) -> List[Dict]:
    root      = Path(root_dir)
    summaries = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        if folder.name not in folder_to_law_key:
            continue
        law_key = folder_to_law_key[folder.name]
        try:
            raw_text = read_all_txt_in_folder(folder, exclude_pattern='new*')
            if not raw_text:
                summaries.append({"folder": folder.name, "law_key": law_key,
                                  "error": "empty"})
                continue
            extractor = get_extractor(law_key, raw_text)
            bundle    = extractor.extract_all()
            summaries.append({
                "folder": folder.name, "law_key": law_key,
                "law_id": bundle.law_meta.get("law_id"),
                "articles": len(bundle.articles),
                "penalties": len(bundle.penalties),
                "definitions": len(bundle.definitions),
                "references": len(bundle.references),
                "topics": len(bundle.topics),
                "schedules": len(bundle.schedules),
                "error": None,
            })
        except Exception as e:
            summaries.append({"folder": folder.name, "law_key": law_key,
                               "error": str(e)})
    return summaries


def ingest_amendment_files(root_dir: str,
                           folder_to_law_key: Dict[str, str]
                           ) -> Dict[str, List[Amendment]]:
    root = Path(root_dir)
    amendments_by_law: Dict[str, List[Amendment]] = {}
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        if folder.name not in folder_to_law_key:
            orphans = sorted(folder.glob("new*.txt"))
            if orphans:
                logger.warning(f"'{folder.name}' has {len(orphans)} new* file(s) "
                               f"but is NOT in folder_to_law_key — skipped")
            continue
        law_key   = folder_to_law_key[folder.name]
        new_files = sorted(folder.glob("new*.txt"))
        if not new_files:
            logger.info(f"No new*.txt in '{folder.name}'")
            continue
        logger.info(f"'{folder.name}' → {len(new_files)} file(s) → law_key='{law_key}'")
        law_amendments: List[Amendment] = []
        for af in new_files:
            text, enc = None, None
            for e in ['utf-8', 'utf-16', 'cp1256', 'windows-1256']:
                try:
                    text = af.read_text(encoding=e)
                    enc  = e
                    break
                except Exception:
                    continue
            if not text or not text.strip():
                logger.warning(f"  [SKIP] {af.name} — unreadable or empty")
                continue
            logger.info(f"  Reading {af.name} (enc={enc})")
            amendment = AmendmentLawExtractor(text, af.name).extract_amendment(
                target_law_id=law_key)
            if amendment:
                law_amendments.append(amendment)
        if law_amendments:
            amendments_by_law[law_key] = law_amendments
            logger.info(f"  → stored {len(law_amendments)} amendment(s) for '{law_key}'")
    return amendments_by_law


def debug_amendment_files(root_dir: str, folder_to_law_key: Dict[str, str]):
    root = Path(root_dir)
    print("\n" + "=" * 70 + "\nDEBUG: AMENDMENT FILE DISCOVERY\n" + "=" * 70)
    total_files = 0
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        new_files = sorted(folder.glob("new*.txt"))
        if not new_files:
            continue
        law_key = folder_to_law_key.get(folder.name, "NOT IN MAPPING ⚠️")
        print(f"\n📁 {folder.name}/ → law_key={law_key}")
        for af in new_files:
            total_files += 1
            text, used_enc = None, None
            for enc in ['utf-8', 'utf-16', 'cp1256', 'windows-1256']:
                try:
                    text = af.read_text(encoding=enc)
                    used_enc = enc
                    break
                except Exception:
                    continue
            if not text:
                print(f"  ❌ {af.name} — unreadable")
                continue
            extractor     = AmendmentLawExtractor(text, af.name)
            law_num, year = extractor._extract_law_num_and_year()
            articles      = extractor._extract_all_articles()
            atype         = extractor._detect_amendment_type()
            adate         = extractor._extract_amendment_date(year or "????")
            print(f"  ✓ {af.name}")
            print(f"    encoding : {used_enc}")
            print(f"    law/year : {law_num}/{year}")
            print(f"    type     : {atype}")
            print(f"    date     : {adate}")
            print(f"    articles : ({len(articles)}) {articles}")
    print(f"\n{'=' * 70}\nTotal new* files found: {total_files}\n{'=' * 70}\n")


# =============================================================================
# SECTION 9: KNOWLEDGE GRAPH
# =============================================================================

class EnhancedLegalKnowledgeGraph:

    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        try:
            self.driver.verify_connectivity()
            logger.info("✓ Neo4j connected")
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            raise

    def close(self):
        self.driver.close()

    def setup_schema(self, drop_existing: bool = False):
        with self.driver.session() as session:
            if drop_existing:
                session.run("MATCH (n) DETACH DELETE n")
                logger.info("✓ Existing data cleared")
            constraints = [
                "CREATE CONSTRAINT law_id_unique        IF NOT EXISTS FOR (l:Law)          REQUIRE l.law_id IS UNIQUE",
                "CREATE CONSTRAINT article_id_unique    IF NOT EXISTS FOR (a:Article)       REQUIRE a.article_id IS UNIQUE",
                "CREATE CONSTRAINT penalty_id_unique    IF NOT EXISTS FOR (p:Penalty)       REQUIRE p.penalty_id IS UNIQUE",
                "CREATE CONSTRAINT definition_id_unique IF NOT EXISTS FOR (d:Definition)    REQUIRE d.definition_id IS UNIQUE",
                "CREATE CONSTRAINT topic_id_unique      IF NOT EXISTS FOR (t:Topic)         REQUIRE t.topic_id IS UNIQUE",
                "CREATE CONSTRAINT schedule_id_unique   IF NOT EXISTS FOR (s:Schedule)      REQUIRE s.schedule_id IS UNIQUE",
                "CREATE CONSTRAINT entry_id_unique      IF NOT EXISTS FOR (e:ScheduleEntry) REQUIRE e.entry_id IS UNIQUE",
                "CREATE CONSTRAINT substance_id_unique  IF NOT EXISTS FOR (sub:Substance)   REQUIRE sub.substance_id IS UNIQUE",
                "CREATE CONSTRAINT amendment_id_unique  IF NOT EXISTS FOR (am:Amendment)    REQUIRE am.amendment_id IS UNIQUE",
            ]
            indexes = [
                "CREATE INDEX supersedes_date_idx IF NOT EXISTS FOR ()-[r:SUPERSEDES]-() ON (r.amendment_date)",
            ]
            for stmt in constraints + indexes:
                try:
                    session.run(stmt)
                except Exception:
                    pass
            logger.info("✓ Schema ready")

    def create_law(self, law_meta: Dict[str, Any]):
        with self.driver.session() as session:
            session.run("""
                MERGE (l:Law {law_id: $law_id})
                SET l.title             = $title,
                    l.promulgation_date = $promulgation_date,
                    l.source            = $source,
                    l.language          = $language
            """, law_id=law_meta.get("law_id"),
                title=law_meta.get("title"),
                promulgation_date=law_meta.get("promulgation_date"),
                source=law_meta.get("source"),
                language=law_meta.get("language", "ar"))

    def create_article(self, law_id: str, article_number: Optional[str],
                       text: str, position: int):
        article_id = _stable_id(law_id, article_number or "preamble", position)
        with self.driver.session() as session:
            session.run("""
                MATCH (l:Law {law_id: $law_id})
                MERGE (a:Article {article_id: $article_id})
                SET a.article_number = $article_number,
                    a.text           = $text,
                    a.law_id         = $law_id
                MERGE (l)-[r:CONTAINS]->(a)
                SET r.position = $position
            """, law_id=law_id, article_id=article_id,
                article_number=article_number, text=text, position=position)

    def create_penalty(self, penalty: Dict[str, Any], law_id: str):
        penalty_id = _stable_id(law_id, penalty.get("article_number"),
                                penalty.get("penalty_type"))
        with self.driver.session() as session:
            session.run("""
                MATCH (a:Article {law_id: $law_id, article_number: $article_number})
                MERGE (p:Penalty {penalty_id: $penalty_id})
                SET p.penalty_type = $penalty_type,
                    p.min_value    = $min_value,
                    p.max_value    = $max_value,
                    p.unit         = $unit
                MERGE (a)-[:HAS_PENALTY]->(p)
            """, law_id=law_id, penalty_id=penalty_id,
                article_number=penalty.get("article_number"),
                penalty_type=penalty.get("penalty_type"),
                min_value=penalty.get("min_value"),
                max_value=penalty.get("max_value"),
                unit=penalty.get("unit"))

    def create_definition(self, definition: Dict[str, Any], law_id: str):
        def_id = _stable_id(law_id, definition.get("term"))
        with self.driver.session() as session:
            session.run("""
                MATCH (a:Article {law_id: $law_id, article_number: $article_number})
                MERGE (d:Definition {definition_id: $def_id})
                SET d.term            = $term,
                    d.definition_text = $definition_text
                MERGE (a)-[:DEFINES]->(d)
            """, law_id=law_id, def_id=def_id,
                article_number=definition.get("defined_in_article"),
                term=definition.get("term"),
                definition_text=definition.get("definition_text"))

    def create_topic(self, topic_name: str):
        with self.driver.session() as session:
            session.run("MERGE (t:Topic {topic_id: $id, name: $name})",
                        id=_stable_id(topic_name), name=topic_name)

    def link_article_to_topic(self, law_id: str, article_number: str,
                               topic_name: str, confidence: float):
        with self.driver.session() as session:
            session.run("""
                MATCH (a:Article {law_id: $law_id, article_number: $article_number})
                MATCH (t:Topic {name: $topic_name})
                MERGE (a)-[r:TAGGED_WITH]->(t)
                SET r.confidence = $confidence
            """, law_id=law_id, article_number=article_number,
                topic_name=topic_name, confidence=confidence)

    def create_article_reference(self, law_id: str, from_article: str,
                                  to_article: str, ref_type: str):
        with self.driver.session() as session:
            session.run("""
                MATCH (from:Article {law_id: $law_id, article_number: $from_article})
                MATCH (to:Article   {law_id: $law_id, article_number: $to_article})
                MERGE (from)-[r:REFERENCES]->(to)
                SET r.reference_type = $ref_type
            """, law_id=law_id, from_article=from_article,
                to_article=to_article, ref_type=ref_type)

    def create_schedule(self, law_id: str, schedule: Schedule):
        with self.driver.session() as session:
            session.run("""
                MATCH (l:Law {law_id: $law_id})
                MERGE (s:Schedule {schedule_id: $schedule_id})
                SET s.schedule_number = $schedule_number,
                    s.title           = $title,
                    s.entry_count     = $count
                MERGE (l)-[:HAS_SCHEDULE]->(s)
            """, law_id=law_id, schedule_id=schedule.schedule_id,
                schedule_number=schedule.schedule_number,
                title=schedule.title, count=len(schedule.entries))

    def create_schedule_entry(self, schedule_id: str,
                               entry: ScheduleEntry, position: int):
        entry_id = _stable_id(schedule_id, entry.entry_number, position)
        with self.driver.session() as session:
            session.run("""
                MATCH (s:Schedule {schedule_id: $schedule_id})
                MERGE (e:ScheduleEntry {entry_id: $entry_id})
                SET e.entry_number = $entry_number,
                    e.arabic_name  = $arabic_name,
                    e.english_name = $english_name,
                    e.description  = $description
                MERGE (s)-[r:CONTAINS_ENTRY]->(e)
                SET r.position = $position
            """, schedule_id=schedule_id, entry_id=entry_id,
                entry_number=entry.entry_number, arabic_name=entry.arabic_name,
                english_name=entry.english_name, description=entry.description,
                position=position)
        if entry.chemical_name:
            self.create_substance(entry_id, entry)

    def create_substance(self, entry_id: str, entry: ScheduleEntry):
        sub_id = _stable_id("substance", entry.arabic_name or entry.english_name)
        with self.driver.session() as session:
            session.run("""
                MATCH (e:ScheduleEntry {entry_id: $entry_id})
                MERGE (sub:Substance {substance_id: $substance_id})
                SET sub.arabic_name   = $arabic_name,
                    sub.english_name  = $english_name,
                    sub.chemical_name = $chemical_name,
                    sub.trade_names   = $trade_names
                MERGE (e)-[:DEFINES_SUBSTANCE]->(sub)
            """, entry_id=entry_id, substance_id=sub_id,
                arabic_name=entry.arabic_name, english_name=entry.english_name,
                chemical_name=entry.chemical_name,
                trade_names=entry.trade_names or [])

    def create_amendment(self, law_id: str, amendment: Amendment):
        desc = (amendment.description or "")[:500]

        with self.driver.session() as session:
            session.run("""
                MATCH (l:Law {law_id: $law_id})
                MERGE (am:Amendment {amendment_id: $amendment_id})
                SET am.amendment_law_number = $amendment_law_number,
                    am.amendment_law_title  = $amendment_law_title,
                    am.amendment_date       = $amendment_date,
                    am.amendment_type       = $amendment_type,
                    am.description          = $description,
                    am.effective_date       = $effective_date
                MERGE (l)-[:HAS_AMENDMENT]->(am)
            """, law_id=law_id,
                amendment_id=amendment.amendment_id,
                amendment_law_number=amendment.amendment_law_number,
                amendment_law_title=amendment.amendment_law_title,
                amendment_date=amendment.amendment_date,
                amendment_type=amendment.amendment_type,
                description=desc,
                effective_date=amendment.effective_date)

        for article_number in amendment.amended_article_numbers:
            if not article_number:
                continue

            if amendment.amendment_type == 'addition':
                new_article_id = _stable_id(law_id, article_number,
                                            "added", amendment.amendment_date)
                with self.driver.session() as session:
                    session.run("""
                        MATCH (l:Law {law_id: $law_id})
                        MATCH (am:Amendment {amendment_id: $amendment_id})
                        MERGE (a_new:Article {article_id: $new_article_id})
                        ON CREATE SET
                            a_new.article_number = $article_number,
                            a_new.law_id         = $law_id,
                            a_new.text           = $description,
                            a_new.version        = $amendment_date,
                            a_new.is_addition    = true
                        MERGE (l)-[:CONTAINS]->(a_new)
                        MERGE (a_new)-[:AMENDED_BY {
                            amendment_type: $amendment_type,
                            amendment_date: $amendment_date
                        }]->(am)
                    """, law_id=law_id,
                        amendment_id=amendment.amendment_id,
                        new_article_id=new_article_id,
                        article_number=article_number,
                        description=desc,
                        amendment_date=amendment.amendment_date,
                        amendment_type=amendment.amendment_type)

                    session.run("""
                        MATCH (a_new:Article {article_id: $new_article_id})
                        MATCH (a_old:Article {law_id: $law_id,
                                              article_number: $article_number})
                        WHERE a_old.article_id <> $new_article_id
                          AND a_old.is_addition IS NULL
                        MERGE (a_new)-[:SUPERSEDES {
                            amendment_id:   $amendment_id,
                            amendment_date: $amendment_date
                        }]->(a_old)
                    """, new_article_id=new_article_id,
                        law_id=law_id,
                        article_number=article_number,
                        amendment_id=amendment.amendment_id,
                        amendment_date=amendment.amendment_date)

            else:
                new_article_id = _stable_id(law_id, article_number,
                                            "amended", amendment.amendment_date)
                with self.driver.session() as session:
                    session.run("""
                        MATCH (am:Amendment {amendment_id: $amendment_id})
                        MATCH (a:Article {law_id: $law_id,
                                          article_number: $article_number})
                        MERGE (a)-[r:AMENDED_BY]->(am)
                        SET r.amendment_type = $amendment_type,
                            r.amendment_date = $amendment_date
                    """, amendment_id=amendment.amendment_id,
                        law_id=law_id,
                        article_number=article_number,
                        amendment_type=amendment.amendment_type,
                        amendment_date=amendment.amendment_date)

                with self.driver.session() as session:
                    session.run("""
                        MATCH (l:Law {law_id: $law_id})
                        MATCH (am:Amendment {amendment_id: $amendment_id})
                        MATCH (a_old:Article {law_id: $law_id,
                                              article_number: $article_number})
                        WHERE a_old.article_id <> $new_article_id
                          AND a_old.is_amended IS NULL
                        MERGE (a_new:Article {article_id: $new_article_id})
                        ON CREATE SET
                            a_new.article_number = $article_number,
                            a_new.law_id         = $law_id,
                            a_new.text           = $description,
                            a_new.version        = $amendment_date,
                            a_new.is_amended     = true
                        MERGE (l)-[:CONTAINS]->(a_new)
                        MERGE (a_new)-[:SUPERSEDES {
                            amendment_id:   $amendment_id,
                            amendment_date: $amendment_date
                        }]->(a_old)
                    """, law_id=law_id,
                        amendment_id=amendment.amendment_id,
                        article_number=article_number,
                        new_article_id=new_article_id,
                        description=desc,
                        amendment_date=amendment.amendment_date)

    def import_law(self, bundle: ExtractedLaw, verbose: bool = True) -> Dict[str, int]:
        stats = {"laws": 0, "articles": 0, "penalties": 0, "definitions": 0,
                 "references": 0, "topics": 0, "schedules": 0, "entries": 0,
                 "substances": 0, "amendments": 0}
        law_id = bundle.law_meta.get("law_id")
        if verbose:
            print(f"\n{'=' * 70}\nIMPORTING: {bundle.law_meta.get('title')}\n{'=' * 70}")
        self.create_law(bundle.law_meta)
        stats["laws"] = 1
        for i, article in enumerate(bundle.articles):
            self.create_article(law_id, article.get("article_number"),
                                article.get("text", ""), i)
            stats["articles"] += 1
        for penalty in bundle.penalties:
            self.create_penalty(penalty, law_id)
            stats["penalties"] += 1
        for definition in bundle.definitions:
            self.create_definition(definition, law_id)
            stats["definitions"] += 1
        for topic_name in set(t["topic_name"] for t in bundle.topics):
            self.create_topic(topic_name)
        for topic in bundle.topics:
            self.link_article_to_topic(law_id, topic["article_number"],
                                       topic["topic_name"], topic.get("confidence", 1.0))
            stats["topics"] += 1
        for ref in bundle.references:
            self.create_article_reference(law_id, ref.get("from_article"),
                                          ref.get("to_article"),
                                          ref.get("reference_type", "direct"))
            stats["references"] += 1
        for schedule in bundle.schedules:
            self.create_schedule(law_id, schedule)
            stats["schedules"] += 1
            for i, entry in enumerate(schedule.entries):
                self.create_schedule_entry(schedule.schedule_id, entry, i)
                stats["entries"] += 1
                if entry.chemical_name:
                    stats["substances"] += 1
        if verbose:
            print(f"✓ {stats['articles']} articles | {stats['penalties']} penalties | "
                  f"{stats['schedules']} schedules")
        return stats

    def get_statistics(self) -> Dict[str, Any]:
        with self.driver.session() as session:
            stats: Dict[str, Any] = {"node_counts": {}, "relationship_counts": {}}
            for label in ["Law", "Article", "Penalty", "Definition", "Topic",
                          "Schedule", "ScheduleEntry", "Substance", "Amendment"]:
                result = session.run(f"MATCH (n:{label}) RETURN count(n) AS count")
                stats["node_counts"][label] = result.single()["count"]
            for rel in ["CONTAINS", "REFERENCES", "HAS_PENALTY", "DEFINES",
                        "TAGGED_WITH", "HAS_SCHEDULE", "CONTAINS_ENTRY",
                        "DEFINES_SUBSTANCE", "HAS_AMENDMENT", "AMENDED_BY", "SUPERSEDES"]:
                result = session.run(f"MATCH ()-[r:{rel}]->() RETURN count(r) AS count")
                stats["relationship_counts"][rel] = result.single()["count"]
            return stats

    def print_statistics(self):
        stats = self.get_statistics()
        print("\n" + "=" * 70 + "\nKNOWLEDGE GRAPH STATISTICS\n" + "=" * 70)
        print("\nNodes:")
        for label, count in stats["node_counts"].items():
            print(f"  {label}: {count}")
        print("\nRelationships:")
        for rel, count in stats["relationship_counts"].items():
            print(f"  {rel}: {count}")
        print("\n" + "=" * 70 + "\n")

    def query_amendments(self, law_id: Optional[str] = None) -> List[Dict]:
        with self.driver.session() as session:
            if law_id:
                result = session.run("""
                    MATCH (l:Law {law_id: $law_id})-[:HAS_AMENDMENT]->(am:Amendment)
                    OPTIONAL MATCH (a:Article)-[:AMENDED_BY]->(am)
                    RETURN am, collect(DISTINCT a.article_number) AS affected_articles
                    ORDER BY am.amendment_date
                """, law_id=law_id)
            else:
                result = session.run("""
                    MATCH (l:Law)-[:HAS_AMENDMENT]->(am:Amendment)
                    OPTIONAL MATCH (a:Article)-[:AMENDED_BY]->(am)
                    RETURN l.law_id AS law_id, l.title AS law_title,
                           am, collect(DISTINCT a.article_number) AS affected_articles
                    ORDER BY am.amendment_date
                """)
            amendments = []
            for record in result:
                amendment_data = dict(record["am"])
                amendment_data["affected_articles"] = record["affected_articles"]
                if not law_id:
                    amendment_data["law_id"]    = record["law_id"]
                    amendment_data["law_title"] = record["law_title"]
                amendments.append(amendment_data)
            return amendments

    def query_article_history(self, law_id: str,
                               article_number: str) -> List[Dict]:
        with self.driver.session() as session:
            result = session.run("""
                MATCH (a:Article {law_id: $law_id, article_number: $article_number})
                OPTIONAL MATCH (a)-[:AMENDED_BY]->(am:Amendment)
                OPTIONAL MATCH (a_new:Article)-[:SUPERSEDES]->(a)
                OPTIONAL MATCH (a)-[:SUPERSEDES]->(a_old:Article)
                RETURN
                    a.article_number               AS article,
                    a.version                      AS version,
                    a.is_amended                   AS is_amended,
                    a.is_addition                  AS is_addition,
                    collect(DISTINCT {
                        amendment_id: am.amendment_id,
                        law_num:      am.amendment_law_number,
                        date:         am.amendment_date,
                        type:         am.amendment_type
                    })                             AS amendments,
                    collect(DISTINCT a_new.article_id) AS superseded_by,
                    collect(DISTINCT a_old.article_id) AS supersedes
                ORDER BY a.version
            """, law_id=law_id, article_number=article_number)
            return [dict(r) for r in result]

    def build_from_documents(
        self,
        docs:    List[Document],
        verbose: bool = True,
    ) -> Dict[str, int]:
        """
        Import List[Document] (from src/Chunking/chunking.py) into Neo4j.

        chunk_type routing:
            'article'         → create_article()
            'preamble'        → create_article() with article_number=None
            'amended_article' → create_amendment() → SUPERSEDES chain
        """
        stats: Dict[str, int] = {
            "laws": 0, "articles": 0, "amendments": 0, "skipped": 0}

        seen_laws: set = set()
        for doc in docs:
            m      = doc.metadata
            law_id = m.get("law_id")
            title  = m.get("law_title", law_id)
            if law_id and law_id not in seen_laws:
                self.create_law({"law_id": law_id, "title": title,
                                 "promulgation_date": None,
                                 "source": "chunking_pipeline", "language": "ar"})
                seen_laws.add(law_id)
                stats["laws"] += 1
                if verbose:
                    print(f"  ✓ Law: {law_id} — {title}")

        for position, doc in enumerate(docs):
            m          = doc.metadata
            law_id     = m.get("law_id")
            chunk_type = m.get("chunk_type")
            article_no = m.get("article_number")
            text       = doc.page_content
            if not law_id or not text:
                stats["skipped"] += 1
                continue
            try:
                if chunk_type in ("article", "preamble"):
                    self.create_article(law_id, article_no, text, position)
                    stats["articles"] += 1
                elif chunk_type == "amended_article":
                    law_num    = m.get("amendment_law_number")
                    amend_date = m.get("amendment_date")
                    amend_type = m.get("amendment_type", "modification")
                    if not law_num or not amend_date:
                        self.create_article(law_id, article_no, text, position)
                        stats["articles"] += 1
                        stats["skipped"]  += 1
                        continue
                    year         = amend_date.split("-")[0]
                    amendment_id = _stable_id(law_id, law_num, year)
                    amendment    = Amendment(
                        amendment_id            = amendment_id,
                        amendment_law_number    = law_num,
                        amendment_date          = amend_date,
                        amendment_law_title     = f"قانون رقم {law_num} لسنة {year}",
                        amended_article_numbers = [article_no] if article_no else [],
                        amendment_type          = amend_type,
                        description             = text[:500],
                        effective_date          = amend_date,
                    )
                    self.create_amendment(law_id, amendment)
                    stats["amendments"] += 1
                else:
                    stats["skipped"] += 1
            except Exception as e:
                logger.error(f"  ✗ position={position} law={law_id} "
                             f"article={article_no}: {e}")
                stats["skipped"] += 1

        if verbose:
            print(f"\n  KG import summary:")
            print(f"    Laws       : {stats['laws']}")
            print(f"    Articles   : {stats['articles']}")
            print(f"    Amendments : {stats['amendments']}")
            print(f"    Skipped    : {stats['skipped']}")
        return stats


# =============================================================================
# SECTION 10: PIPELINE RUNNERS
# =============================================================================

def import_separate_amendments(graph: EnhancedLegalKnowledgeGraph,
                               amendments_by_law: Dict[str, List[Amendment]],
                               verbose: bool = True) -> int:
    total = 0
    for law_id, amendments in amendments_by_law.items():
        if verbose:
            print(f"\n{'=' * 70}\nIMPORTING AMENDMENTS → {law_id}\n{'=' * 70}")
        for amendment in amendments:
            try:
                graph.create_amendment(law_id, amendment)
                total += 1
                if verbose:
                    print(f"  ✓ Law {amendment.amendment_law_number}/"
                          f"{amendment.amendment_date[:4]} "
                          f"| type={amendment.amendment_type} "
                          f"| articles={amendment.amended_article_numbers}")
            except Exception as e:
                logger.error(f"  ✗ {amendment.amendment_id}: {e}")
                if verbose:
                    traceback.print_exc()
    if verbose:
        print(f"\n✓ Total amendments imported: {total}")
    return total


def run_file_based_pipeline(
    root_dir:          str,
    neo4j_uri:         str,
    neo4j_user:        str,
    neo4j_password:    str,
    folder_to_law_key: Optional[Dict[str, str]] = None,
    drop_existing:     bool = True,
    verbose:           bool = True,
) -> EnhancedLegalKnowledgeGraph:
    """
    Classic file-based pipeline. Reads files directly — does NOT require chunking.py.

    Phase 1: extract & summarise all laws.
    Phase 2: import main law files into Neo4j.
    Phase 3: import amendment (new*) files into Neo4j.
    """
    if folder_to_law_key is None:
        folder_to_law_key = DEFAULT_FOLDER_TO_LAW_KEY

    print("\n" + "=" * 70 + "\nPHASE 1: EXTRACTION SUMMARY\n" + "=" * 70)
    summaries  = ingest_dataset(root_dir, folder_to_law_key)
    df         = pd.DataFrame(summaries)
    print(df.to_string())
    successful = [s for s in summaries if s.get("error") is None]
    print(f"\n✓ Extracted {len(successful)}/{len(summaries)} laws\n")

    debug_amendment_files(root_dir, folder_to_law_key)

    graph = EnhancedLegalKnowledgeGraph(neo4j_uri, neo4j_user, neo4j_password)
    graph.setup_schema(drop_existing=drop_existing)

    for summary in successful:
        law_key     = summary["law_key"]
        folder_path = Path(root_dir) / summary["folder"]
        raw_text    = read_all_txt_in_folder(folder_path, exclude_pattern='new*')
        extractor   = get_extractor(law_key, raw_text)
        bundle      = extractor.extract_all()
        graph.import_law(bundle, verbose=verbose)

    print("\n" + "=" * 70 + "\nPHASE 3: IMPORTING AMENDMENTS FROM new* FILES\n" + "=" * 70)
    amendments_by_law = ingest_amendment_files(root_dir, folder_to_law_key)
    import_separate_amendments(graph, amendments_by_law, verbose=verbose)

    print("\n✅ BUILD COMPLETE")
    graph.print_statistics()
    return graph


def build_knowledge_graph(
    docs:           List[Document],
    neo4j_uri:      str,
    neo4j_user:     str,
    neo4j_password: str,
    drop_existing:  bool = False,
    verbose:        bool = True,
) -> EnhancedLegalKnowledgeGraph:
    """
    Document-based pipeline.
    Receives List[Document] from src/Chunking/chunking.py.

    Usage:
        from src.Chunking.chunking import build_rag_documents
        from src.KG.knowledge_graph_builder import build_knowledge_graph

        docs  = build_rag_documents("Datasets/Unstructured_Data")
        graph = build_knowledge_graph(docs, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        graph.print_statistics()
        graph.close()
    """
    print("\n" + "=" * 70 + "\nBUILDING KNOWLEDGE GRAPH FROM DOCUMENTS\n" + "=" * 70)
    print(f"  Input documents : {len(docs):,}")
    graph = EnhancedLegalKnowledgeGraph(neo4j_uri, neo4j_user, neo4j_password)
    graph.setup_schema(drop_existing=drop_existing)
    graph.build_from_documents(docs, verbose=verbose)
    graph.print_statistics()
    return graph


# =============================================================================
# SECTION 11: STANDALONE ENTRY POINT
# =============================================================================

def run_KG():
    ROOT_DIR = DataPath
    try:
        from src.Config.config import NEO4J_PASSWORD, NEO4J_USERNAME, NEO4J_URI
    except Exception as e:
        print(e)
        return

    graph = run_file_based_pipeline(
        root_dir       = ROOT_DIR,
        neo4j_uri      = NEO4J_URI,
        neo4j_user     = NEO4J_USERNAME,
        neo4j_password = NEO4J_PASSWORD,
        drop_existing  = True,
        verbose        = True,
    )
    graph.close()