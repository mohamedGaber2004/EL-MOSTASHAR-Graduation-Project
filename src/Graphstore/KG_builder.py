from __future__ import annotations

import fnmatch
import logging
import re , hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from langchain_core.documents import Document
from neo4j import GraphDatabase

from src.Config.config import DataPath
from src.Utils.regex_utils import (
    _to_western_digits,
    ORIGINAL_LAW_RE , 
    _DATE_RE , 
    _MONTH_MAP
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

ENCODINGS = ("utf-8", "utf-16", "cp1256", "windows-1256")

# =============================================================================
# REGEX
# =============================================================================


PENALTY_PATTERNS = {
    'سجن': re.compile(
        r'(?:السجن|بالسجن|سجن[اً]?)\s*(?:مدة|لمدة)?\s*(?:(?:من|لا\s+تقل\s+عن)\s*)?([0-9٠-٩]+)\s*(?:إلى|حتى|-)\s*([0-9٠-٩]+)\s*(سنة|سنوات|عام|أعوام)',
        re.UNICODE),
    'غرامة': re.compile(
        r'(?:غرامة|بغرامة)\s*(?:مالية)?\s*(?:لا\s+)?(?:تقل\s+عن|من)\s*([0-9٠-٩,]+)\s*(?:جنيه|دولار|ريال)?(?:\s*(?:ولا\s+تزيد\s+على|إلى|حتى|-)\s*([0-9٠-٩,]+))?',
        re.UNICODE),
    'إعدام':       re.compile(r'(?:الإعدام|بالإعدام|القتل|قتل[اً]?|إعدام[اً]?)', re.UNICODE),
    'أشغال_شاقة':  re.compile(r'الأشغال\s+الشاقة', re.UNICODE),
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

DEFAULT_FOLDER_TO_LAW_KEY: Dict[str, str] = {
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

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ScheduleEntry:
    entry_number:  str
    arabic_name:   Optional[str]       = None
    english_name:  Optional[str]       = None
    chemical_name: Optional[str]       = None
    description:   Optional[str]       = None
    trade_names:   Optional[List[str]] = None

@dataclass
class Schedule:
    schedule_id:     str
    schedule_number: str
    title:           str
    entries:         List[ScheduleEntry] = field(default_factory=list)

@dataclass
class Amendment:
    amendment_id:            str
    amendment_law_number:    Optional[str]  = None
    amendment_law_title:     Optional[str]  = None
    amendment_date:          Optional[str]  = None
    amendment_type:          Optional[str]  = None
    amended_article_numbers: List[str]      = field(default_factory=list)
    description:             Optional[str]  = None
    effective_date:          Optional[str]  = None

@dataclass
class ExtractedLaw:
    law_meta:    Dict[str, Any]
    articles:    List[Dict[str, Any]]
    penalties:   List[Dict[str, Any]]
    definitions: List[Dict[str, Any]]
    references:  List[Dict[str, Any]]
    topics:      List[Dict[str, Any]]
    amendments:  List[Amendment]      = field(default_factory=list)
    schedules:   List[Schedule]       = field(default_factory=list)


# =============================================================================
# HELPERS
# =============================================================================

def _normalize_article_no(token: str) -> str:
    token = _to_western_digits(token.strip())
    token = re.sub(r"\s+", " ", token)
    token = re.sub(r"مكر(?:ر[اأإآىًٍَُِّْ]?)", "مكرر", token)
    return token.strip()


def _stable_id(*components) -> str:
    combined = "|".join(str(c) for c in components)
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()[:16]

def _read_file(fp: Path) -> Optional[str]:
    for enc in ENCODINGS:
        try:
            return fp.read_text(encoding=enc)
        except Exception:
            continue
    return fp.read_bytes().decode("utf-8", errors="replace")


def _dedup(items: List[Dict], keys: List[str]) -> List[Dict]:
    seen, out = set(), []
    for item in items:
        k = tuple(item.get(f) for f in keys)
        if k not in seen:
            seen.add(k)
            out.append(item)
    return out


def _normalize_arabic(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"[\u0617-\u061A\u064B-\u0652\u0670]", "", text).replace("\u0640", "")
    for old, new in [("أ","ا"),("إ","ا"),("آ","ا"),("ى","ي"),("ئ","ي"),("ة","ه"),("ؤ","و")]:
        text = text.replace(old, new)
    return re.sub(r"\s+", " ", text).strip()


# =============================================================================
# SCHEDULE EXTRACTORS
# =============================================================================

def _extract_schedule_entries_drug(text: str) -> List[ScheduleEntry]:
    entries, sections = [], re.split(r'\n(\d+)\s*[–\-]\s*', text)
    for i in range(1, len(sections), 2):
        if i + 1 >= len(sections):
            break
        body = sections[i + 1].strip()
        nm = re.match(r'^([^(]+)(?:\(([^)]+)\))?', body.split('\n')[0])
        cm = re.search(r'الاسم الكيميائي:\s*(.+?)(?=\n|$)', body, re.UNICODE)
        tm = re.search(r'الأسماء التجارية:\s*(.+?)(?=\n---|\n\d+|$)', body, re.UNICODE)
        entries.append(ScheduleEntry(
            entry_number = sections[i].strip(),
            arabic_name  = nm.group(1).strip() if nm else None,
            english_name = nm.group(2).strip() if nm and nm.group(2) else None,
            chemical_name= cm.group(1).strip() if cm else None,
            trade_names  = [n.strip() for n in re.split(r'[،,\-–]', tm.group(1)) if n.strip()] if tm else None,
        ))
    return entries


def _extract_schedule_entries_weapon(text: str) -> List[ScheduleEntry]:
    return [
        ScheduleEntry(entry_number=m.group(1), arabic_name=m.group(2).strip(), description=m.group(2).strip())
        for line in text.split('\n')
        if (m := re.match(r'^(\d+)\s*[–\-]\s*(.+)$', line.strip()))
    ]


def _extract_schedules(text: str, prefix: str, entry_fn) -> List[Schedule]:
    HEADER_RE = re.compile(r'(?:الجدول|جدول)\s+رقم\s*\(?([0-9٠-٩]+)\)?', re.UNICODE)
    matches   = list(HEADER_RE.finditer(text))
    schedules = []
    for i, m in enumerate(matches):
        num   = _to_western_digits(m.group(1))
        start = m.end()
        end   = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk = text[start:end].strip()
        lines = [l.strip() for l in chunk.split('\n') if l.strip()]
        schedules.append(Schedule(
            schedule_id     = f"{prefix}_schedule_{num}",
            schedule_number = num,
            title           = lines[0] if lines else f"جدول رقم {num}",
            entries         = entry_fn(chunk),
        ))
    return schedules


# =============================================================================
# LAW EXTRACTOR
# =============================================================================

# law_key → (title, promulgation_date, schedule_fn)
_LAW_REGISTRY: Dict[str, Tuple[str, Optional[str], Optional[Any]]] = {
    "penal_code":            ("قانون العقوبات",                              "1937-07-31", None),
    "money_laundering":      ("قانون مكافحة غسل الأموال",                   "2002-05-15", None),
    "weapons_ammunition":    ("قانون الأسلحة والذخيرة",                     "1954-09-22", lambda t: _extract_schedules(t, "weapon", _extract_schedule_entries_weapon)),
    "criminal_constitution": ("الدستور الجنائي",                            None,         None),
    "anti_drugs":            ("قانون مكافحة المخدرات",                      "1960-05-01", lambda t: _extract_schedules(t, "drug",   _extract_schedule_entries_drug)),
    "criminal_procedure":    ("قانون الإجراءات الجنائية",                   "1950-10-18", None),
    "anti_terror":           ("قانون مكافحة الإرهاب",                       "2015-08-16", None),
    "emergency_law":         ("قانون الطوارئ",                              "1958-01-01", None),
    "cybercrime":            ("قانون مكافحة جرائم تقنية المعلومات",         "2018-07-17", None),
}

TOPICS_MAP = {
    "قتل":    ["قتل", "قاتل"],
    "سرقة":   ["سرقة", "سارق"],
    "مخدرات": ["مخدر", "مخدرات"],
    "أسلحة":  ["سلاح", "أسلحة"],
    "إرهاب":  ["إرهاب", "إرهابي"],
}


class LawExtractor:
    """Extract all structured data from a single law text."""

    def __init__(self, law_key: str, raw_text: str):
        if law_key not in _LAW_REGISTRY:
            raise KeyError(f"Unknown law_key: '{law_key}'")
        title, date, schedule_fn = _LAW_REGISTRY[law_key]
        self.law_key     = law_key
        self.raw_text    = raw_text or ""
        self.title       = title
        self.date        = date
        self._schedule_fn = schedule_fn

    def _articles(self) -> List[Dict]:
        text    = self.raw_text
        matches = list(_ARTICLE_MARK_RE.finditer(text))
        if not matches:
            return [{"article_number": None, "text": text.strip()}]
        articles = []
        if matches[0].start() > 0 and (pre := text[:matches[0].start()].strip()):
            articles.append({"article_number": None, "text": pre})
        for i, m in enumerate(matches):
            end  = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[m.end():end].strip()
            num  = _normalize_article_no(m.group("num") or "")
            articles.append({"article_number": num, "text": body})
        return articles

    def _penalties(self, articles: List[Dict]) -> List[Dict]:
        out = []
        for a in articles:
            no, text = a.get("article_number"), a.get("text", "")
            if not no:
                continue
            for m in PENALTY_PATTERNS['سجن'].finditer(text):
                out.append({"article_number": no, "penalty_type": "سجن",
                            "min_value": int(_to_western_digits(m.group(1))),
                            "max_value": int(_to_western_digits(m.group(2))), "unit": "سنة"})
            for m in PENALTY_PATTERNS['غرامة'].finditer(text):
                out.append({"article_number": no, "penalty_type": "غرامة",
                            "min_value": int(_to_western_digits(m.group(1).replace(',', ''))) if m.group(1) else None,
                            "max_value": int(_to_western_digits(m.group(2).replace(',', ''))) if m.group(2) else None,
                            "unit": "جنيه"})
            if PENALTY_PATTERNS['إعدام'].search(text):
                out.append({"article_number": no, "penalty_type": "إعدام", "min_value": None, "max_value": None, "unit": None})
            if PENALTY_PATTERNS['أشغال_شاقة'].search(text):
                out.append({"article_number": no, "penalty_type": "أشغال_شاقة", "min_value": None, "max_value": None, "unit": None})
        return _dedup(out, ["article_number", "penalty_type", "min_value", "max_value"])

    def _definitions(self, articles: List[Dict]) -> List[Dict]:
        out = []
        for a in articles:
            no = a.get("article_number")
            if not no:
                continue
            for m in DEF_RE.finditer(a.get("text", "")):
                out.append({"term": m.group(1).strip(), "definition_text": m.group(2).strip(), "defined_in_article": no})
        return _dedup(out, ["term", "defined_in_article"])

    def _references(self, articles: List[Dict]) -> List[Dict]:
        refs = set()
        for a in articles:
            from_no = str(a.get("article_number", "")).strip()
            if not from_no or from_no == "None":
                continue
            for m in REF_RE.finditer(a.get("text", "")):
                for to_no in [m.group("num"), m.group("num2")]:
                    if to_no and (to_no := _normalize_article_no(to_no)) and to_no != from_no:
                        refs.add((from_no, to_no))
        return [{"from_article": f, "to_article": t, "reference_type": "direct"} for f, t in sorted(refs)]

    def _topics(self, articles: List[Dict]) -> List[Dict]:
        out = []
        for a in articles:
            no = a.get("article_number")
            if not no:
                continue
            text = _normalize_arabic(a.get("text", ""))
            for topic, keywords in TOPICS_MAP.items():
                if any(_normalize_arabic(kw) in text for kw in keywords):
                    out.append({"article_number": no, "topic_name": topic, "confidence": 0.8})
                    break
        return _dedup(out, ["article_number", "topic_name"])

    def extract(self) -> ExtractedLaw:
        articles = self._articles()
        return ExtractedLaw(
            law_meta    = {"law_id": self.law_key, "title": self.title,
                           "promulgation_date": self.date, "effective_date": self.date,
                           "source": "الجريدة الرسمية", "language": "ar"},
            articles    = articles,
            penalties   = self._penalties(articles),
            definitions = self._definitions(articles),
            references  = self._references(articles),
            topics      = self._topics(articles),
            schedules   = self._schedule_fn(self.raw_text) if self._schedule_fn else [],
        )


# =============================================================================
# AMENDMENT EXTRACTOR
# =============================================================================

class AmendmentExtractor:

    def __init__(self, raw_text: str, filename: Optional[str] = None):
        self.raw_text = raw_text
        self.filename = filename

    def _law_num_year(self) -> Tuple[Optional[str], Optional[str]]:
        for m in ORIGINAL_LAW_RE.finditer(self.raw_text):
            try:
                a, b = int(_to_western_digits(m.group(1))), int(_to_western_digits(m.group(2)))
                year, num = (str(a), str(b)) if 1800 <= a <= 2100 else (str(b), str(a))
                if 1800 <= int(year) <= 2100 and 0 < int(num) < 10000:
                    return num, year
            except Exception:
                continue
        return None, None

    def _article_numbers(self) -> List[str]:
        seen, ordered = set(), []
        for m in _ANY_ARTICLE_RE.finditer(self.raw_text):
            norm = _normalize_article_no(m.group(1))
            if norm and norm not in seen:
                seen.add(norm)
                ordered.append(norm)
        return ordered

    def _amendment_type(self) -> str:
        sample = self.raw_text[:800]
        if any(w in sample for w in ['تضاف','يضاف','أضيف','إضافة','جديدة','جديد']):
            return 'addition'
        if any(w in sample for w in ['يلغى','ألغي','يحذف','حذف','إلغاء']):
            return 'deletion'
        return 'modification'

    def _amendment_date(self, year: str) -> str:
        m = _DATE_RE.search(self.raw_text)
        if m:
            day   = _to_western_digits(m.group(1) or '01').zfill(2)
            month = _MONTH_MAP.get(m.group(2), '01')
            yr    = _to_western_digits(m.group(3))
            return f"{yr}-{month}-{day}"
        return f"{year}-01-01"

    def extract(self, target_law_id: str) -> Optional[Amendment]:
        law_num, year = self._law_num_year()
        if not law_num or not year:
            logger.warning(f"[SKIP] No law number/year | {self.filename}")
            return None
        articles = self._article_numbers()
        if not articles:
            logger.warning(f"[SKIP] No articles found | {self.filename}")
            return None
        atype  = self._amendment_type()
        adate  = self._amendment_date(year)
        desc_m = re.search(r'المادة الأولى\s*[:：]?\s*(.*?)(?=المادة الثانية|$)', self.raw_text, re.DOTALL)
        return Amendment(
            amendment_id            = _stable_id(target_law_id, law_num, year),
            amendment_law_number    = law_num,
            amendment_law_title     = f"قانون رقم {law_num} لسنة {year}",
            amendment_date          = adate,
            amendment_type          = atype,
            amended_article_numbers = articles,
            description             = (desc_m.group(1).strip()[:500] if desc_m else self.raw_text[:500]),
            effective_date          = adate,
        )

# =============================================================================
# INGESTION HELPERS
# =============================================================================

def _read_law_folder(folder: Path) -> str:
    parts = []
    for fp in sorted(f for f in folder.glob("*.txt") if not fnmatch.fnmatch(f.name.lower(), "new*")):
        text = _read_file(fp)
        if text:
            parts.append(text)
    return "\n\n".join(parts).strip()


def ingest_dataset(root_dir: str, folder_to_law_key: Dict[str, str]) -> List[Dict]:
    summaries = []
    for folder in sorted(p for p in Path(root_dir).iterdir() if p.is_dir()):
        if folder.name not in folder_to_law_key:
            continue
        law_key = folder_to_law_key[folder.name]
        try:
            raw = _read_law_folder(folder)
            if not raw:
                summaries.append({"folder": folder.name, "law_key": law_key, "error": "empty"})
                continue
            bundle = LawExtractor(law_key, raw).extract()
            summaries.append({"folder": folder.name, "law_key": law_key,
                               "articles": len(bundle.articles), "penalties": len(bundle.penalties),
                               "definitions": len(bundle.definitions), "references": len(bundle.references),
                               "schedules": len(bundle.schedules), "error": None})
        except Exception as e:
            summaries.append({"folder": folder.name, "law_key": law_key, "error": str(e)})
    return summaries


def ingest_amendments(root_dir: str, folder_to_law_key: Dict[str, str]) -> Dict[str, List[Amendment]]:
    result: Dict[str, List[Amendment]] = {}
    for folder in sorted(p for p in Path(root_dir).iterdir() if p.is_dir()):
        if folder.name not in folder_to_law_key:
            continue
        law_key, law_amendments = folder_to_law_key[folder.name], []
        for af in sorted(folder.glob("new*.txt")):
            text = _read_file(af)
            if not text or not text.strip():
                continue
            amendment = AmendmentExtractor(text, af.name).extract(law_key)
            if amendment:
                law_amendments.append(amendment)
        if law_amendments:
            result[law_key] = law_amendments
            logger.info(f"  '{folder.name}' → {len(law_amendments)} amendment(s)")
    return result


# =============================================================================
# KNOWLEDGE GRAPH
# =============================================================================

class LegalKnowledgeGraph:

    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.driver.verify_connectivity()
        logger.info("✓ Neo4j connected")

    def close(self):
        self.driver.close()

    def setup_schema(self, drop_existing: bool = False):
        with self.driver.session() as s:
            if drop_existing:
                s.run("MATCH (n) DETACH DELETE n")
            for stmt in [
                "CREATE CONSTRAINT law_id_unique        IF NOT EXISTS FOR (l:Law)          REQUIRE l.law_id IS UNIQUE",
                "CREATE CONSTRAINT article_id_unique    IF NOT EXISTS FOR (a:Article)       REQUIRE a.article_id IS UNIQUE",
                "CREATE CONSTRAINT penalty_id_unique    IF NOT EXISTS FOR (p:Penalty)       REQUIRE p.penalty_id IS UNIQUE",
                "CREATE CONSTRAINT definition_id_unique IF NOT EXISTS FOR (d:Definition)    REQUIRE d.definition_id IS UNIQUE",
                "CREATE CONSTRAINT topic_id_unique      IF NOT EXISTS FOR (t:Topic)         REQUIRE t.topic_id IS UNIQUE",
                "CREATE CONSTRAINT schedule_id_unique   IF NOT EXISTS FOR (s:Schedule)      REQUIRE s.schedule_id IS UNIQUE",
                "CREATE CONSTRAINT entry_id_unique      IF NOT EXISTS FOR (e:ScheduleEntry) REQUIRE e.entry_id IS UNIQUE",
                "CREATE CONSTRAINT substance_id_unique  IF NOT EXISTS FOR (sub:Substance)   REQUIRE sub.substance_id IS UNIQUE",
                "CREATE CONSTRAINT amendment_id_unique  IF NOT EXISTS FOR (am:Amendment)    REQUIRE am.amendment_id IS UNIQUE",
                "CREATE INDEX supersedes_date_idx IF NOT EXISTS FOR ()-[r:SUPERSEDES]-() ON (r.amendment_date)",
            ]:
                try:
                    s.run(stmt)
                except Exception:
                    pass
        logger.info("✓ Schema ready")

    def _run(self, query: str, **params):
        with self.driver.session() as s:
            s.run(query, **params)

    def create_law(self, meta: Dict[str, Any]):
        self._run("""
            MERGE (l:Law {law_id: $law_id})
            SET l.title=$title, l.promulgation_date=$promulgation_date,
                l.source=$source, l.language=$language
        """, law_id=meta.get("law_id"), title=meta.get("title"),
            promulgation_date=meta.get("promulgation_date"),
            source=meta.get("source"), language=meta.get("language", "ar"))

    def create_article(self, law_id: str, article_number: Optional[str], text: str, position: int):
        article_id = _stable_id(law_id, article_number or "preamble", position)
        self._run("""
            MATCH (l:Law {law_id: $law_id})
            MERGE (a:Article {article_id: $article_id})
            SET a.article_number=$article_number, a.text=$text, a.law_id=$law_id
            MERGE (l)-[r:CONTAINS]->(a) SET r.position=$position
        """, law_id=law_id, article_id=article_id,
            article_number=article_number, text=text, position=position)

    def create_penalty(self, penalty: Dict, law_id: str):
        self._run("""
            MATCH (a:Article {law_id: $law_id, article_number: $article_number})
            MERGE (p:Penalty {penalty_id: $penalty_id})
            SET p.penalty_type=$penalty_type, p.min_value=$min_value,
                p.max_value=$max_value, p.unit=$unit
            MERGE (a)-[:HAS_PENALTY]->(p)
        """, law_id=law_id,
            penalty_id=_stable_id(law_id, penalty.get("article_number"), penalty.get("penalty_type")),
            article_number=penalty.get("article_number"), penalty_type=penalty.get("penalty_type"),
            min_value=penalty.get("min_value"), max_value=penalty.get("max_value"), unit=penalty.get("unit"))

    def create_definition(self, definition: Dict, law_id: str):
        self._run("""
            MATCH (a:Article {law_id: $law_id, article_number: $article_number})
            MERGE (d:Definition {definition_id: $def_id})
            SET d.term=$term, d.definition_text=$definition_text
            MERGE (a)-[:DEFINES]->(d)
        """, law_id=law_id, def_id=_stable_id(law_id, definition.get("term")),
            article_number=definition.get("defined_in_article"),
            term=definition.get("term"), definition_text=definition.get("definition_text"))

    def create_topic(self, topic_name: str):
        self._run("MERGE (t:Topic {topic_id: $id, name: $name})",
                  id=_stable_id(topic_name), name=topic_name)

    def link_article_topic(self, law_id: str, article_number: str, topic_name: str, confidence: float):
        self._run("""
            MATCH (a:Article {law_id: $law_id, article_number: $article_number})
            MATCH (t:Topic {name: $topic_name})
            MERGE (a)-[r:TAGGED_WITH]->(t) SET r.confidence=$confidence
        """, law_id=law_id, article_number=article_number, topic_name=topic_name, confidence=confidence)

    def create_reference(self, law_id: str, from_article: str, to_article: str):
        self._run("""
            MATCH (from:Article {law_id: $law_id, article_number: $from_article})
            MATCH (to:Article   {law_id: $law_id, article_number: $to_article})
            MERGE (from)-[r:REFERENCES]->(to) SET r.reference_type='direct'
        """, law_id=law_id, from_article=from_article, to_article=to_article)

    def create_schedule(self, law_id: str, schedule: Schedule):
        self._run("""
            MATCH (l:Law {law_id: $law_id})
            MERGE (s:Schedule {schedule_id: $schedule_id})
            SET s.schedule_number=$schedule_number, s.title=$title, s.entry_count=$count
            MERGE (l)-[:HAS_SCHEDULE]->(s)
        """, law_id=law_id, schedule_id=schedule.schedule_id,
            schedule_number=schedule.schedule_number, title=schedule.title, count=len(schedule.entries))

    def create_schedule_entry(self, schedule_id: str, entry: ScheduleEntry, position: int):
        entry_id = _stable_id(schedule_id, entry.entry_number, position)
        self._run("""
            MATCH (s:Schedule {schedule_id: $schedule_id})
            MERGE (e:ScheduleEntry {entry_id: $entry_id})
            SET e.entry_number=$entry_number, e.arabic_name=$arabic_name,
                e.english_name=$english_name, e.description=$description
            MERGE (s)-[r:CONTAINS_ENTRY]->(e) SET r.position=$position
        """, schedule_id=schedule_id, entry_id=entry_id, entry_number=entry.entry_number,
            arabic_name=entry.arabic_name, english_name=entry.english_name,
            description=entry.description, position=position)
        if entry.chemical_name:
            sub_id = _stable_id("substance", entry.arabic_name or entry.english_name)
            self._run("""
                MATCH (e:ScheduleEntry {entry_id: $entry_id})
                MERGE (sub:Substance {substance_id: $sub_id})
                SET sub.arabic_name=$arabic_name, sub.english_name=$english_name,
                    sub.chemical_name=$chemical_name, sub.trade_names=$trade_names
                MERGE (e)-[:DEFINES_SUBSTANCE]->(sub)
            """, entry_id=entry_id, sub_id=sub_id, arabic_name=entry.arabic_name,
                english_name=entry.english_name, chemical_name=entry.chemical_name,
                trade_names=entry.trade_names or [])

    def create_amendment(self, law_id: str, amendment: Amendment):
        desc = (amendment.description or "")[:500]
        self._run("""
            MATCH (l:Law {law_id: $law_id})
            MERGE (am:Amendment {amendment_id: $amendment_id})
            SET am.amendment_law_number=$law_num, am.amendment_law_title=$law_title,
                am.amendment_date=$date, am.amendment_type=$atype,
                am.description=$desc, am.effective_date=$effective_date
            MERGE (l)-[:HAS_AMENDMENT]->(am)
        """, law_id=law_id, amendment_id=amendment.amendment_id,
            law_num=amendment.amendment_law_number, law_title=amendment.amendment_law_title,
            date=amendment.amendment_date, atype=amendment.amendment_type,
            desc=desc, effective_date=amendment.effective_date)

        for article_number in amendment.amended_article_numbers:
            if not article_number:
                continue
            new_id = _stable_id(law_id, article_number,
                                "added" if amendment.amendment_type == "addition" else "amended",
                                amendment.amendment_date)
            if amendment.amendment_type == "addition":
                self._run("""
                    MATCH (l:Law {law_id: $law_id})
                    MATCH (am:Amendment {amendment_id: $amendment_id})
                    MERGE (a_new:Article {article_id: $new_id})
                    ON CREATE SET a_new.article_number=$article_number, a_new.law_id=$law_id,
                                  a_new.text=$desc, a_new.version=$date, a_new.is_addition=true
                    MERGE (l)-[:CONTAINS]->(a_new)
                    MERGE (a_new)-[:AMENDED_BY {amendment_type: $atype, amendment_date: $date}]->(am)
                """, law_id=law_id, amendment_id=amendment.amendment_id, new_id=new_id,
                    article_number=article_number, desc=desc,
                    date=amendment.amendment_date, atype=amendment.amendment_type)
                self._run("""
                    MATCH (a_new:Article {article_id: $new_id})
                    MATCH (a_old:Article {law_id: $law_id, article_number: $article_number})
                    WHERE a_old.article_id <> $new_id AND a_old.is_addition IS NULL
                    MERGE (a_new)-[:SUPERSEDES {amendment_id: $amendment_id, amendment_date: $date}]->(a_old)
                """, new_id=new_id, law_id=law_id, article_number=article_number,
                    amendment_id=amendment.amendment_id, date=amendment.amendment_date)
            else:
                self._run("""
                    MATCH (am:Amendment {amendment_id: $amendment_id})
                    MATCH (a:Article {law_id: $law_id, article_number: $article_number})
                    MERGE (a)-[r:AMENDED_BY]->(am)
                    SET r.amendment_type=$atype, r.amendment_date=$date
                """, amendment_id=amendment.amendment_id, law_id=law_id,
                    article_number=article_number, atype=amendment.amendment_type, date=amendment.amendment_date)
                self._run("""
                    MATCH (l:Law {law_id: $law_id})
                    MATCH (am:Amendment {amendment_id: $amendment_id})
                    MATCH (a_old:Article {law_id: $law_id, article_number: $article_number})
                    WHERE a_old.article_id <> $new_id AND a_old.is_amended IS NULL
                    MERGE (a_new:Article {article_id: $new_id})
                    ON CREATE SET a_new.article_number=$article_number, a_new.law_id=$law_id,
                                  a_new.text=$desc, a_new.version=$date, a_new.is_amended=true
                    MERGE (l)-[:CONTAINS]->(a_new)
                    MERGE (a_new)-[:SUPERSEDES {amendment_id: $amendment_id, amendment_date: $date}]->(a_old)
                """, law_id=law_id, amendment_id=amendment.amendment_id,
                    article_number=article_number, new_id=new_id,
                    desc=desc, date=amendment.amendment_date)

    def import_law(self, bundle: ExtractedLaw):
        law_id = bundle.law_meta["law_id"]
        self.create_law(bundle.law_meta)
        for i, a in enumerate(bundle.articles):
            self.create_article(law_id, a.get("article_number"), a.get("text", ""), i)
        for p in bundle.penalties:
            self.create_penalty(p, law_id)
        for d in bundle.definitions:
            self.create_definition(d, law_id)
        for topic_name in set(t["topic_name"] for t in bundle.topics):
            self.create_topic(topic_name)
        for t in bundle.topics:
            self.link_article_topic(law_id, t["article_number"], t["topic_name"], t.get("confidence", 1.0))
        for r in bundle.references:
            self.create_reference(law_id, r["from_article"], r["to_article"])
        for sched in bundle.schedules:
            self.create_schedule(law_id, sched)
            for i, entry in enumerate(sched.entries):
                self.create_schedule_entry(sched.schedule_id, entry, i)
        logger.info(f"  ✓ {law_id}: {len(bundle.articles)} articles | {len(bundle.penalties)} penalties")

    def import_amendments(self, law_id: str, amendments: List[Amendment]):
        for am in amendments:
            try:
                self.create_amendment(law_id, am)
            except Exception as e:
                logger.error(f"  ✗ {am.amendment_id}: {e}")

    def build_from_documents(self, docs: List[Document]) -> Dict[str, int]:
        """Import List[Document] from chunking.py into Neo4j."""
        stats = {"laws": 0, "articles": 0, "amendments": 0, "skipped": 0}
        seen_laws: set = set()
        for doc in docs:
            law_id = doc.metadata.get("law_id")
            if law_id and law_id not in seen_laws:
                self.create_law({"law_id": law_id, "title": doc.metadata.get("law_title", law_id),
                                 "promulgation_date": None, "source": "chunking_pipeline", "language": "ar"})
                seen_laws.add(law_id)
                stats["laws"] += 1
        for pos, doc in enumerate(docs):
            m, law_id, text = doc.metadata, doc.metadata.get("law_id"), doc.page_content
            if not law_id or not text:
                stats["skipped"] += 1
                continue
            try:
                chunk_type = m.get("chunk_type")
                if chunk_type in ("article", "preamble"):
                    self.create_article(law_id, m.get("article_number"), text, pos)
                    stats["articles"] += 1
                elif chunk_type == "amended_article":
                    law_num, adate, atype = m.get("amendment_law_number"), m.get("amendment_date"), m.get("amendment_type", "modification")
                    if not law_num or not adate:
                        stats["skipped"] += 1
                        continue
                    year = adate.split("-")[0]
                    self.create_amendment(law_id, Amendment(
                        amendment_id=_stable_id(law_id, law_num, year),
                        amendment_law_number=law_num, amendment_date=adate,
                        amendment_law_title=f"قانون رقم {law_num} لسنة {year}",
                        amended_article_numbers=[m["article_number"]] if m.get("article_number") else [],
                        amendment_type=atype, description=text[:500], effective_date=adate,
                    ))
                    stats["amendments"] += 1
                else:
                    stats["skipped"] += 1
            except Exception as e:
                logger.error(f"  ✗ pos={pos} law={law_id}: {e}")
                stats["skipped"] += 1
        logger.info(f"✓ KG import: {stats}")
        return stats

    def get_statistics(self) -> Dict[str, Any]:
        stats: Dict[str, Any] = {"nodes": {}, "relationships": {}}
        with self.driver.session() as s:
            for label in ["Law","Article","Penalty","Definition","Topic","Schedule","ScheduleEntry","Substance","Amendment"]:
                stats["nodes"][label] = s.run(f"MATCH (n:{label}) RETURN count(n) AS c").single()["c"]
            for rel in ["CONTAINS","REFERENCES","HAS_PENALTY","DEFINES","TAGGED_WITH","HAS_SCHEDULE","CONTAINS_ENTRY","DEFINES_SUBSTANCE","HAS_AMENDMENT","AMENDED_BY","SUPERSEDES"]:
                stats["relationships"][rel] = s.run(f"MATCH ()-[r:{rel}]->() RETURN count(r) AS c").single()["c"]
        return stats

    def print_statistics(self):
        stats = self.get_statistics()
        print("\n" + "="*60 + "\nKNOWLEDGE GRAPH STATISTICS")
        print("\nNodes:")
        for k, v in stats["nodes"].items():
            print(f"  {k}: {v}")
        print("\nRelationships:")
        for k, v in stats["relationships"].items():
            print(f"  {k}: {v}")


# =============================================================================
# PIPELINE
# =============================================================================

def run_pipeline(
    root_dir:          str,
    neo4j_uri:         str,
    neo4j_user:        str,
    neo4j_password:    str,
    folder_to_law_key: Optional[Dict[str, str]] = None,
    drop_existing:     bool = True,
) -> LegalKnowledgeGraph:

    folder_to_law_key = folder_to_law_key or DEFAULT_FOLDER_TO_LAW_KEY

    print("\nPHASE 1: EXTRACTION SUMMARY")
    summaries = ingest_dataset(root_dir, folder_to_law_key)
    print(pd.DataFrame(summaries).to_string())

    graph = LegalKnowledgeGraph(neo4j_uri, neo4j_user, neo4j_password)
    graph.setup_schema(drop_existing=drop_existing)

    print("\nPHASE 2: IMPORTING LAWS")
    for s in (s for s in summaries if not s.get("error")):
        raw    = _read_law_folder(Path(root_dir) / s["folder"])
        bundle = LawExtractor(s["law_key"], raw).extract()
        graph.import_law(bundle)

    print("\nPHASE 3: IMPORTING AMENDMENTS")
    for law_id, amendments in ingest_amendments(root_dir, folder_to_law_key).items():
        graph.import_amendments(law_id, amendments)

    graph.print_statistics()
    return graph


def build_knowledge_graph(
    docs:           List[Document],
    neo4j_uri:      str,
    neo4j_user:     str,
    neo4j_password: str,
    drop_existing:  bool = False,
) -> LegalKnowledgeGraph:
    graph = LegalKnowledgeGraph(neo4j_uri, neo4j_user, neo4j_password)
    graph.setup_schema(drop_existing=drop_existing)
    graph.build_from_documents(docs)
    graph.print_statistics()
    return graph


# =============================================================================
# ENTRY POINT
# =============================================================================

def run_KG():
    try:
        from src.Config.config import NEO4J_PASSWORD, NEO4J_USERNAME, NEO4J_URI
    except Exception as e:
        print(e)
        return
    run_pipeline(DataPath, NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, drop_existing=True).close()