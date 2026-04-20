from __future__ import annotations

import logging , re 
from dataclasses import dataclass, field
from typing import (
    Any , 
    Dict , 
    List, 
    Optional , 
    Tuple
)
from Utils import (
    _to_western_digits,
    _normalize_article_no, 
    _stable_id,
    reg,
    norm_regu
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================



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
    amendments:  List[Amendment] = field(default_factory=list)


# =============================================================================
# LAW EXTRACTOR
# =============================================================================


class LawExtractor: 
    """Extract all structured data from a single law text."""

    def __init__(self, law_key: str, raw_text: str):
        if law_key not in norm_regu._LAW_REGISTRY.value:
            raise KeyError(f"Unknown law_key: '{law_key}'")
        title, date, schedule_type = norm_regu._LAW_REGISTRY.value[law_key]
        self.law_key       = law_key
        self.raw_text      = raw_text or ""
        self.title         = title
        self.date          = date
        self._schedule_type = schedule_type

    # ── private helpers ───────────────────────────────────────────────────

    def _dedup(self, items: List[Dict], keys: List[str]) -> List[Dict]:
        seen, out = set(), []
        for item in items:
            k = tuple(item.get(f) for f in keys)
            if k not in seen:
                seen.add(k)
                out.append(item)
        return out

    def _normalize_arabic(self, text: str) -> str:
        if not text:
            return ""
        text = re.sub(reg.PDF_UNICODES_REG.value, "", text).replace("\u0640", "")
        for old, new in norm_regu.NORMALIZING_ARABIC_LITTERS.value:
            text = text.replace(old, new)
        return re.sub(r"\s+", " ", text).strip()


    def _articles(self) -> List[Dict]:
        text    = self.raw_text
        matches = list(reg._ARTICLE_MARK_RE.value.finditer(text))
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
            for m in reg.PENALTY_PATTERNS.value['سجن'].finditer(text):
                out.append({"article_number": no, "penalty_type": "سجن",
                            "min_value": int(_to_western_digits(m.group(1))),
                            "max_value": int(_to_western_digits(m.group(2))), "unit": "سنة"})
            for m in reg.PENALTY_PATTERNS.value['غرامة'].finditer(text):
                out.append({"article_number": no, "penalty_type": "غرامة",
                            "min_value": int(_to_western_digits(m.group(1).replace(',', ''))) if m.group(1) else None,
                            "max_value": int(_to_western_digits(m.group(2).replace(',', ''))) if m.group(2) else None,
                            "unit": "جنيه"})
            if reg.PENALTY_PATTERNS.value['إعدام'].search(text):
                out.append({"article_number": no, "penalty_type": "إعدام", "min_value": None, "max_value": None, "unit": None})
            if reg.PENALTY_PATTERNS.value['أشغال_شاقة'].search(text):
                out.append({"article_number": no, "penalty_type": "أشغال_شاقة", "min_value": None, "max_value": None, "unit": None})
        return self._dedup(out, ["article_number", "penalty_type", "min_value", "max_value"])

    def _definitions(self, articles: List[Dict]) -> List[Dict]:
        out = []
        for a in articles:
            no = a.get("article_number")
            if not no:
                continue
            for m in reg.DEF_RE.value.finditer(a.get("text", "")):
                out.append({"term": m.group(1).strip(), "definition_text": m.group(2).strip(), "defined_in_article": no})
        return self._dedup(out, ["term", "defined_in_article"])

    def _references(self, articles: List[Dict]) -> List[Dict]:
        refs = set()
        for a in articles:
            from_no = str(a.get("article_number", "")).strip()
            if not from_no or from_no == "None":
                continue
            for m in reg.REF_RE.value.finditer(a.get("text", "")):
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
            text = self._normalize_arabic(a.get("text", ""))
            for topic, keywords in norm_regu.TOPICS_MAP.value.items():
                if any(self._normalize_arabic(kw) in text for kw in keywords):
                    out.append({"article_number": no, "topic_name": topic, "confidence": 0.8})
                    break
        return self._dedup(out, ["article_number", "topic_name"])

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
        )


# =============================================================================
# AMENDMENT EXTRACTOR
# =============================================================================

class AmendmentExtractor:

    def __init__(self, raw_text: str, filename: Optional[str] = None):
        self.raw_text = raw_text
        self.filename = filename

    def _law_num_year(self) -> Tuple[Optional[str], Optional[str]]:
        for m in reg.ORIGINAL_LAW_RE.value.finditer(self.raw_text):
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
        for m in reg._ANY_ARTICLE_RE.value.finditer(self.raw_text):
            norm = _normalize_article_no(m.group(1))
            if norm and norm not in seen:
                seen.add(norm)
                ordered.append(norm)
        return ordered

    def _amendment_type(self) -> str:
        sample = self.raw_text[:800]
        if any(w in sample for w in norm_regu.ADDITION_KEYWORDS.value):
            return 'addition'
        if any(w in sample for w in norm_regu.DELETION_KEYWORDS.value):
            return 'deletion'
        return 'modification'

    def _amendment_date(self, year: str) -> str:
        m = reg._DATE_RE.value.search(self.raw_text)
        if m:
            day   = _to_western_digits(m.group(1) or '01').zfill(2)
            month = reg._MONTH_MAP.value.get(m.group(2), '01')
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
        desc_m = re.search(reg.ARTICLE_1_2_ONLY_REG.value, self.raw_text, re.DOTALL)
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
