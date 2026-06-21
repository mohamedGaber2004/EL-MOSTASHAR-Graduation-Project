from __future__ import annotations

from typing import List, Optional, Any, Literal
from pydantic import BaseModel, Field, field_validator, ConfigDict
import json
import re
import logging


# =========================================================================
# Strict schema literals
# =========================================================================

LawCodeLiteral = Literal[
    "anti_drugs",
    "anti_terror",
    "criminal_constitution",
    "criminal_procedure",
    "cybercrime",
    "emergency_law",
    "money_laundering",
    "penal_code",
    "weapons_ammunition",
]

LegalBasisType = Literal[
    "explicit_warrant",
    "flagrancy",
    "incidental_to_lawful_arrest",
    "statutory_authority",
    "express_no_authority",
    "unknown",
]

VALID_LAW_CODES = set(LawCodeLiteral.__args__)

logger = logging.getLogger(__name__)


# =========================================================================
# Normalisation helpers
# =========================================================================

_ARABIC_INDIC_TRANS = str.maketrans(
    "\u0660\u0661\u0662\u0663\u0664\u0665\u0666\u0667\u0668\u0669"  # ٠١٢٣٤٥٦٧٨٩
    "\u06F0\u06F1\u06F2\u06F3\u06F4\u06F5\u06F6\u06F7\u06F8\u06F9",  # ۰۱۲۳۴۵۶۷۸۹
    "01234567890123456789",
)

_LETTER_EQUIV = {
    "ا": "أ",
    "أ": "أ",
    "إ": "أ",
    "آ": "أ",
    "ى": "ي",
    "ي": "ي",
}

_VALID_ARTICLE_RE = re.compile(
    r"^\d+"
    r"(?:\s+(?:مكرر(?:\s*\([^)]+\))?|\([^)]+\)|[أبجدھهوزحطيكلمنسعفصقرشتثخذضظغ]))?"
    r"(?:\s+\d+)?$"
)


def _clean_spaces(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s*/\s*", "/", text)
    text = re.sub(r"\s*\(\s*", " (", text)
    text = re.sub(r"\s*\)\s*", ")", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def _normalize_article_number(v: Any) -> Optional[str]:
    """
    Robustly coerce an article_number value to a clean string.

    Accepts common legal forms such as:
      - 3 مكرر
      - 309/ب
      - 77 (أ)
      - 78(أ)
      - 86 مكرر (أ)
      - 1/7
      - 1/34
      - 1 ف(1)
      - 1/1/30
      - 25 مكرراً

    The goal is to preserve the legal form while removing noise and
    standardizing spacing/punctuation enough for exact matching and
    downstream comparison.
    """
    if v is None:
        return None

    if isinstance(v, (int, float)):
        if isinstance(v, float) and not v.is_integer():
            return str(v)
        return str(int(v))

    if not isinstance(v, str):
        v = str(v)

    s = v.strip()
    if not s:
        return None

    # Translate Arabic-Indic and Eastern Arabic numerals to Western numerals
    s = s.translate(_ARABIC_INDIC_TRANS)

    # Normalize common spelling variants of "مكرر"
    s = s.replace("مكرراً", "مكرر")
    s = s.replace("مكررا", "مكرر")
    s = s.replace("مكررة", "مكرر")
    s = s.replace("مكـرر", "مكرر")

    # Remove zero-width and decorative artifacts
    s = s.replace("\u200f", "").replace("\u200e", "").replace("ـ", "")

    s = _clean_spaces(s)
    s = s.strip("\"'،,.;: ")

    # Normalize forms like 78(أ) -> 78 (أ)
    s = re.sub(r"^(\d+)\(([^)]+)\)$", r"\1 (\2)", s)

    # Normalize forms like 78أ -> 78 (أ)
    s = re.sub(
        r"^(\d+)\s*([أإآا])$",
        lambda m: f"{m.group(1)} ({_LETTER_EQUIV.get(m.group(2), m.group(2))})",
        s,
    )

    # Normalize forms like 86 مكررأ -> 86 مكرر (أ)
    s = re.sub(
        r"^(\d+)\s*مكرر\s*([أإآا])$",
        lambda m: f"{m.group(1)} مكرر ({_LETTER_EQUIV.get(m.group(2), m.group(2))})",
        s,
    )

    # Normalize forms like 86 مكرر(أ) -> 86 مكرر (أ)
    s = re.sub(r"^(\d+)\s*مكرر\s*\(([^)]+)\)$", r"\1 مكرر (\2)", s)

    # Normalize forms like 1 ف (1) or 1 ف(1) -> 1 ف(1)
    s = re.sub(r"^(\d+)\s*ف\s*\(\s*(\d+)\s*\)$", r"\1 ف(\2)", s)
    s = re.sub(r"^(\d+)\s*ف\s+(\d+)$", r"\1 ف(\2)", s)

    # Normalize some weird slash spacing
    s = re.sub(r"\s*/\s*", "/", s)

    # Final cleanup
    s = _clean_spaces(s)

    return s or None


# =========================================================================
# Shared coercion helpers
# =========================================================================
def _coerce_str_list(v: Any) -> List[str]:
    """
    Normalise an LLM list-of-strings field.

    Handles:
      - None / missing           → []
      - bare string              → [string]
      - list of strings          → as-is
      - list of dicts            → extract first non-null string value per dict
      - list of mixed types      → best-effort extraction
    """
    if v is None:
        return []
    if isinstance(v, str):
        return [v] if v.strip() else []
    if isinstance(v, list):
        result: list[str] = []
        for item in v:
            if isinstance(item, str):
                if item.strip():
                    result.append(item.strip())
            elif isinstance(item, dict):
                for val in item.values():
                    if isinstance(val, str) and val.strip():
                        result.append(val.strip())
                        break
        return result
    return []

def _coerce_bool_required(v: Any) -> bool:
    """
    Coerce an LLM boolean field where the default is False.
    Handles Arabic affirmatives, English strings, ints.
    """
    if isinstance(v, bool):
        return v
    if isinstance(v, int):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in {"true", "yes", "1", "نعم", "صحيح", "موجود"}
    return False

def _coerce_bool_optional(v: Any) -> Optional[bool]:
    """
    Coerce an LLM boolean field where None is semantically meaningful
    (i.e. 'we don't know').
    """
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, int):
        return bool(v)
    if isinstance(v, str):
        stripped = v.strip().lower()
        if stripped in {"true", "yes", "1", "نعم", "صحيح", "موجود"}:
            return True
        if stripped in {"false", "no", "0", "لا", "غير موجود", "لم"}:
            return False
        return None
    return None

def _coerce_name_with_title(v: Any) -> Optional[str]:
    """
    Coerce a field meant to contain a person's name plus optional title/rank.
    """
    if v is None:
        return None
    if isinstance(v, str):
        return v.strip() or None
    if isinstance(v, dict):
        name = v.get("name") or v.get("الاسم") or ""
        title = v.get("title") or v.get("الصفة") or v.get("role") or v.get("صفة") or ""
        parts = [str(p).strip() for p in (name, title) if p and str(p).strip()]
        return " - ".join(parts) or None
    if isinstance(v, list):
        flat = _coerce_str_list(v)
        return "، ".join(flat) or None
    return str(v).strip() or None

def _coerce_singular_name(v: Any) -> Optional[str]:
    """
    Coerce a field meant to hold one person's name, but which LLMs may
    return as list/dict/string.
    """
    if v is None:
        return None
    if isinstance(v, str):
        return v.strip() or None
    if isinstance(v, list):
        flat = _coerce_str_list(v)
        return "، ".join(flat) or None
    if isinstance(v, dict):
        for val in v.values():
            if isinstance(val, str) and val.strip():
                return val.strip()
        return None
    return str(v).strip() or None

# =========================================================================
# Core Entities
# =========================================================================
class Defendant(BaseModel):
    """متهم في القضية"""

    name:               Optional[str] = None
    alias:              Optional[str] = None
    national_id:        Optional[str] = None
    gender:             Optional[str] = None
    date_of_birth:      Optional[str] = None
    age:                Optional[int] = None
    occupation:         Optional[str] = None
    nationality:        Optional[str] = "مصري"
    address:            Optional[str] = None
    complicity_role:    Optional[str] = None
    source_document_id: Optional[str] = None

    @field_validator("age", mode="before")
    @classmethod
    def coerce_age(cls, v: Any) -> Optional[int]:
        """
        LLMs often return age as a string ('32', 'نحو 30', '~30').
        Extract the first integer found; return None if not parsable.
        """
        if v is None:
            return None
        if isinstance(v, int):
            return v
        if isinstance(v, float):
            return int(v)
        if isinstance(v, str):
            m = re.search(r"\d+", v)
            return int(m.group()) if m else None
        return None

    @field_validator("nationality", mode="before")
    @classmethod
    def coerce_nationality(cls, v: Any) -> str:
        """
        Preserve default 'مصري' when the model returns null.
        """
        if v is None:
            return "مصري"
        if isinstance(v, str) and v.strip():
            return v.strip()
        return "مصري"

class Charge(BaseModel):
    """تهمة موجهة"""

    law_code:              Optional[LawCodeLiteral] = None
    article_number:        Optional[str] = None
    description:           Optional[str] = None
    incident_type:         Optional[str] = None
    charge_classification:  Optional[str] = None
    attempt_flag:          bool = False
    charge_date:           Optional[str] = None
    charge_location:       Optional[str] = None
    linked_defendant_names: List[str] = Field(default_factory=list)
    source_document_id:    Optional[str] = None

    @field_validator("law_code", mode="before")
    @classmethod
    def coerce_law_code(cls, v: Any) -> Optional[str]:
        if not v:
            return None

        val_str = str(v).strip().lower()

        if val_str in VALID_LAW_CODES:
            return val_str

        arabic_fallback_map = {
            "قانون العقوبات": "penal_code",
            "عقوبات": "penal_code",
            "قانون الإجراءات الجنائية": "criminal_procedure",
            "إجراءات جنائية": "criminal_procedure",
            "قانون مكافحة المخدرات": "anti_drugs",
            "مخدرات": "anti_drugs",
            "قانون مكافحة الإرهاب": "anti_terror",
            "إرهاب": "anti_terror",
            "قانون الأسلحة والذخائر": "weapons_ammunition",
            "أسلحة وذخائر": "weapons_ammunition",
            "قانون مكافحة غسل الأموال": "money_laundering",
            "غسل أموال": "money_laundering",
            "قانون مكافحة جرائم تقنية المعلومات": "cybercrime",
            "جرائم إلكترونية": "cybercrime",
            "جرائم تقنية": "cybercrime",
            "قانون الطوارئ": "emergency_law",
            "طوارئ": "emergency_law",
            "الدستور": "criminal_constitution",
        }

        for arabic_key, english_val in arabic_fallback_map.items():
            if arabic_key in val_str:
                return english_val

        logger.warning("law_code has unexpected format or value: %s", val_str)
        return None

    @field_validator("article_number", mode="before")
    @classmethod
    def coerce_article_number(cls, v: Any) -> Optional[str]:
        result = _normalize_article_number(v)
        if result and not _VALID_ARTICLE_RE.match(result):
            logger.warning(
                "article_number has unexpected format after normalisation: %r",
                result,
            )
        return result

    @field_validator("attempt_flag", mode="before")
    @classmethod
    def coerce_attempt_flag(cls, v: Any) -> bool:
        return _coerce_bool_required(v)

    @field_validator("linked_defendant_names", mode="before")
    @classmethod
    def coerce_linked_defendant_names(cls, v: Any) -> List[str]:
        return _coerce_str_list(v)

class CaseIncident(BaseModel):
    """واقعة / حادثة"""

    incident_type:        Optional[str] = None
    incident_date:        Optional[str] = None
    incident_location:    Optional[str] = None
    incident_description: Optional[str] = None
    perpetrator_names:    List[str] = Field(default_factory=list)
    victim_names:         List[str] = Field(default_factory=list)
    source_document_id:   Optional[str] = None

    @field_validator("incident_date", mode="before")
    @classmethod
    def coerce_date(cls, v: Any) -> Optional[str]:
        """
        Normalise date fields — LLMs sometimes return a dict with
        'start' / 'end' keys for date ranges.
        """
        if isinstance(v, dict):
            start = v.get("start", "")
            end = v.get("end", "")
            if start and end:
                return f"{start} إلى {end}"
            return start or end or None
        if isinstance(v, str):
            s = v.strip()
            return s or None
        return v

    @field_validator("perpetrator_names", "victim_names", mode="before")
    @classmethod
    def coerce_name_lists(cls, v: Any) -> List[str]:
        return _coerce_str_list(v)

class Evidence(BaseModel):
    """دليل مادي / إجرائي (حرز)"""

    evidence_type:         Optional[str] = None
    description:           Optional[str] = None
    detailed_text:         Optional[str] = None
    seizure_date:          Optional[str] = None
    seizure_location:      Optional[str] = None
    seized_by:             Optional[str] = None
    seizure_warrant_present: bool = False
    linked_defendant_name:  Optional[str] = None
    source_document_id:     Optional[str] = None

    @field_validator("seizure_warrant_present", mode="before")
    @classmethod
    def coerce_seizure_warrant_present(cls, v: Any) -> bool:
        return _coerce_bool_required(v)

    @field_validator("seized_by", mode="before")
    @classmethod
    def coerce_seized_by(cls, v: Any) -> Optional[str]:
        return _coerce_name_with_title(v)

    @field_validator("linked_defendant_name", mode="before")
    @classmethod
    def coerce_linked_defendant_name(cls, v: Any) -> Optional[str]:
        return _coerce_singular_name(v)

class EvidenceItem(BaseModel):
    description: str
    evidence_number: Optional[str] = None

class LabReport(BaseModel):
    report_type:          Optional[str] = None
    report_number:        Optional[str] = None
    examination_date:     Optional[str] = None
    examiner_name:        Optional[str] = None
    prosecutor_name:      Optional[str] = None
    items_sent_for_analysis: list[EvidenceItem] = Field(default_factory=list)
    result:               Optional[str] = None
    linked_defendant_name: Optional[str] = None
    source_document_id:   Optional[str] = None

    @field_validator("examiner_name", mode="before")
    @classmethod
    def coerce_examiner_name(cls, v: Any) -> Optional[str]:
        return _coerce_name_with_title(v)

    @field_validator("linked_defendant_name", mode="before")
    @classmethod
    def coerce_linked_defendant_name(cls, v: Any) -> Optional[str]:
        return _coerce_singular_name(v)

    @field_validator("items_sent_for_analysis", mode="before")
    @classmethod
    def coerce_items(cls, v: Any) -> list:
        if not isinstance(v, list):
            if isinstance(v, str) and v.strip():
                return [{"description": v}]
            return []

        coerced = []
        for item in v:
            if isinstance(item, str):
                if item.strip():
                    coerced.append({"description": item})
            elif isinstance(item, dict):
                if not item.get("description"):
                    item["description"] = (
                        item.get("name")
                        or item.get("item")
                        or item.get("content")
                        or item.get("text")
                        or str(item)
                    )
                coerced.append(item)
        return coerced

    @field_validator("result", mode="before")
    @classmethod
    def coerce_result(cls, v: Any) -> Optional[str]:
        if isinstance(v, dict):
            return json.dumps(v, ensure_ascii=False)
        if isinstance(v, list):
            return json.dumps(v, ensure_ascii=False)
        return v

class WitnessStatement(BaseModel):
    witness_name:          Optional[str] = None
    witness_type:          Optional[str] = None
    occupation:            Optional[str] = None
    id_number:             Optional[str] = None
    relation_to_defendant: Optional[str] = None
    statement_date:        Optional[str] = None
    statement_summary:     Optional[str] = None
    was_sworn_in:          Optional[bool] = None
    presence_at_scene:     Optional[bool] = None
    source_document_id:    Optional[str] = None

    @field_validator("statement_summary", mode="before")
    @classmethod
    def coerce_statement_summary(cls, v: Any) -> Optional[str]:
        """
        Normalise statement_summary:
          - list of Q/A dicts → formatted string
          - list of strings   → joined string
          - plain string      → as-is
        """
        if isinstance(v, list):
            parts: list[str] = []
            for item in v:
                if isinstance(item, dict):
                    q = item.get("question") or ""
                    a = item.get("answer") or ""
                    parts.append(f"س: {q}\nج: {a}".strip() if q else a)
                elif isinstance(item, str):
                    parts.append(item)
            return "\n\n".join(filter(None, parts)) or None
        return v

    @field_validator("was_sworn_in", "presence_at_scene", mode="before")
    @classmethod
    def coerce_optional_bools(cls, v: Any) -> Optional[bool]:
        return _coerce_bool_optional(v)

class Confession(BaseModel):
    """اعتراف / إنكار في محضر الاستجواب"""

    defendant_name:       Optional[str] = None
    text:                 Optional[str] = None
    confession_date:      Optional[str] = None
    confession_stage:     Optional[str] = None
    legal_counsel_present: bool = False
    coercion_claimed:     bool = False
    key_admissions:       List[str] = Field(default_factory=list)
    source_document_id:   Optional[str] = None

    @field_validator("legal_counsel_present", "coercion_claimed", mode="before")
    @classmethod
    def coerce_bools(cls, v: Any) -> bool:
        return _coerce_bool_required(v)

    @field_validator("key_admissions", mode="before")
    @classmethod
    def coerce_key_admissions(cls, v: Any) -> List[str]:
        return _coerce_str_list(v)

class CriminalRecord(BaseModel):
    """صحيفة السوابق الجنائية لمتهم"""

    defendant_name:     Optional[str] = None
    record_summary:     Optional[str] = None
    prior_cases:        List[str] = Field(default_factory=list)
    source_document_id: Optional[str] = None

    @field_validator("prior_cases", mode="before")
    @classmethod
    def coerce_prior_cases(cls, v: Any) -> List[str]:
        return _coerce_str_list(v)

class CriminalProceedings(BaseModel):
    model_config = ConfigDict(extra="ignore")

    procedure_type:        Optional[str] = None  # ضبط / قبض / استجواب / تفتيش / معاينة / انتقال / إخطار / غيره
    description:           Optional[str] = None  # الوصف كما ورد
    procedure_datetime:    Optional[str] = None  # null إن لم يُذكر
    legal_basis_present:   Optional[bool] = None  # true/false/null
    legal_basis_type:      Optional[LegalBasisType] = None
    legal_basis_note:      Optional[str] = None
    conducting_officer:    Optional[str] = None
    source_document_id:    Optional[str] = None

    @field_validator("legal_basis_present", mode="before")
    @classmethod
    def coerce_legal_basis_present(cls, v: Any) -> Optional[bool]:
        return _coerce_bool_optional(v)

    @field_validator("legal_basis_type", mode="before")
    @classmethod
    def coerce_legal_basis_type(cls, v: Any) -> Optional[str]:
        if v is None:
            return None

        raw = str(v).strip().lower()

        mapping = {
            "إذن نيابة": "explicit_warrant",
            "إذن قضائي": "explicit_warrant",
            "warrant": "explicit_warrant",
            "explicit_warrant": "explicit_warrant",
            "تلبس": "flagrancy",
            "flagrancy": "flagrancy",
            "في حالة تلبس": "flagrancy",
            "incidental_to_lawful_arrest": "incidental_to_lawful_arrest",
            "تبعي للقبض الصحيح": "incidental_to_lawful_arrest",
            "statutory_authority": "statutory_authority",
            "سند قانوني آخر": "statutory_authority",
            "express_no_authority": "express_no_authority",
            "no_authority": "express_no_authority",
            "unknown": "unknown",
        }

        return mapping.get(raw, "unknown")

    @field_validator("conducting_officer", mode="before")
    @classmethod
    def coerce_conducting_officer(cls, v: Any) -> Optional[str]:
        return _coerce_name_with_title(v)

    @field_validator("procedure_datetime", mode="before")
    @classmethod
    def coerce_procedure_datetime(cls, v: Any) -> Optional[str]:
        """
        Same dict-shape drift as CaseIncident.incident_date. Also normalises
        an unfilled placeholder copied verbatim from a محضر (e.g. '--------')
        down to a clean None.
        """
        if isinstance(v, dict):
            date = v.get("date", "") or v.get("التاريخ", "")
            time = v.get("time", "") or v.get("الوقت", "")
            combined = " ".join(p for p in (date, time) if p).strip()
            return combined or None

        if isinstance(v, str):
            stripped = v.strip()
            if not stripped or set(stripped) <= {"-", "_", "."}:
                return None
            return stripped

        return v

class DefenseDocument(BaseModel):
    submitted_by:          Optional[str] = None
    defendant_name:        Optional[str] = None
    formal_defenses:       List[str] = Field(default_factory=list)
    substantive_defenses:  List[str] = Field(default_factory=list)
    supporting_principles: List[str] = Field(default_factory=list)
    alibi_claimed:         bool = False
    alibi_description:     Optional[str] = None
    source_document_id:    Optional[str] = None

    @field_validator("alibi_claimed", mode="before")
    @classmethod
    def coerce_alibi_claimed(cls, v: Any) -> bool:
        return _coerce_bool_required(v)

    @field_validator(
        "formal_defenses",
        "substantive_defenses",
        "supporting_principles",
        mode="before",
    )
    @classmethod
    def coerce_defense_lists(cls, v: Any) -> List[str]:
        return _coerce_str_list(v)

    @field_validator("alibi_description", mode="before")
    @classmethod
    def coerce_alibi_description(cls, v: Any) -> Optional[str]:
        """
        LLMs sometimes return alibi as a structured dict with
        time / location / supporting_witnesses keys.
        """
        if isinstance(v, dict):
            time = v.get("time", "")
            location = v.get("location", "")
            witnesses = "، ".join(v.get("supporting_witnesses", []))
            parts = filter(None, [time, location, f"شهود: {witnesses}" if witnesses else ""])
            return " — ".join(parts) or None
        if isinstance(v, str):
            s = v.strip()
            return s or None
        return v