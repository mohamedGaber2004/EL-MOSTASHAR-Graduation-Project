# src/utils/article_number_normalizer.py
from __future__ import annotations

import re
from typing import Iterable

_ARABIC_DIGIT_TRANSLATION = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

_LETTER_EQUIV = {
    "ا": "أ",
    "أ": "أ",
    "إ": "أ",
    "آ": "أ",
    "ى": "ي",
    "ي": "ي",
}


def _clean_spaces(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s*/\s*", "/", text)
    text = re.sub(r"\s*\(\s*", " (", text)
    text = re.sub(r"\s*\)\s*", ")", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_article_number(value: str | None) -> str | None:
    if value is None:
        return None

    s = str(value).strip()
    if not s:
        return None

    s = s.translate(_ARABIC_DIGIT_TRANSLATION)
    s = s.replace("مكرراً", "مكرر")
    s = s.replace("مكررا", "مكرر")
    s = s.replace("مكررة", "مكرر")
    s = s.replace("ـ", "")

    s = _clean_spaces(s)

    # Normalize bracketed Arabic letter forms like: 78(أ) -> 78 (أ)
    s = re.sub(r"^(\d+)\(([^)]+)\)$", r"\1 (\2)", s)
    s = re.sub(r"^(\d+)\s*([أإآا])$", lambda m: f"{m.group(1)} ({_LETTER_EQUIV.get(m.group(2), m.group(2))})", s)

    # Normalize common "مكرر" forms
    s = re.sub(r"^(\d+)\s*مكرر\s*\(([^)]+)\)$", r"\1 مكرر (\2)", s)
    s = re.sub(r"^(\d+)\s*مكرر\s*([أإآا])$", lambda m: f"{m.group(1)} مكرر ({_LETTER_EQUIV.get(m.group(2), m.group(2))})", s)

    # Normalize "ف(1)" spacing
    s = re.sub(r"^(\d+)\s*ف\s*\(\s*(\d+)\s*\)$", r"\1 ف(\2)", s)

    return s


def generate_article_number_variants(value: str | None) -> list[str]:
    base = normalize_article_number(value)
    if not base:
        return []

    variants: list[str] = []
    seen: set[str] = set()

    def add(x: str | None):
        if not x:
            return
        x = normalize_article_number(x) or str(x).strip()
        if x and x not in seen:
            seen.add(x)
            variants.append(x)

    add(base)

    # Keep slash forms as-is and with optional spaces
    add(base.replace(" / ", "/"))
    add(base.replace("/", " / "))

    # 78 (أ) -> 78أ / 78/أ / 78 (أ)
    m = re.match(r"^(\d+)\s*\(([^)]+)\)$", base)
    if m:
        n, letter = m.group(1), m.group(2).strip()
        add(f"{n}{letter}")
        add(f"{n}/{letter}")
        add(f"{n} {letter}")

    # 78أ -> 78 (أ) / 78/أ / 78أ
    m = re.match(r"^(\d+)([أإآا])$", base)
    if m:
        n, letter = m.group(1), _LETTER_EQUIV.get(m.group(2), m.group(2))
        add(f"{n} ({letter})")
        add(f"{n}/{letter}")
        add(f"{n}{letter}")

    # 86 مكرر (أ) -> variants without/with spaces and slash
    m = re.match(r"^(\d+)\s*مكرر\s*\(([^)]+)\)$", base)
    if m:
        n, letter = m.group(1), m.group(2).strip()
        add(f"{n} مكرر({letter})")
        add(f"{n} مكرر {letter}")
        add(f"{n} مكرر/{letter}")

    # 86 مكررأ -> normalize into bracketed form too
    m = re.match(r"^(\d+)\s*مكرر\s*([أإآا])$", base)
    if m:
        n, letter = m.group(1), _LETTER_EQUIV.get(m.group(2), m.group(2))
        add(f"{n} مكرر ({letter})")
        add(f"{n} مكرر({letter})")

    # 1 ف(1) variants
    m = re.match(r"^(\d+)\s*ف\((\d+)\)$", base)
    if m:
        n, idx = m.group(1), m.group(2)
        add(f"{n} ف ({idx})")
        add(f"{n} ف {idx}")
        add(f"{n}/ف({idx})")

    return variants