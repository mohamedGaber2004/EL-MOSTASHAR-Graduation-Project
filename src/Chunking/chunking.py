from __future__ import annotations

import fnmatch
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional , Tuple

from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

from src.Config.config import DataPath, na2d_data_path
from src.Utils.text_loader import MultiEncodingTextLoader
from src.Utils.regex_utils import (
    _to_western_digits,
    ORIGINAL_LAW_RE , 
    _DATE_RE , 
    _MONTH_MAP
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

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
    "mbade2_ultra_clean":        "مبادئ محكمة النقض",
    "1762919079675_ultra_clean": "مبادئ محكمة النقض - المجلد الثاني",
    "2019-2018_المستحدث":        "المستحدث من المبادئ 2018-2019",
}

ENCODINGS = ("utf-8", "utf-16", "cp1256", "windows-1256")

_CASE_NUM_RE = re.compile(r"(?:الطعن|طعن)\s+رق[مم]\s+([٠-٩0-9]+(?:\s*[,،]\s*[٠-٩0-9]+)*)\s+لسنة\s+([٠-٩0-9]+)", re.UNICODE)
_RULING_DATE_RE = re.compile(r"جلسة\s+(?:[٠-٩0-9]+\s*/\s*[٠-٩0-9]+\s*/\s*[٠-٩0-9]{2,4}|[٠-٩0-9]+\s+(?:يناير|فبراير|مارس|أبريل|مايو|يونيو|يوليو|أغسطس|سبتمبر|أكتوبر|نوفمبر|ديسمبر)\s+[٠-٩0-9]{4})", re.UNICODE)
_CHAMBER_RE = re.compile(r"(?:الدائرة|دائرة)\s+([^\n،,]+?)(?=\n|،|,|$)", re.UNICODE)

_article_splitter = CharacterTextSplitter(separator="\n\n", chunk_size=10_000, chunk_overlap=0)
_table_splitter   = CharacterTextSplitter(separator="\n\n\n\n", chunk_size=500_000, chunk_overlap=0)

# =============================================================================
# HELPERS
# =============================================================================

def _amendment_type(raw_text: str) -> str:
    sample = raw_text[:800]
    if any(w in sample for w in ["تضاف","يضاف","أضيف","إضافة","جديدة","جديد"]):
        return "addition"
    if any(w in sample for w in ["يلغى","ألغي","يحذف","حذف","إلغاء"]):
        return "deletion"
    return "modification"


def _amendment_date(raw_text: str, year: str) -> str:
    m = _DATE_RE.search(raw_text)
    if m:
        day   = _to_western_digits(m.group(1) or "01").zfill(2)
        month = _MONTH_MAP.get(m.group(2), "01")
        yr    = _to_western_digits(m.group(3))
        return f"{yr}-{month}-{day}"
    return f"{year}-01-01"

def _num_year_from_header(raw_text: str) -> Tuple[Optional[str], Optional[str]]:
    for m in ORIGINAL_LAW_RE.finditer(raw_text):
        try:
            a, b = int(_to_western_digits(m.group(1))), int(_to_western_digits(m.group(2)))
            year, num = (str(a), str(b)) if 1800 <= a <= 2100 else (str(b), str(a))
            if 1800 <= int(year) <= 2100 and 0 < int(num) < 10000:
                return num, year
        except Exception:
            continue
    return None, None

def _get_amendment_metadata(raw_text: str, filename: str) -> Optional[Dict[str, str]]:
    law_num, year = _num_year_from_header(raw_text)
    if not law_num or not year:
        return None
    return {
        "law_num":        law_num,
        "year":           year,
        "amendment_date": _amendment_date(raw_text, year),
        "amendment_type": _amendment_type(raw_text),
    }

def _read_file(fp: Path) -> Optional[str]:
    for enc in ENCODINGS:
        try:
            return fp.read_text(encoding=enc)
        except Exception:
            continue
    return fp.read_bytes().decode("utf-8", errors="replace")


def _book_title(stem: str) -> str:
    for key, title in BOOK_TITLE_MAP.items():
        if key in stem:
            return title
    return stem


def _ruling_metadata(text: str, folder: str, filename: str) -> Dict[str, Any]:
    header = text[:1000]
    meta   = {"doc_type": "ruling", "crime_category": folder, "source_file": filename}
    m = _CASE_NUM_RE.search(header)
    if m:
        meta["case_number"] = _to_western_digits(m.group(1).replace(" ", ""))
        meta["case_year"]   = _to_western_digits(m.group(2))
    m = _RULING_DATE_RE.search(header)
    if m:
        meta["ruling_date"] = m.group(0).replace("جلسة", "").strip()
    m = _CHAMBER_RE.search(header)
    if m:
        meta["chamber"] = m.group(1).strip()
    return meta


def _make_splitter(max_words: int, overlap: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size     = max_words,
        chunk_overlap  = overlap,
        length_function= lambda t: len(t.split()),
        separators     = ["\n\n", "\n", ".", " ", ""],
    )

# =============================================================================
# CORPUS CHUNKER
# =============================================================================

class CorpusChunker:
    """
    Unified chunker for both corpora.

        chunker   = CorpusChunker()
        law_docs  = chunker.get_chunks()
        na2d_docs = chunker.get_na2d_chunks()  # {"rulings": [...], "principles": [...]}
    """

    def __init__(
        self,
        laws_dir:       str | Path = DataPath,
        na2d_dir:       str | Path = na2d_data_path,
        max_words_book: int        = 400,
        overlap_book:   int        = 50,
        max_words_rule: int        = 300,
        overlap_rule:   int        = 40,
    ) -> None:
        self.laws_dir         = Path(laws_dir)
        self.na2d_dir         = Path(na2d_dir)
        self._book_splitter   = _make_splitter(max_words_book, overlap_book)
        self._ruling_splitter = _make_splitter(max_words_rule, overlap_rule)

    def get_chunks(self) -> List[Document]:
        """One Document per article or table block for all law files."""
        all_docs: List[Document] = []

        for folder in sorted(p for p in self.laws_dir.iterdir() if p.is_dir()):
            if folder.name not in FOLDER_TO_LAW_KEY:
                continue
            law_key   = FOLDER_TO_LAW_KEY[folder.name]
            law_title = LAW_KEY_TO_TITLE.get(law_key, law_key)

            for fp in sorted(f for f in folder.glob("*.txt") if not fnmatch.fnmatch(f.name.lower(), "new*")):
                is_table   = "tables" in fp.name.lower()
                splitter   = _table_splitter if is_table else _article_splitter
                chunk_type = "table" if is_table else "article"
                raw = MultiEncodingTextLoader(fp).load()
                if not raw:
                    continue
                docs = splitter.create_documents(
                    texts     = [raw[0].page_content],
                    metadatas = [{"law_id": law_key, "law_title": law_title, "chunk_type": chunk_type}],
                )
                all_docs.extend(docs)
                logger.info(f"  {fp.name} [{chunk_type}] → {len(docs)} chunks")

            for af in sorted(folder.glob("new*.txt")):
                raw = MultiEncodingTextLoader(af).load()
                if not raw:
                    continue
                meta = _get_amendment_metadata(raw[0].page_content, af.name)
                if not meta:
                    logger.warning(f"  [SKIP] {af.name} — no amendment metadata")
                    continue
                docs = _article_splitter.create_documents(
                    texts     = [raw[0].page_content],
                    metadatas = [{"law_id": law_key, "law_title": law_title,
                                  "chunk_type":           "amended_article",
                                  "amendment_law_number": meta["law_num"],
                                  "amendment_date":       meta["amendment_date"],
                                  "amendment_type":       meta["amendment_type"]}],
                )
                all_docs.extend(docs)
                logger.info(f"  {af.name} [amendment] → {len(docs)} chunks")

        logger.info(f"✓ laws total: {len(all_docs)} chunks")
        return all_docs

    def get_na2d_chunks(self) -> Dict[str, List[Document]]:
        """Rulings and principles chunks from the na2d corpus."""
        ruling_docs:    List[Document] = []
        principle_docs: List[Document] = []

        for item in sorted(self.na2d_dir.iterdir()):
            if item.is_file() and item.suffix.lower() == ".txt":
                text = _read_file(item)
                if not text:
                    continue
                chunks = self._book_splitter.create_documents(
                    texts     = [text.strip()],
                    metadatas = [{"doc_type": "principle", "book_title": _book_title(item.stem), "source_file": item.name}],
                )
                principle_docs.extend(chunks)
                logger.info(f"  [BOOK] {item.name} → {len(chunks)} chunks")

            elif item.is_dir():
                folder_count = 0
                for fp in sorted(item.glob("*.txt")):
                    text = _read_file(fp)
                    if not text:
                        continue
                    chunks = self._ruling_splitter.create_documents(
                        texts     = [text.strip()],
                        metadatas = [_ruling_metadata(text, item.name, fp.name)],
                    )
                    ruling_docs.extend(chunks)
                    folder_count += len(chunks)
                logger.info(f"  [RULINGS] {item.name} → {folder_count} chunks")

        logger.info(f"✓ na2d: rulings={len(ruling_docs)} | principles={len(principle_docs)}")
        return {"rulings": ruling_docs, "principles": principle_docs}


# =============================================================================
# ENTRY POINTS
# =============================================================================

def get_chunks() -> List[Document]:
    return CorpusChunker().get_chunks()

def get_na2d_chunks() -> Dict[str, List[Document]]:
    return CorpusChunker().get_na2d_chunks()