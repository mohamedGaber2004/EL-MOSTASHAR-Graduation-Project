from __future__ import annotations

import fnmatch
from src.Config.log_config import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter,CharacterTextSplitter

from src.Config import get_settings
from src.Utils import (
    MultiEncodingTextLoader,
    _read_file,
    _to_western_digits,
    norm_regu,
    reg,
    _normalize_article_no
)

 # Logging is configured globally in src/log_config.py
logger = logging.getLogger(__name__)

# =============================================================================
# CORPUS CHUNKER
# =============================================================================

def _split_into_tables(
    full_text:  str,
    law_id:     str,
    law_title:  str,
    source_file: str,
) -> List[Document]:
    """
    Split a tables file into one Document per logical table.
    Each Document gets the raw text of that table only.
    """
    matches = list(reg._TABLE_HEADER_RE.value.finditer(full_text))

    if not matches:
        # No headers found — return the whole file as one chunk
        return [Document(
            page_content = full_text.strip(),
            metadata     = {
                "law_id":        law_id,
                "law_title":     law_title,
                "chunk_type":    "table",
                "table_number":  "0",
                "source_file":   source_file,
            },
        )]

    docs = []
    for i, m in enumerate(matches):
        table_num = _to_western_digits(m.group(1))
        start     = m.start()                                      # include the header line
        end       = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        text      = full_text[start:end].strip()
        if text:
            docs.append(Document(
                page_content = text,
                metadata     = {
                    "law_id":       law_id,
                    "law_title":    law_title,
                    "chunk_type":   "table",
                    "table_number": table_num,
                    "source_file":  source_file,
                },
            ))
    return docs

def _split_into_articles(full_text: str) -> List[Dict[str, Optional[str]]]:
    """
    Split raw law text into per-article dicts::

        {"article_number": "5", "text": "..."}

    Text before the first article header gets ``article_number=None``
    and is tagged as a preamble by the caller.
    """
    matches  = list(reg._ARTICLE_BOUNDARY_RE.value.finditer(full_text))
    articles: List[Dict[str, Optional[str]]] = []

    if not matches:
        # No article markers found → treat entire text as one chunk
        return [{"article_number": None, "text": full_text.strip()}]

    # Preamble before first article
    if matches[0].start() > 0:
        preamble = full_text[: matches[0].start()].strip()
        if preamble:
            articles.append({"article_number": None, "text": preamble})

    for i, m in enumerate(matches):
        start      = m.end()
        end        = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        body       = full_text[start:end].strip()
        article_no = _normalize_article_no(m.group("num") or "")
        if body:
            articles.append({"article_number": article_no, "text": body})

    return articles

_table_splitter = CharacterTextSplitter(
    separator="\n\n", chunk_size=500_000, chunk_overlap=0
)

class CorpusChunker:
    def __init__(
        self,
        laws_dir:       str | Path = get_settings().DataPath,
        na2d_dir:       str | Path = get_settings().na2d_data_path,
        max_words_book: int        = 400,
        overlap_book:   int        = 50,
        max_words_rule: int        = 300,
        overlap_rule:   int        = 40,
    ) -> None:
        self.laws_dir         = Path(laws_dir)
        self.na2d_dir         = Path(na2d_dir)
        self._book_splitter   = self._make_splitter(max_words_book, overlap_book)
        self._ruling_splitter = self._make_splitter(max_words_rule, overlap_rule)

    # ── internal utilities ────────────────────────────────────────────────

    @staticmethod
    def _make_splitter(max_words: int, overlap: int) -> RecursiveCharacterTextSplitter:
        return RecursiveCharacterTextSplitter(
            chunk_size      = max_words,
            chunk_overlap   = overlap,
            length_function = lambda t: len(t.split()),
            separators      = ["\n\n", "\n", ".", " ", ""],
        )

    @staticmethod
    def _book_title(stem: str) -> str:
        for key, title in norm_regu.BOOK_TITLE_MAP.value.items():
            if key in stem:
                return title
        return stem

    @staticmethod
    def _ruling_metadata(text: str, folder: str, filename: str) -> Dict[str, Any]:
        header = text[:1000]
        meta: Dict[str, Any] = {
            "doc_type":       "ruling",
            "crime_category": folder,
            "source_file":    filename,
        }
        m = reg._CASE_NUM_RE.value.search(header)
        if m:
            meta["case_number"] = _to_western_digits(m.group(1).replace(" ", ""))
            meta["case_year"]   = _to_western_digits(m.group(2))
        m = reg._RULING_DATE_RE.value.search(header)
        if m:
            meta["ruling_date"] = m.group(0).replace("جلسة", "").strip()
        m = reg._CHAMBER_RE.value.search(header)
        if m:
            meta["chamber"] = m.group(1).strip()
        return meta

    def _get_amendment_metadata(
        self, raw_text: str, filename: str
    ) -> Optional[Dict[str, str]]:
        """
        Extract amendment law number, year, date, and type.

        **Filename is the primary source** for law number + year (same logic as
        :class:`AmendmentLawExtractor`).  The text header is used only as a
        fallback because it refers to the *original* law being amended, not
        the amending law.
        """
        law_num: Optional[str] = None
        year:    Optional[str] = None

        # ── Primary: filename ─────────────────────────────────────────────
        # Pattern 1: ..._num{N}_{YYYY}.txt
        m = re.search(r'_num([0-9]+)_([0-9]{4})\.txt$', filename, re.IGNORECASE)
        if m:
            law_num, year = m.group(1), m.group(2)

        # Pattern 2: ..._{N}_{YYYY}.txt
        if not law_num or not year:
            m = re.search(r'_([0-9]+)_([0-9]{4})\.txt$', filename, re.IGNORECASE)
            if m:
                law_num, year = m.group(1), m.group(2)

        # Pattern 3: any 4-digit year + any other number in filename
        if not law_num or not year:
            years = re.findall(r'((?:19|20)\d{2})', filename)
            nums  = [n for n in re.findall(r'(\d+)', filename) if n not in years]
            law_num = nums[-1] if nums else None
            year    = years[-1] if years else None

        # ── Fallback: scan text header ────────────────────────────────────
        if not law_num or not year:
            for m in reg.ORIGINAL_LAW_RE.value.finditer(raw_text):
                try:
                    a = int(_to_western_digits(m.group(1)))
                    b = int(_to_western_digits(m.group(2)))
                    year, law_num = (
                        (str(a), str(b)) if 1800 <= a <= 2100 else (str(b), str(a))
                    )
                    if 1800 <= int(year) <= 2100 and 0 < int(law_num) < 10000:
                        break
                except Exception:
                    continue

        if not law_num or not year:
            return None

        # ── Date ──────────────────────────────────────────────────────────
        date_m = reg._DATE_RE.value.search(raw_text)
        if date_m:
            day   = _to_western_digits(date_m.group(1) or "01").zfill(2)
            month = reg._MONTH_MAP.value.get(date_m.group(2), "01")
            yr    = _to_western_digits(date_m.group(3))
            adate = f"{yr}-{month}-{day}"
        else:
            adate = f"{year}-01-01"

        # ── Amendment type ─────────────────────────────────────────────────
        sample = raw_text[:800]
        if any(w in sample for w in norm_regu.ADDITION_KEYWORDS.value):
            atype = "addition"
        elif any(w in sample for w in norm_regu.DELETION_KEYWORDS.value):
            atype = "deletion"
        else:
            atype = "modification"

        return {
            "law_num":        law_num,
            "year":           year,
            "amendment_date": adate,
            "amendment_type": atype,
        }

    # ── public API ────────────────────────────────────────────────────────

    def get_chunks(self) -> List[Document]:
        """
        Return one :class:`Document` per article / preamble / amended-article
        / table row for every law folder in ``laws_dir``.
        """
        all_docs: List[Document] = []

        for folder in sorted(p for p in self.laws_dir.iterdir() if p.is_dir()):
            if folder.name not in norm_regu.FOLDER_TO_LAW_KEY.value:
                continue

            law_key   = norm_regu.FOLDER_TO_LAW_KEY.value[folder.name]
            law_title = norm_regu.LAW_KEY_TO_TITLE.value.get(law_key, law_key)

            # ── main law files (non-amendment) ────────────────────────────
            for fp in sorted(
                f for f in folder.glob("*.txt")
                if not fnmatch.fnmatch(f.name.lower(), "new*")
            ):
                raw = MultiEncodingTextLoader(fp).load()
                if not raw:
                    continue
                full_text = raw[0].page_content

                if "tables" in fp.name.lower():
                    table_docs = _split_into_tables(full_text, law_key, law_title, fp.name)
                    all_docs.extend(table_docs)
                    logger.info(f"  {fp.name} [table] -> {len(table_docs)} tables")

                else:
                    # Articles: exactly one Document per article
                    articles = _split_into_articles(full_text)
                    for art in articles:
                        chunk_type = (
                            "preamble" if art["article_number"] is None else "article"
                        )
                        all_docs.append(Document(
                            page_content = art["text"],
                            metadata     = {
                                "law_id":         law_key,
                                "law_title":      law_title,
                                "chunk_type":     chunk_type,
                                "article_number": art["article_number"],
                                "source_file":    fp.name,
                            },
                        ))
                    logger.info(f"  {fp.name} [article] -> {len(articles)} chunks")

            # ── amendment files (new*) ─────────────────────────────────────
            for af in sorted(folder.glob("new*.txt")):
                raw = MultiEncodingTextLoader(af).load()
                if not raw:
                    continue
                full_text = raw[0].page_content
                meta      = self._get_amendment_metadata(full_text, af.name)
                if not meta:
                    logger.warning(f"  SKIP {af.name} - no amendment metadata")
                    continue

                articles = _split_into_articles(full_text)
                for art in articles:
                    all_docs.append(Document(
                        page_content = art["text"],
                        metadata     = {
                            "law_id":                law_key,
                            "law_title":             law_title,
                            "chunk_type":            "amended_article",
                            "article_number":        art["article_number"],
                            "amendment_law_number":  meta["law_num"],
                            "amendment_date":        meta["amendment_date"],
                            "amendment_type":        meta["amendment_type"],
                            "source_file":           af.name,
                        },
                    ))
                logger.info(f"  {af.name} [amendment] -> {len(articles)} chunks")

        logger.info(f"laws total: {len(all_docs)} chunks")
        return all_docs

    def get_na2d_chunks(self) -> Dict[str, List[Document]]:
        """
        Return ruling and legal-principle chunks from the Na2d corpus.

        Returns a dict with two keys:

        * ``"rulings"``    — court rulings, split by ``_ruling_splitter``
        * ``"principles"`` — legal-principle books, split by ``_book_splitter``
        """
        ruling_docs:    List[Document] = []
        principle_docs: List[Document] = []

        for item in sorted(self.na2d_dir.iterdir()):

            if item.is_file() and item.suffix.lower() == ".txt":
                # Legal-principle book
                text = _read_file(item)
                if not text:
                    continue
                chunks = self._book_splitter.create_documents(
                    texts     = [text.strip()],
                    metadatas = [{
                        "doc_type":   "principle",
                        "book_title": self._book_title(item.stem),
                        "source_file": item.name,
                    }],
                )
                principle_docs.extend(chunks)
                logger.info(f"  BOOK {item.name} -> {len(chunks)} chunks")

            elif item.is_dir():
                # Court rulings folder
                folder_count = 0
                for fp in sorted(item.glob("*.txt")):
                    text = _read_file(fp)
                    if not text:
                        continue
                    chunks = self._ruling_splitter.create_documents(
                        texts     = [text.strip()],
                        metadatas = [self._ruling_metadata(text, item.name, fp.name)],
                    )
                    ruling_docs.extend(chunks)
                    folder_count += len(chunks)
                logger.info(f"  RULINGS {item.name} -> {folder_count} chunks")

        logger.info(
            f"na2d: rulings={len(ruling_docs)} | principles={len(principle_docs)}"
        )
        return {"rulings": ruling_docs, "principles": principle_docs}


# =============================================================================
# ENTRY POINTS
# =============================================================================

def get_chunks() -> List[Document]:
    """Convenience wrapper — returns law chunks from a default :class:`CorpusChunker`."""
    return CorpusChunker().get_chunks()


def get_na2d_chunks() -> Dict[str, List[Document]]:
    """Convenience wrapper — returns Na2d chunks from a default :class:`CorpusChunker`."""
    return CorpusChunker().get_na2d_chunks()