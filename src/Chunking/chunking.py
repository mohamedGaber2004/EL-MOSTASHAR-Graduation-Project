from __future__ import annotations

import fnmatch
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.Config import get_settings
from src.Config.log_config import logging
from src.Utils import (
    MultiEncodingTextLoader,
    _normalize_article_no,
    _read_file,
    _to_western_digits,
)
from src.Chunking.chunking_enums import (
    AmendmentType,
    ChunkType,
    DocType,
    Na2dOutputKey,
    LawKeys,
    regex
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
_ArticleDict  = Dict[str, Optional[str]]   # {"article_number": ..., "text": ...}
_AmendMeta    = Dict[str, str]             # {"law_num", "year", "amendment_date", "amendment_type"}
_Na2dChunks   = Dict[str, List[Document]]  # keyed by Na2dOutputKey values


# =============================================================================
# Module-level helpers
# =============================================================================

def _build_table_docs(
    full_text:   str,
    law_id:      str,
    law_title:   str,
    source_file: str,
) -> List[Document]:
    """
    Split a tables file into one :class:`Document` per logical table.

    If no table-header markers are found the entire text is returned as a
    single document with ``table_number = "0"``.
    """
    matches = list(regex._TABLE_HEADER_RE.value.finditer(full_text))

    def _make_table_doc(text: str, table_number: str) -> Document:
        return Document(
            page_content=text.strip(),
            metadata={
                "law_id":       law_id,
                "law_title":    law_title,
                "chunk_type":   ChunkType.TABLE,
                "table_number": table_number,
                "source_file":  source_file,
            },
        )

    if not matches:
        return [_make_table_doc(full_text, "0")]

    docs: List[Document] = []
    for i, m in enumerate(matches):
        table_num = _to_western_digits(m.group(1))
        start     = m.start()
        end       = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        text      = full_text[start:end].strip()
        if text:
            docs.append(_make_table_doc(text, table_num))
    return docs

def _split_into_articles(full_text: str) -> List[_ArticleDict]:
    """
    Split raw law text into per-article dicts::

        {"article_number": "5", "text": "..."}

    Text before the first article header gets ``article_number = None``
    and is tagged as a preamble by the caller.
    """
    matches  = list(regex._ARTICLE_BOUNDARY_RE.value.finditer(full_text))
    articles: List[_ArticleDict] = []

    if not matches:
        return [{"article_number": None, "text": full_text.strip()}]

    # Preamble (text before the first article)
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

def _detect_amendment_type(sample: str) -> AmendmentType:
    """Return the :class:`AmendmentType` detected from *sample* (first ~800 chars)."""
    if any(w in sample for w in AmendmentType.ADDITION_KEYWORDS.value):
        return AmendmentType.ADDITION
    if any(w in sample for w in AmendmentType.DELETION_KEYWORDS.value):
        return AmendmentType.DELETION
    return AmendmentType.MODIFICATION

def _parse_amendment_meta(raw_text: str, filename: str) -> Optional[_AmendMeta]:
    """
    Extract amendment ``law_num``, ``year``, ``amendment_date``, and
    ``amendment_type`` from *filename* (primary) or *raw_text* header
    (fallback).

    Returns ``None`` when neither source yields a usable law number + year.
    """
    law_num, year = _extract_law_num_and_year(filename, raw_text)
    if not law_num or not year:
        return None

    amendment_date = _parse_amendment_date(raw_text, year)
    amendment_type = _detect_amendment_type(raw_text[:800])

    return {
        "law_num":        law_num,
        "year":           year,
        "amendment_date": amendment_date,
        "amendment_type": amendment_type.value,
    }

def _extract_law_num_and_year(
    filename: str,
    raw_text: str,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Try three filename patterns then fall back to the text header.
    Returns ``(law_num, year)``; either value may be ``None`` on failure.
    """
    # Pattern 1: ..._num{N}_{YYYY}.txt
    m = re.search(r'_num([0-9]+)_([0-9]{4})\.txt$', filename, re.IGNORECASE)
    if m:
        return m.group(1), m.group(2)

    # Pattern 2: ..._{N}_{YYYY}.txt
    m = re.search(r'_([0-9]+)_([0-9]{4})\.txt$', filename, re.IGNORECASE)
    if m:
        return m.group(1), m.group(2)

    # Pattern 3: any 4-digit year + any other number present in filename
    years = re.findall(r'((?:19|20)\d{2})', filename)
    nums  = [n for n in re.findall(r'(\d+)', filename) if n not in years]
    if nums and years:
        return nums[-1], years[-1]

    # Fallback: scan the first 1 000 chars of the file header
    for m in regex.ORIGINAL_LAW_RE.value.finditer(raw_text[:1000]):
        try:
            a = int(_to_western_digits(m.group(1)))
            b = int(_to_western_digits(m.group(2)))
            year   = str(a) if 1800 <= a <= 2100 else str(b)
            law_num = str(b) if 1800 <= a <= 2100 else str(a)
            if 1800 <= int(year) <= 2100 and 0 < int(law_num) < 10_000:
                return law_num, year
        except (ValueError, AttributeError):
            continue

    return None, None

def _parse_amendment_date(raw_text: str, fallback_year: str) -> str:
    """
    Extract an ISO-8601 date from *raw_text*.
    Returns ``"{fallback_year}-01-01"`` when no date pattern matches.
    """
    m = regex._DATE_RE.value.search(raw_text)
    if not m:
        return f"{fallback_year}-01-01"

    day   = _to_western_digits(m.group(1) or "01").zfill(2)
    month = regex._MONTH_MAP.value.get(m.group(2), "01")
    year  = _to_western_digits(m.group(3))
    return f"{year}-{month}-{day}"

def _extract_ruling_metadata(text: str, folder: str, filename: str) -> Dict[str, Any]:
    """
    Parse case number, ruling date, and chamber from the first 1 000 chars
    of a ruling file, supplementing with folder / filename provenance.
    """
    header = text[:1000]
    meta: Dict[str, Any] = {
        "doc_type":       DocType.RULING,
        "crime_category": folder,
        "source_file":    filename,
    }

    m = regex._CASE_NUM_RE.value.search(header)
    if m:
        meta["case_number"] = _to_western_digits(m.group(1).replace(" ", ""))
        meta["case_year"]   = _to_western_digits(m.group(2))

    m = regex._RULING_DATE_RE.value.search(header)
    if m:
        meta["ruling_date"] = m.group(0).replace("جلسة", "").strip()

    m = regex._CHAMBER_RE.value.search(header)
    if m:
        meta["chamber"] = m.group(1).strip()

    return meta

# =============================================================================
# CorpusChunker
# =============================================================================

class CorpusChunker:
    """
    Convert raw .txt law / ruling / principle files into LangChain Document chunks.

    Parameters
    ----------
    laws_dir:
        Root directory containing one sub-folder per law.
    na2d_dir:
        Root directory for the Na2d corpus (ruling folders + principle books).
    max_words_book:
        Approximate maximum *word* count per principle-book chunk.
    overlap_book:
        Word overlap between consecutive principle-book chunks.
    max_words_rule:
        Approximate maximum word count per ruling chunk.
    overlap_rule:
        Word overlap between consecutive ruling chunks.
    """

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
        self._book_splitter   = _make_word_splitter(max_words_book, overlap_book)
        self._ruling_splitter = _make_word_splitter(max_words_rule, overlap_rule)

    # ── public API ────────────────────────────────────────────────────────
    def get_chunks(self) -> List[Document]:
        """
        Return one :class:`Document` per article / preamble / amended-article
        / table for every recognised law folder under ``laws_dir``.
        """
        all_docs: List[Document] = []

        for folder in sorted(p for p in self.laws_dir.iterdir() if p.is_dir()):
            if folder.name not in LawKeys.FOLDER_TO_LAW_KEY.value:
                continue

            law_key   = LawKeys.FOLDER_TO_LAW_KEY.value[folder.name]
            law_title = LawKeys.LAW_KEY_TO_TITLE.value.get(law_key, law_key)

            all_docs.extend(self._process_main_files(folder, law_key, law_title))
            all_docs.extend(self._process_amendment_files(folder, law_key, law_title))

        logger.info("laws total: %d chunks", len(all_docs))
        return all_docs

    def get_na2d_chunks(self) -> _Na2dChunks:
        """
        Return ruling and legal-principle chunks from the Na2d corpus.

        Returns a dict with keys defined by :class:`Na2dOutputKey`:

        * ``"rulings"``    — court rulings split by ``_ruling_splitter``
        * ``"principles"`` — legal-principle books split by ``_book_splitter``
        """
        ruling_docs:    List[Document] = []
        principle_docs: List[Document] = []

        for item in sorted(self.na2d_dir.iterdir()):
            if item.is_file() and item.suffix.lower() == ".txt":
                principle_docs.extend(self._process_principle_book(item))
            elif item.is_dir():
                ruling_docs.extend(self._process_ruling_folder(item))

        logger.info(
            "na2d: rulings=%d | principles=%d",
            len(ruling_docs),
            len(principle_docs),
        )
        return {
            Na2dOutputKey.RULINGS:    ruling_docs,
            Na2dOutputKey.PRINCIPLES: principle_docs,
        }

    # ── private: law-folder processors ───────────────────────────────────
    def _process_main_files(
        self,
        folder:    Path,
        law_key:   str,
        law_title: str,
    ) -> List[Document]:
        """Process all non-amendment ``.txt`` files in *folder*."""
        docs: List[Document] = []

        for fp in sorted(
            f for f in folder.glob("*.txt")
            if not fnmatch.fnmatch(f.name.lower(), "new*")
        ):
            full_text = self._load_text(fp)
            if not full_text:
                continue

            if "tables" in fp.name.lower():
                table_docs = _build_table_docs(full_text, law_key, law_title, fp.name)
                docs.extend(table_docs)
                logger.info("  %s [table] -> %d tables", fp.name, len(table_docs))
            else:
                article_docs = self._articles_to_docs(
                    full_text, law_key, law_title, fp.name, chunk_type_override=None
                )
                docs.extend(article_docs)
                logger.info("  %s [article] -> %d chunks", fp.name, len(article_docs))

        return docs

    def _process_amendment_files(
        self,
        folder:    Path,
        law_key:   str,
        law_title: str,
    ) -> List[Document]:
        """Process all amendment (``new*.txt``) files in *folder*."""
        docs: List[Document] = []

        for af in sorted(folder.glob("new*.txt")):
            full_text = self._load_text(af)
            if not full_text:
                continue

            meta = _parse_amendment_meta(full_text, af.name)
            if not meta:
                logger.warning("  SKIP %s — no amendment metadata", af.name)
                continue

            for art in _split_into_articles(full_text):
                docs.append(Document(
                    page_content=art["text"],
                    metadata={
                        "law_id":               law_key,
                        "law_title":            law_title,
                        "chunk_type":           ChunkType.AMENDED_ARTICLE,
                        "article_number":       art["article_number"],
                        "amendment_law_number": meta["law_num"],
                        "amendment_date":       meta["amendment_date"],
                        "amendment_type":       meta["amendment_type"],
                        "source_file":          af.name,
                    },
                ))
            logger.info("  %s [amendment] -> %d chunks", af.name, len(docs))

        return docs

    def _articles_to_docs(
        self,
        full_text:           str,
        law_id:              str,
        law_title:           str,
        source_file:         str,
        chunk_type_override: Optional[ChunkType],
    ) -> List[Document]:
        """Convert article dicts to :class:`Document` objects."""
        docs: List[Document] = []
        for art in _split_into_articles(full_text):
            chunk_type = chunk_type_override or (
                ChunkType.PREAMBLE if art["article_number"] is None else ChunkType.ARTICLE
            )
            docs.append(Document(
                page_content=art["text"],
                metadata={
                    "law_id":         law_id,
                    "law_title":      law_title,
                    "chunk_type":     chunk_type,
                    "article_number": art["article_number"],
                    "source_file":    source_file,
                },
            ))
        return docs

    # ── private: Na2d processors ──────────────────────────────────────────
    def _process_principle_book(self, file: Path) -> List[Document]:
        """Chunk a single legal-principle book file."""
        text = _read_file(file)
        if not text:
            return []
        chunks = self._book_splitter.create_documents(
            texts=[text.strip()],
            metadatas=[{
                "doc_type":    DocType.PRINCIPLE,
                "book_title":  _resolve_book_title(file.stem),
                "source_file": file.name,
            }],
        )
        logger.info("  BOOK %s -> %d chunks", file.name, len(chunks))
        return chunks

    def _process_ruling_folder(self, folder: Path) -> List[Document]:
        """Chunk all ruling files inside one crime-category folder."""
        docs: List[Document] = []
        for fp in sorted(folder.glob("*.txt")):
            text = _read_file(fp)
            if not text:
                continue
            chunks = self._ruling_splitter.create_documents(
                texts=[text.strip()],
                metadatas=[_extract_ruling_metadata(text, folder.name, fp.name)],
            )
            docs.extend(chunks)
        logger.info("  RULINGS %s -> %d chunks", folder.name, len(docs))
        return docs

    # ── private: utilities ────────────────────────────────────────────────
    @staticmethod
    def _load_text(path: Path) -> Optional[str]:
        """Load *path* via :class:`MultiEncodingTextLoader`; return ``None`` on failure."""
        raw = MultiEncodingTextLoader(path).load()
        return raw[0].page_content if raw else None


# =============================================================================
# Module-level utilities
# =============================================================================
def _make_word_splitter(max_words: int, overlap: int) -> RecursiveCharacterTextSplitter:
    """
    Build a :class:`RecursiveCharacterTextSplitter` whose *chunk_size* is
    measured in **words** rather than characters.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size      = max_words,
        chunk_overlap   = overlap,
        length_function = lambda t: len(t.split()),
        separators      = ["\n\n", "\n", ".", " ", ""],
    )

def _resolve_book_title(stem: str) -> str:
    """Map a filename stem to its canonical book title, or return *stem* as-is."""
    for key, title in LawKeys.BOOK_TITLE_MAP.value.items():
        if key in stem:
            return title
    return stem

# =============================================================================
# Convenience entry points
# =============================================================================
def get_chunks() -> List[Document]:
    """Return law chunks from a default :class:`CorpusChunker`."""
    return CorpusChunker().get_chunks()


def get_na2d_chunks() -> _Na2dChunks:
    """Return Na2d chunks from a default :class:`CorpusChunker`."""
    return CorpusChunker().get_na2d_chunks()