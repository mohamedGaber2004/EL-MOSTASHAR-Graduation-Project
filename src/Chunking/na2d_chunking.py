from __future__ import annotations

from src.Config.config import na2d_data_path
from src.Utils.arabic_utils import _stable_id, _to_western_digits

import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from langchain_community.document_loaders.base import BaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# SECTION 1: CONSTANTS
# =============================================================================

NA2D_DIR = na2d_data_path

# Root-level txt files are treated as principle books
BOOK_TITLE_MAP: Dict[str, str] = {
    "mbade2_ultra_clean":                                   "مبادئ محكمة النقض",
    "1762919079675_ultra_clean":                            "مبادئ محكمة النقض - المجلد الثاني",
    "2019-2018_المستحدث-من-المبادئ-الصادرة-من-الدوائر-الجنائية-بمحكمة-النقض_ultra_clean":
        "المستحدث من المبادئ 2018-2019",
}

# Regex patterns for ruling metadata extraction
_CASE_NUM_RE = re.compile(
    r"(?:الطعن|طعن)\s+رق[مم]\s+([٠-٩0-9]+(?:\s*[,،]\s*[٠-٩0-9]+)*)\s+لسنة\s+([٠-٩0-9]+)",
    re.UNICODE,
)
_DATE_RE = re.compile(
    r"جلسة\s+(?:[٠-٩0-9]+\s*/\s*[٠-٩0-9]+\s*/\s*[٠-٩0-9]{2,4}"
    r"|[٠-٩0-9]+\s+"
    r"(?:يناير|فبراير|مارس|أبريل|مايو|يونيو|يوليو|أغسطس|سبتمبر|أكتوبر|نوفمبر|ديسمبر)"
    r"\s+[٠-٩0-9]{4})",
    re.UNICODE,
)
_CHAMBER_RE = re.compile(
    r"(?:الدائرة|دائرة)\s+([^\n،,]+?)(?=\n|،|,|$)",
    re.UNICODE,
)

ENCODINGS = ("utf-8", "utf-16", "cp1256", "windows-1256")


# =============================================================================
# SECTION 2: SHARED UTILITY
# =============================================================================

def _read_file(fp: Path) -> Optional[str]:
    """Try multiple encodings; return None if all fail."""
    for enc in ENCODINGS:
        try:
            return fp.read_text(encoding=enc)
        except Exception:
            continue
    try:
        return fp.read_bytes().decode("utf-8", errors="replace")
    except Exception:
        logger.warning(f"[SKIP] {fp.name} — unreadable")
        return None


# =============================================================================
# SECTION 3: RULING METADATA EXTRACTOR
# Extracts structured metadata from the header of a ruling txt file.
# =============================================================================

def _extract_ruling_metadata(text: str, folder_name: str, filename: str) -> Dict[str, Any]:
    """
    Parse the first 1000 characters of a ruling for key metadata.

    Returns a dict with:
        case_number, case_year, ruling_date, chamber, crime_category
    """
    header = text[:1000]

    # Case number + year: "الطعن رقم 123 لسنة 45"
    case_number, case_year = None, None
    m = _CASE_NUM_RE.search(header)
    if m:
        case_number = _to_western_digits(m.group(1).replace(" ", ""))
        case_year   = _to_western_digits(m.group(2))

    # Session date: "جلسة 12/3/2020" or "جلسة 12 مارس 2020"
    ruling_date = None
    m = _DATE_RE.search(header)
    if m:
        ruling_date = m.group(0).replace("جلسة", "").strip()

    # Chamber: "الدائرة الجنائية الأولى"
    chamber = None
    m = _CHAMBER_RE.search(header)
    if m:
        chamber = m.group(1).strip()

    return {
        "doc_type":       "ruling",
        "crime_category": folder_name,          # Arabic folder name as-is
        "case_number":    case_number,
        "case_year":      case_year,
        "ruling_date":    ruling_date,
        "chamber":        chamber,
        "source_file":    filename,
    }


# =============================================================================
# SECTION 4: COURT RULING LOADER
# One txt file = one ruling → one base Document (may be sub-split downstream).
# =============================================================================

class CourtRulingLoader(BaseLoader):
    """
    Loads a single court ruling txt file as a Document.
    Metadata is extracted automatically from folder name + file content.
    """

    def __init__(self, file_path: Path, folder_name: str) -> None:
        self.file_path   = file_path
        self.folder_name = folder_name

    def lazy_load(self) -> Iterator[Document]:
        text = _read_file(self.file_path)
        if not text or not text.strip():
            return

        metadata = _extract_ruling_metadata(
            text        = text,
            folder_name = self.folder_name,
            filename    = self.file_path.name,
        )
        metadata["chunk_id"] = _stable_id(
            "ruling", self.folder_name, self.file_path.stem
        )

        yield Document(page_content=text.strip(), metadata=metadata)


# =============================================================================
# SECTION 5: RULING SPLITTER
# Splits rulings that exceed the word budget while preserving context.
# Each sub-chunk inherits all ruling metadata + adds sub_index + chunk_id.
# =============================================================================

class CourtRulingSplitter(TextSplitter):
    """
    LangChain TextSplitter for court ruling documents.

    Short rulings (≤ max_words) are kept whole.
    Long rulings are split by RecursiveCharacterTextSplitter with a
    word-count length function, so token limits are respected.

    Every sub-chunk inherits the parent ruling's metadata and gets its own
    unique chunk_id.
    """

    def __init__(self, max_words: int = 300, overlap_words: int = 40, **kwargs) -> None:
        super().__init__(**kwargs)
        self._max_words = max_words
        self._word_splitter = RecursiveCharacterTextSplitter(
            chunk_size     = max_words,
            chunk_overlap  = overlap_words,
            length_function= lambda t: len(t.split()),
            separators     = ["\n\n", "\n", ".", " ", ""],
        )

    def split_text(self, text: str) -> List[str]:
        if len(text.split()) <= self._max_words:
            return [text]
        return self._word_splitter.split_text(text)

    def split_documents(self, documents: List[Document]) -> List[Document]:
        result: List[Document] = []
        for doc in documents:
            parts = self.split_text(doc.page_content)
            for i, part in enumerate(parts):
                meta = {
                    **doc.metadata,
                    "sub_index":  i,
                    "chunk_type": "ruling_chunk" if len(parts) > 1 else "ruling",
                    "word_count": len(part.split()),
                    "chunk_id":   _stable_id(
                        doc.metadata.get("chunk_id", ""), str(i)
                    ),
                }
                result.append(Document(page_content=part.strip(), metadata=meta))
        return result


# =============================================================================
# SECTION 6: PRINCIPLES BOOK LOADER + SPLITTER
# The 3 root-level txt files are treated as reference books.
# They are chunked with overlap like laws (no article markers expected).
# =============================================================================

class PrinciplesBookLoader(BaseLoader):
    """
    Loads one of the three principles books and tags it with book metadata.
    """

    def __init__(self, file_path: Path) -> None:
        self.file_path = file_path

    def _resolve_title(self) -> str:
        stem = self.file_path.stem
        for key, title in BOOK_TITLE_MAP.items():
            if key in stem:
                return title
        return stem   # fallback: use filename stem

    def lazy_load(self) -> Iterator[Document]:
        text = _read_file(self.file_path)
        if not text or not text.strip():
            return
        yield Document(
            page_content=text.strip(),
            metadata={
                "doc_type":   "principle",
                "chunk_type": "principle",
                "book_title": self._resolve_title(),
                "source_file": self.file_path.name,
                "chunk_id":   _stable_id("book", self.file_path.stem),
            },
        )


class PrinciplesBookSplitter(TextSplitter):
    """
    Splits principle books into overlapping word-bounded chunks.
    Each chunk inherits book metadata.
    """

    def __init__(self, max_words: int = 400, overlap_words: int = 50, **kwargs) -> None:
        super().__init__(**kwargs)
        self._word_splitter = RecursiveCharacterTextSplitter(
            chunk_size     = max_words,
            chunk_overlap  = overlap_words,
            length_function= lambda t: len(t.split()),
            separators     = ["\n\n", "\n", ".", " ", ""],
        )

    def split_text(self, text: str) -> List[str]:
        return self._word_splitter.split_text(text)

    def split_documents(self, documents: List[Document]) -> List[Document]:
        result: List[Document] = []
        for doc in documents:
            parts = self.split_text(doc.page_content)
            for i, part in enumerate(parts):
                meta = {
                    **doc.metadata,
                    "sub_index":  i,
                    "word_count": len(part.split()),
                    "chunk_id":   _stable_id(
                        doc.metadata.get("chunk_id", ""), str(i)
                    ),
                }
                result.append(Document(page_content=part.strip(), metadata=meta))
        return result


# =============================================================================
# SECTION 7: MAIN PIPELINE
# =============================================================================

def build_na2d_documents(
    na2d_dir:       str | Path = NA2D_DIR,
    max_words_book: int        = 400,
    overlap_book:   int        = 50,
    max_words_rule: int        = 300,
    overlap_rule:   int        = 40,
) -> Dict[str, List[Document]]:
    """
    Build Documents for the na2d corpus.

    Returns a dict with two keys so callers can build separate indices
    or merge them as needed::

        {
            "rulings":    List[Document],   # court rulings from 11 folders
            "principles": List[Document],   # 3 principle books
        }

    Example::

        from src.Na2d.na2d_chunking import build_na2d_documents

        corpus = build_na2d_documents()
        ruling_store    = FAISS.from_documents(corpus["rulings"],    embeddings)
        principle_store = FAISS.from_documents(corpus["principles"], embeddings)
    """
    na2d_dir = Path(na2d_dir)
    if not na2d_dir.exists():
        raise FileNotFoundError(f"na2d directory not found: {na2d_dir}")

    ruling_splitter    = CourtRulingSplitter(max_words=max_words_rule, overlap_words=overlap_rule)
    principles_splitter = PrinciplesBookSplitter(max_words=max_words_book, overlap_words=overlap_book)

    ruling_docs: List[Document]    = []
    principle_docs: List[Document] = []

    for item in sorted(na2d_dir.iterdir()):

        # ── Root-level txt files → principle books ─────────────────────
        if item.is_file() and item.suffix.lower() == ".txt":
            raw = PrinciplesBookLoader(item).load()
            if not raw:
                continue
            chunks = principles_splitter.split_documents(raw)
            principle_docs.extend(chunks)
            logger.info(f"  [BOOK] {item.name} → {len(chunks)} chunks")

        # ── Sub-folders → court ruling collections ─────────────────────
        elif item.is_dir():
            folder_name  = item.name          # Arabic category name preserved
            folder_count = 0
            txt_files    = sorted(item.glob("*.txt"))

            if not txt_files:
                logger.warning(f"  [EMPTY] {item.name} — no txt files found")
                continue

            for fp in txt_files:
                raw = CourtRulingLoader(fp, folder_name).load()
                if not raw:
                    continue
                chunks = ruling_splitter.split_documents(raw)
                ruling_docs.extend(chunks)
                folder_count += len(chunks)

            logger.info(
                f"  [RULINGS] {item.name} — {len(txt_files)} rulings "
                f"→ {folder_count} chunks"
            )

    logger.info(
        f"\n✓ na2d build complete | "
        f"rulings: {len(ruling_docs)} chunks | "
        f"principles: {len(principle_docs)} chunks"
    )

    return {
        "rulings":    ruling_docs,
        "principles": principle_docs,
    }


# =============================================================================
# SECTION 8: ENTRY POINT
# =============================================================================

def get_na2d_chunks() -> Dict[str, List[Document]]:
    print(f"\nna2d directory   : {NA2D_DIR}")
    print(f"Book chunk size  : 400 words | overlap: 50")
    print(f"Ruling chunk size: 300 words | overlap: 40\n")
    return build_na2d_documents()



