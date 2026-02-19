from __future__ import annotations

from src.Config.config import DataPath
from src.Utils.arabic_utils import (
    _normalize_article_no,
    _ARTICLE_MARK_RE,
    _to_western_digits,
    _stable_id,
    _DATE_RE,
    _MONTH_MAP,
    ORIGINAL_LAW_RE,
    DEFAULT_FOLDER_TO_LAW_KEY,
    LAW_KEY_TO_TITLE,
)

import fnmatch
import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from langchain_community.document_loaders.base import BaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# SECTION 1: FILE LOADER
# =============================================================================

class MultiEncodingTextLoader(BaseLoader):
    """
    LangChain BaseLoader that tries multiple encodings before giving up.
    Designed for Arabic legal texts that may be UTF-8, UTF-16, or Windows-1256.
    """

    ENCODINGS: Sequence[str] = ("utf-8", "utf-16", "cp1256", "windows-1256")

    def __init__(self, file_path: str | Path) -> None:
        self.file_path = Path(file_path)

    def lazy_load(self) -> Iterator[Document]:
        text: Optional[str] = None
        for enc in self.ENCODINGS:
            try:
                text = self.file_path.read_text(encoding=enc)
                break
            except Exception:
                continue

        if text is None:
            try:
                text = self.file_path.read_bytes().decode("utf-8", errors="replace")
            except Exception:
                logger.warning(f"[SKIP] {self.file_path.name} — unreadable")
                return

        if not text.strip():
            logger.warning(f"[SKIP] {self.file_path.name} — empty")
            return

        yield Document(
            page_content=text,
            metadata={"source": str(self.file_path)},
        )


# =============================================================================
# SECTION 2: ARTICLE SPLITTER
# =============================================================================

def _parse_articles(raw_text: str) -> List[Dict[str, Optional[str]]]:
    """
    Internal helper: split raw text into article dicts.
    Returns: [{"article_number": str | None, "text": str}]
    """
    text    = raw_text.strip()
    matches = list(_ARTICLE_MARK_RE.finditer(text))

    if not matches:
        return [{"article_number": None, "text": text}]

    articles: List[Dict[str, Optional[str]]] = []

    if matches[0].start() > 0:
        preamble = text[: matches[0].start()].strip()
        if preamble:
            articles.append({"article_number": None, "text": preamble})

    for i, m in enumerate(matches):
        start      = m.end()
        end        = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body       = text[start:end].strip()
        article_no = _normalize_article_no(m.group("num") or "")
        if body:
            articles.append({"article_number": article_no, "text": body})

    return articles


class LegalArticleSplitter(TextSplitter):
    """
    LangChain TextSplitter for Arabic legal texts.

    - Splits at article boundaries (المادة / مادة markers).
    - Sub-splits articles exceeding `max_words` using
      RecursiveCharacterTextSplitter with a word-count length function,
      preserving context via `overlap_words`.
    - Injects full legal metadata into every Document.

    Usage::

        splitter = LegalArticleSplitter(max_words=400, overlap_words=50)
        docs = splitter.create_documents([text], metadatas=[law_meta_dict])
    """

    def __init__(self, max_words: int = 400, overlap_words: int = 50, **kwargs) -> None:
        super().__init__(**kwargs)
        self._max_words    = max_words
        self._word_splitter = RecursiveCharacterTextSplitter(
            chunk_size     = max_words,
            chunk_overlap  = overlap_words,
            length_function= lambda t: len(t.split()),
            separators     = ["\n\n", "\n", " ", ""],
        )

    # ------------------------------------------------------------------
    # TextSplitter protocol
    # ------------------------------------------------------------------

    def split_text(self, text: str) -> List[str]:
        """Return bare article body strings (metadata-free)."""
        return [
            a["text"]
            for a in _parse_articles(text)
            if a.get("text")
        ]

    def create_documents(
        self,
        texts:     List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Document]:
        """
        Override to inject per-article metadata and sub-split long articles.

        Args:
            texts:     One full law text per source file.
            metadatas: One law-level metadata dict per text. Expected keys:
                         law_id, law_title, chunk_type,
                         (optional) amendment_law_number, amendment_date,
                         amendment_type.
        """
        documents: List[Document] = []

        for i, full_text in enumerate(texts):
            law_meta   = (metadatas[i] if metadatas else {}) or {}
            law_id     = law_meta.get("law_id", "")
            chunk_type = law_meta.get("chunk_type", "article")
            is_amended = chunk_type == "amended_article"

            for article in _parse_articles(full_text):
                article_no = article.get("article_number")
                body       = (article.get("text") or "").strip()
                if not body:
                    continue

                full_article_text = (
                    f"المادة {article_no}\n{body}" if article_no else body
                )

                # Sub-split only if the article exceeds the word budget
                parts = (
                    self._word_splitter.split_text(full_article_text)
                    if len(full_article_text.split()) > self._max_words
                    else [full_article_text]
                )

                for j, part in enumerate(parts):
                    chunk_id = _stable_id(
                        law_id, article_no or "preamble", chunk_type, j
                    )
                    metadata: Dict[str, Any] = {
                        # ── inherited from law-level ──────────────────
                        **law_meta,
                        # ── chunk identity ────────────────────────────
                        "chunk_id":   chunk_id,
                        "chunk_type": chunk_type,
                        "sub_index":  j,
                        # ── article ───────────────────────────────────
                        "article_number": article_no,
                        "is_amended":     is_amended,
                        # ── stats ─────────────────────────────────────
                        "word_count": len(part.split()),
                    }
                    documents.append(
                        Document(page_content=part.strip(), metadata=metadata)
                    )

        return documents


# =============================================================================
# SECTION 3: AMENDMENT METADATA EXTRACTOR
# =============================================================================

def _parse_law_num_and_year(a: str, b: str) -> Tuple[str, str]:
    aw, bw = int(_to_western_digits(a)), int(_to_western_digits(b))
    if aw >= 1900:
        return str(bw), str(aw)
    return str(aw), str(bw)


def _amendment_meta_from_filename(
    filename: str,
) -> Tuple[Optional[str], Optional[str]]:
    m = re.search(r"_num([0-9]+)_([0-9]{4})\.txt$", filename, re.IGNORECASE)
    if m:
        return m.group(1), m.group(2)
    m = re.search(r"_([0-9]+)_([0-9]{4})\.txt$", filename, re.IGNORECASE)
    if m:
        return m.group(1), m.group(2)
    years = re.findall(r"((?:19|20)\d{2})", filename)
    nums  = [n for n in re.findall(r"(\d+)", filename) if n not in years]
    return (nums[-1] if nums else None), (years[-1] if years else None)


def _amendment_meta_from_header(
    raw_text: str,
) -> Tuple[Optional[str], Optional[str]]:
    for m in ORIGINAL_LAW_RE.finditer(raw_text):
        try:
            law_num, year = _parse_law_num_and_year(m.group(1), m.group(2))
            if 1800 <= int(year) <= 2100 and 0 < int(law_num) < 10000:
                return law_num, year
        except Exception:
            continue
    return None, None


def _detect_amendment_type(raw_text: str) -> str:
    sample = raw_text[:800]
    if any(w in sample for w in ["تضاف", "يضاف", "أضيف", "إضافة", "جديدة", "جديد"]):
        return "addition"
    if any(w in sample for w in ["يلغى", "ألغي", "يحذف", "حذف", "إلغاء"]):
        return "deletion"
    return "modification"


def _extract_amendment_date(raw_text: str, year: str) -> str:
    m = _DATE_RE.search(raw_text)
    if m:
        day   = _to_western_digits(m.group(1) or "01").zfill(2)
        month = _MONTH_MAP.get(m.group(2), "01")
        yr    = _to_western_digits(m.group(3))
        return f"{yr}-{month}-{day}"
    return f"{year}-01-01"


def _get_amendment_metadata(
    raw_text: str, filename: str
) -> Optional[Dict[str, str]]:
    law_num, year = _amendment_meta_from_filename(filename)
    if not law_num or not year:
        law_num, year = _amendment_meta_from_header(raw_text)
    if not law_num or not year:
        return None
    return {
        "law_num":        law_num,
        "year":           year,
        "amendment_date": _extract_amendment_date(raw_text, year),
        "amendment_type": _detect_amendment_type(raw_text),
    }


# =============================================================================
# SECTION 4: MAIN PIPELINE
# =============================================================================

def build_rag_documents(
    root_dir:          str,
    folder_to_law_key: Optional[Dict[str, str]] = None,
    max_words:         int = 400,
    overlap_words:     int = 50,
) -> List[Document]:
    """
    Full RAG pipeline using LangChain loaders and splitters.
    Returns a flat list of LangChain Document objects.

    Per folder:
      1. Main *.txt files (new* excluded) → chunk_type = 'article'
      2. new*.txt amendment files         → chunk_type = 'amended_article'

    Args:
        root_dir:          Path to Unstructured_Data root directory.
        folder_to_law_key: {folder_name: law_id}.
                           Defaults to DEFAULT_FOLDER_TO_LAW_KEY.
        max_words:         Max words per Document (tune to your embedding model).
        overlap_words:     Word overlap between sub-chunks of long articles.

    Returns:
        List[Document] — pass directly to any LangChain vector store.

    Example::

        docs = build_rag_documents("Datasets/Unstructured_Data")
        vectorstore = FAISS.from_documents(docs, embeddings)
    """
    if folder_to_law_key is None:
        folder_to_law_key = DEFAULT_FOLDER_TO_LAW_KEY

    root     = Path(root_dir)
    splitter = LegalArticleSplitter(max_words=max_words, overlap_words=overlap_words)
    all_docs: List[Document] = []

    for folder in sorted(p for p in root.iterdir() if p.is_dir()):

        if folder.name not in folder_to_law_key:
            orphans = sorted(folder.glob("new*.txt"))
            if orphans:
                logger.warning(
                    f"'{folder.name}' has {len(orphans)} new* file(s) "
                    f"but is NOT in folder_to_law_key — skipped"
                )
            continue

        law_key      = folder_to_law_key[folder.name]
        law_title    = LAW_KEY_TO_TITLE.get(law_key, law_key)
        folder_count = 0

        # ── 1. Main law files (exclude new*) ──────────────────────────────
        main_files = sorted(
            f for f in folder.glob("*.txt")
            if not fnmatch.fnmatch(f.name.lower(), "new*")
        )

        for fp in main_files:
            raw_docs = MultiEncodingTextLoader(fp).load()
            if not raw_docs:
                continue

            docs = splitter.create_documents(
                texts     = [raw_docs[0].page_content],
                metadatas = [{"law_id": law_key, "law_title": law_title,
                              "chunk_type": "article"}],
            )
            all_docs.extend(docs)
            folder_count += len(docs)
            logger.info(f"  {fp.name} → {len(docs)} documents")

        # ── 2. Amendment files (new*) ──────────────────────────────────────
        for af in sorted(folder.glob("new*.txt")):
            raw_docs = MultiEncodingTextLoader(af).load()
            if not raw_docs:
                continue

            text = raw_docs[0].page_content
            meta = _get_amendment_metadata(text, af.name)
            if not meta:
                logger.warning(
                    f"  [SKIP] {af.name} — could not extract amendment "
                    f"law number/year"
                )
                continue

            docs = splitter.create_documents(
                texts     = [text],
                metadatas = [{"law_id":               law_key,
                              "law_title":             law_title,
                              "chunk_type":            "amended_article",
                              "amendment_law_number":  meta["law_num"],
                              "amendment_date":        meta["amendment_date"],
                              "amendment_type":        meta["amendment_type"]}],
            )
            all_docs.extend(docs)
            folder_count += len(docs)
            logger.info(
                f"  {af.name} → {len(docs)} documents "
                f"| law={meta['law_num']}/{meta['year']} "
                f"type={meta['amendment_type']}"
            )

        logger.info(
            f"✓ '{folder.name}' ({law_key}) — "
            f"{folder_count} docs | total: {len(all_docs)}"
        )

    return all_docs


# =============================================================================
# SECTION 5: ENTRY POINT
# =============================================================================

def get_chunks() -> List[Document]:
    ROOT_DIR = DataPath
    print(f"\nRoot directory  : {ROOT_DIR}")
    print(f"Max words/chunk : 400")
    print(f"Overlap words   : 50\n")
    return build_rag_documents(
        root_dir      = ROOT_DIR,
        max_words     = 400,
        overlap_words = 50,
    )