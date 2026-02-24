from __future__ import annotations

import fnmatch
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

from src.Config.config import get_settings
from src.Utils.regex_utils import _to_western_digits , reg
from src.Utils.norm_and_regu import norm_regu
from src.Utils.text_loader import MultiEncodingTextLoader, _read_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


_article_splitter = CharacterTextSplitter(separator="\n\n", chunk_size=10_000, chunk_overlap=0)
_table_splitter   = CharacterTextSplitter(separator="\n\n\n\n", chunk_size=500_000, chunk_overlap=0)


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



    def _book_title(self, stem: str) -> str:
        for key, title in norm_regu.BOOK_TITLE_MAP.value.items():
            if key in stem:
                return title
        return stem


    def _ruling_metadata(self,text: str, folder: str, filename: str) -> Dict[str, Any]:
        header = text[:1000]
        meta   = {"doc_type": "ruling", "crime_category": folder, "source_file": filename}
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


    def _get_amendment_metadata(self,raw_text: str, filename: str) -> Optional[Dict[str, str]]:
        law_num, year = None, None
        for m in reg.ORIGINAL_LAW_RE.value.finditer(raw_text):
            try:
                a, b = int(_to_western_digits(m.group(1))), int(_to_western_digits(m.group(2)))
                year, num = (str(a), str(b)) if 1800 <= a <= 2100 else (str(b), str(a))
                if 1800 <= int(year) <= 2100 and 0 < int(num) < 10000:
                    law_num = num
                    break
            except Exception:
                continue
        if not law_num or not year:
            return None

        date_m = reg._DATE_RE.value.search(raw_text)
        if date_m:
            day   = _to_western_digits(date_m.group(1) or "01").zfill(2)
            month = reg._MONTH_MAP.value.get(date_m.group(2), "01")
            yr    = _to_western_digits(date_m.group(3))
            adate = f"{yr}-{month}-{day}"
        else:
            adate = f"{year}-01-01"

        sample = raw_text[:800]
        if any(w in sample for w in norm_regu.ADDITION_KEYWORDS.value):
            atype = "addition"
        elif any(w in sample for w in norm_regu.DELETION_KEYWORDS.value):
            atype = "deletion"
        else:
            atype = "modification"

        return {"law_num": law_num, "year": year, "amendment_date": adate, "amendment_type": atype}


    def _make_splitter(self,max_words: int, overlap: int) -> RecursiveCharacterTextSplitter:
        return RecursiveCharacterTextSplitter(
            chunk_size     = max_words,
            chunk_overlap  = overlap,
            length_function= lambda t: len(t.split()),
            separators     = ["\n\n", "\n", ".", " ", ""],
        )


    def get_chunks(self) -> List[Document]:
        all_docs: List[Document] = []

        for folder in sorted(p for p in self.laws_dir.iterdir() if p.is_dir()):
            if folder.name not in norm_regu.FOLDER_TO_LAW_KEY.value:
                continue
            law_key   = norm_regu.FOLDER_TO_LAW_KEY.value[folder.name]
            law_title = norm_regu.LAW_KEY_TO_TITLE.value.get(law_key, law_key)

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
                meta = self._get_amendment_metadata(raw[0].page_content, af.name)
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
        ruling_docs:    List[Document] = []
        principle_docs: List[Document] = []

        for item in sorted(self.na2d_dir.iterdir()):
            if item.is_file() and item.suffix.lower() == ".txt":
                text = _read_file(item)
                if not text:
                    continue
                chunks = self._book_splitter.create_documents(
                    texts     = [text.strip()],
                    metadatas = [{"doc_type": "principle", "book_title": self._book_title(item.stem), "source_file": item.name}],
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
                        metadatas = [self._ruling_metadata(text, item.name, fp.name)],
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