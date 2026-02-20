from __future__ import annotations
import logging
from pathlib import Path
from typing import Iterator, Optional, Sequence

from langchain_community.document_loaders.base import BaseLoader
from langchain_core.documents import Document

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