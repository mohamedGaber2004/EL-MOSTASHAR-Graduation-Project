from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator, Optional, Sequence

from langchain_community.document_loaders.base import BaseLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

ENCODINGS: Sequence[str] = ("utf-8", "utf-16", "cp1256", "windows-1256")


def _read_file(fp: Path) -> Optional[str]:
    """Try multiple encodings; return None if all fail."""
    for enc in ENCODINGS:
        try:
            return fp.read_text(encoding=enc)
        except Exception:
            continue
    return fp.read_bytes().decode("utf-8", errors="replace")


class MultiEncodingTextLoader(BaseLoader):
    """LangChain BaseLoader that tries multiple encodings for Arabic legal texts."""

    def __init__(self, file_path: str | Path) -> None:
        self.file_path = Path(file_path)

    def lazy_load(self) -> Iterator[Document]:
        text = _read_file(self.file_path)
        if not text or not text.strip():
            logger.warning(f"SKIP {self.file_path.name} - unreadable or empty")
            return
        yield Document(page_content=text, metadata={"source": str(self.file_path)})