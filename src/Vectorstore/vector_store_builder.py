from __future__ import annotations

import logging
from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from src.Vectorstore.vs_builder_enums import DocType

logger = logging.getLogger(__name__)


# =============================================================================
# Internal helpers
# =============================================================================

def _build_filter(**kwargs: Any) -> dict[str, Any]:
    """Return a metadata-filter dict, dropping keys whose value is *None*."""
    return {k: v for k, v in kwargs.items() if v is not None}


# =============================================================================
# Build & persist
# =============================================================================

def build_vector_store(
    docs: list[Document],
    embeddings,
    index_path: str = "Datasets/na2d_faiss_index",
) -> FAISS:
    if not docs:
        raise ValueError("Cannot build a vector store from an empty document list.")

    logger.info("Indexing %d Na2d documents …", len(docs))
    vs = FAISS.from_documents(docs, embeddings)
    vs.save_local(index_path)
    logger.info("FAISS index saved → %s", index_path)
    return vs


# =============================================================================
# Load
# =============================================================================

def load_vector_store(index_path: str, embeddings) -> FAISS:
    """Load a previously persisted FAISS index from *index_path*.

    Args:
        index_path: Directory produced by :func:`build_vector_store`.
        embeddings: Must be the *same* embeddings object used at build time.
    """
    vs = FAISS.load_local(
        index_path,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    logger.info("FAISS index loaded ← %s", index_path)
    return vs


# =============================================================================
# Search
# =============================================================================

def search(
    vs: FAISS,
    query: str,
    *,
    k: int = 5,
    doc_type:        DocType | str | None = None,
    crime_category:  str | None = None,
    book_title:      str | None = None,
) -> list[Document]:
    f = _build_filter(
        doc_type       = doc_type.value if isinstance(doc_type, DocType) else doc_type,
        crime_category = crime_category,
        book_title     = book_title,
    )
    return vs.similarity_search(query, k=k, filter=f or None)


def search_with_scores(
    vs: FAISS,
    query: str,
    *,
    k: int = 5,
    doc_type:        DocType | str | None = None,
    crime_category:  str | None = None,
    book_title:      str | None = None,
) -> list[tuple[Document, float]]:

    f = _build_filter(
        doc_type       = doc_type.value if isinstance(doc_type, DocType) else doc_type,
        crime_category = crime_category,
        book_title     = book_title,
    )
    return vs.similarity_search_with_score(query, k=k, filter=f or None)