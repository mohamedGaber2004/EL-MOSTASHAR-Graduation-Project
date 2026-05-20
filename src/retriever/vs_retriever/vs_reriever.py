from __future__ import annotations

import logging
from typing import Any

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from src.Vectorstore.vs_builder_enums import ChunkType, DocType
from src.Vectorstore.vector_store_builder import _build_filter

logger = logging.getLogger(__name__)


# =============================================================================
# Dense retriever (FAISS only)
# =============================================================================
def get_dense_retriever(vs: FAISS,*,k: int = 5,law_id: str | None = None,chunk_type: ChunkType | str | None = None,doc_type: DocType | str | None = None) -> BaseRetriever:
    f = _build_filter(
        law_id=law_id,
        chunk_type=chunk_type.value if isinstance(chunk_type, ChunkType) else chunk_type,
        doc_type=doc_type.value if isinstance(doc_type, DocType) else doc_type,
    )
    search_kwargs: dict[str, Any] = {"k": k}
    if f:
        search_kwargs["filter"] = f
    return vs.as_retriever(search_kwargs=search_kwargs)

# =============================================================================
# Sparse retriever (BM25)
# =============================================================================
def get_sparse_retriever(docs: list[Document],*,k: int = 5,law_id: str | None = None,chunk_type: ChunkType | str | None = None,doc_type: DocType | str | None = None) -> BM25Retriever:

    filtered = _filter_docs(docs, law_id=law_id, chunk_type=chunk_type, doc_type=doc_type)
    if not filtered:
        raise ValueError(
            "BM25 retriever received an empty document list after filtering. "
            "Check your law_id / chunk_type / doc_type constraints."
        )
    retriever = BM25Retriever.from_documents(filtered)
    retriever.k = k
    logger.debug("BM25 retriever built over %d documents.", len(filtered))
    return retriever

# =============================================================================
# Hybrid retriever (BM25 + FAISS via EnsembleRetriever)
# =============================================================================
def get_hybrid_retriever(
    vs: FAISS,
    docs: list[Document],
    *,
    k: int = 5,
    dense_weight: float = 0.6,
    sparse_weight: float = 0.4,
    law_id: str | None = None,
    chunk_type: ChunkType | str | None = None,
    doc_type: DocType | str | None = None,
) -> EnsembleRetriever:

    if not 0 < dense_weight and not 0 < sparse_weight:
        raise ValueError("Both dense_weight and sparse_weight must be > 0.")

    total = dense_weight + sparse_weight
    weights = [dense_weight / total, sparse_weight / total]

    dense  = get_dense_retriever(vs,   k=k, law_id=law_id, chunk_type=chunk_type, doc_type=doc_type)
    sparse = get_sparse_retriever(docs, k=k, law_id=law_id, chunk_type=chunk_type, doc_type=doc_type)

    hybrid = EnsembleRetriever(
        retrievers=[dense, sparse],
        weights=weights,
    )
    logger.info(
        "Hybrid retriever ready — dense %.2f / sparse %.2f, k=%d.",
        weights[0], weights[1], k,
    )
    return hybrid

# =============================================================================
# Internal helpers
# =============================================================================

def _filter_docs(
    docs: list[Document],
    *,
    law_id: str | None,
    chunk_type: ChunkType | str | None,
    doc_type: DocType | str | None,
) -> list[Document]:
    """Return the subset of *docs* matching all non-None metadata criteria."""
    criteria = _build_filter(
        law_id=law_id,
        chunk_type=chunk_type.value if isinstance(chunk_type, ChunkType) else chunk_type,
        doc_type=doc_type.value if isinstance(doc_type, DocType) else doc_type,
    )
    if not criteria:
        return docs
    return [
        d for d in docs
        if all(d.metadata.get(k) == v for k, v in criteria.items())
    ]