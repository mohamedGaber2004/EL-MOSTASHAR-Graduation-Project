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
def get_dense_retriever(
    vs: FAISS,
    *,
    k: int = 5,
    law_id: str | None = None,
    chunk_type: ChunkType | str | None = None,
    doc_type: DocType | str | None = None,
) -> BaseRetriever:
    """Wrap a FAISS store as a LangChain retriever.

    Args:
        vs:         A loaded or freshly built FAISS store.
        k:          Number of documents to retrieve per query.
        law_id:     Restrict retrieval to one law identifier.
        chunk_type: Restrict by :class:`~vector_store.ChunkType`.
        doc_type:   Restrict by :class:`~vector_store.DocType`.

    Returns:
        A LangChain ``BaseRetriever`` backed by FAISS similarity search.
    """
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
def get_sparse_retriever(
    docs: list[Document],
    *,
    k: int = 5,
    law_id: str | None = None,
    chunk_type: ChunkType | str | None = None,
    doc_type: DocType | str | None = None,
) -> BM25Retriever:
    """Build an in-memory BM25 retriever over *docs*.

    Pre-filters *docs* by the supplied metadata constraints before
    initialising BM25, so the term-frequency statistics are computed
    only over the relevant subset.

    Args:
        docs:       Full corpus of LangChain Documents.
        k:          Number of documents to return per query.
        law_id:     Keep only docs whose metadata ``law_id`` matches.
        chunk_type: Keep only docs whose metadata ``chunk_type`` matches.
        doc_type:   Keep only docs whose metadata ``doc_type`` matches.

    Returns:
        A :class:`BM25Retriever` limited to the filtered subset.

    Raises:
        ValueError: If the filtered subset is empty.
    """
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
    """Combine BM25 (lexical) and FAISS (semantic) into one retriever.

    Uses LangChain's :class:`EnsembleRetriever` with Reciprocal Rank
    Fusion (RRF) to merge ranked lists from both retrievers.

    Args:
        vs:            A loaded or freshly built FAISS store.
        docs:          Full corpus used to initialise the BM25 retriever.
        k:             Final number of merged results to return.
        dense_weight:  Score weight given to FAISS results (0–1).
        sparse_weight: Score weight given to BM25 results (0–1).
                       ``dense_weight + sparse_weight`` need not sum to 1;
                       they are used as relative importance.
        law_id:        Metadata filter applied to *both* sub-retrievers.
        chunk_type:    Metadata filter applied to *both* sub-retrievers.
        doc_type:      Metadata filter applied to *both* sub-retrievers.

    Returns:
        An :class:`EnsembleRetriever` that queries both backends and fuses
        their results.

    Example::

        retriever = get_hybrid_retriever(
            vs=vector_store,
            docs=all_docs,
            k=8,
            dense_weight=0.7,
            sparse_weight=0.3,
            doc_type=DocType.LAW,
        )
        results = retriever.invoke("ما هي عقوبة الاحتيال؟")
    """
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