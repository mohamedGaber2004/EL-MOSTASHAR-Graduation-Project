from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from src.Chunking.chunking import get_chunks, get_na2d_chunks
from src.Config import get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# FILTER
# =============================================================================

def filter_latest_versions(docs: List[Document]) -> List[Document]:
    """Keep only the most recent version of each (law_id, article_number) pair."""
    no_number = [d for d in docs if not d.metadata.get("article_number")]

    groups: Dict[tuple, List[Document]] = defaultdict(list)
    for doc in docs:
        if doc.metadata.get("article_number"):
            key = (doc.metadata.get("law_id"), doc.metadata["article_number"])
            groups[key].append(doc)

    latest = []
    for group in groups.values():
        amended = sorted(
            [d for d in group if d.metadata.get("chunk_type") == "amended_article" and d.metadata.get("amendment_date")],
            key=lambda d: d.metadata["amendment_date"],
            reverse=True,
        )
        if amended:
            latest.append(amended[0])
        else:
            originals = [d for d in group if d.metadata.get("chunk_type") == "article"]
            latest.append((originals or group)[0])

    return no_number + latest


# =============================================================================
# BUILD & LOAD
# =============================================================================

def build_vector_store(docs: List[Document], embeddings, index_path: str = "legal_faiss_index") -> FAISS:
    if not docs:
        raise ValueError("No documents to index.")
    logger.info("Indexing %d documents ...", len(docs))
    vs = FAISS.from_documents(docs, embeddings)
    vs.save_local(index_path)
    logger.info("✓ FAISS index saved → %s", index_path)
    return vs


def load_vector_store(index_path: str, embeddings) -> FAISS:
    vs = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    logger.info("✓ FAISS index loaded ← %s", index_path)
    return vs


# =============================================================================
# SEARCH
# =============================================================================

def _build_filter(**kwargs) -> Dict[str, Any]:
    return {k: v for k, v in kwargs.items() if v is not None}


def search(vs: FAISS, query: str, k: int = 5, law_id: str = None, chunk_type: str = None, doc_type: str = None) -> List[Document]:
    f = _build_filter(law_id=law_id, chunk_type=chunk_type, doc_type=doc_type)
    return vs.similarity_search(query, k=k, filter=f or None)


def search_with_scores(vs: FAISS, query: str, k: int = 5, law_id: str = None, doc_type: str = None) -> List[Tuple[Document, float]]:
    f = _build_filter(law_id=law_id, doc_type=doc_type)
    return vs.similarity_search_with_score(query, k=k, filter=f or None)


def search_latest(vs: FAISS, query: str, k: int = 5, law_id: str = None) -> List[Document]:
    """Search then deduplicate — keeps only the latest version of each article."""
    raw = search(vs, query, k=k * 3, law_id=law_id)
    return filter_latest_versions(raw)[:k]


# =============================================================================
# Retriever helpers
# =============================================================================

def get_retriever(vs: FAISS, k: int = 5, law_id: str = None, chunk_type: str = None, doc_type: str = None):
    """Return a retriever bound to the given FAISS vectorstore.

    NOTE: This function requires an explicit `vs` instance to avoid module-level
    side-effects during import. Build or load the FAISS instance at application
    runtime and pass it here.
    """
    f = _build_filter(law_id=law_id, chunk_type=chunk_type, doc_type=doc_type)
    search_kwargs = {"k": k, **({"filter": f} if f else {})}
    return vs.as_retriever(search_kwargs=search_kwargs)

