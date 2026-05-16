from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Dict, List, Tuple

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


logger = logging.getLogger(__name__)


# =============================================================================
# BUILD & LOAD
# =============================================================================

def build_vector_store(docs: List[Document], embeddings, index_path: str = "legal_faiss_index") -> FAISS:
    if not docs:
        raise ValueError("No documents to index.")
    logger.info("Indexing %d documents ...", len(docs))
    vs = FAISS.from_documents(docs, embeddings)
    vs.save_local(index_path)
    logger.info("FAISS index saved -> %s", index_path)
    return vs

def load_vector_store(index_path: str, embeddings) -> FAISS:
    vs = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    logger.info("FAISS index loaded <- %s", index_path)
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

# =============================================================================
# Retriever helpers
# =============================================================================

def get_retriever(vs: FAISS, k: int = 5, law_id: str = None, chunk_type: str = None, doc_type: str = None):
    f = _build_filter(law_id=law_id, chunk_type=chunk_type, doc_type=doc_type)
    search_kwargs = {"k": k, **({"filter": f} if f else {})}
    return vs.as_retriever(search_kwargs=search_kwargs)