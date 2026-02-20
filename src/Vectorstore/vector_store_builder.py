from __future__ import annotations

import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import Field

from src.Chunking.chunking import CorpusChunker
from src.Config.config import DataPath, EMBEDDING_MODEL

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

VECTORSTORE_ROOT = Path(DataPath).parent / "VectorStores"


# =============================================================================
# RETRIEVER
# =============================================================================

class Na2dRetriever(BaseRetriever):
    """Unified retriever over rulings and/or principles stores."""

    stores:             Dict[str, FAISS] = Field(...)
    k:                  int              = Field(default=5)
    search_rulings:     bool             = Field(default=True)
    search_principles:  bool             = Field(default=True)
    crime_category:     Optional[str]    = Field(default=None)

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        collected: List[Document] = []

        if self.search_rulings and "rulings" in self.stores:
            results = self.stores["rulings"].similarity_search(query, k=self.k * 3)
            if self.crime_category:
                results = [d for d in results if d.metadata.get("crime_category") == self.crime_category]
            collected.extend(results)

        if self.search_principles and "principles" in self.stores:
            collected.extend(self.stores["principles"].similarity_search(query, k=self.k * 3))

        seen, unique = set(), []
        for doc in collected:
            cid = doc.metadata.get("chunk_id", id(doc))
            if cid not in seen:
                seen.add(cid)
                unique.append(doc)

        return unique[:self.k]


# =============================================================================
# VECTOR STORE MANAGER
# =============================================================================

class LegalVectorStore:
    """
    Unified vector store for both law corpus and na2d corpus.

    Usage::

        lv = LegalVectorStore()

        # Laws
        lv.build_laws()
        results = lv.search_laws("عقوبة السرقة", k=5)
        retriever = lv.laws_retriever()

        # Na2d
        lv.build_na2d()
        results = lv.search_na2d("مبدأ شخصية العقوبة", k=5)
        retriever = lv.na2d_retriever()
    """

    def __init__(self, embedding_model: str = EMBEDDING_MODEL, force_rebuild: bool = False):
        self.force_rebuild  = force_rebuild
        self._embeddings    = HuggingFaceEmbeddings(
            model_name    = embedding_model,
            model_kwargs  = {"device": "cpu"},
            encode_kwargs = {"normalize_embeddings": True, "batch_size": 64},
        )
        self._laws_store:       Optional[FAISS]             = None
        self._na2d_stores:      Dict[str, FAISS]            = {}
        self._laws_docs:        List[Document]               = []
        self._laws_path         = VECTORSTORE_ROOT / "laws"
        self._rulings_path      = VECTORSTORE_ROOT / "na2d" / "rulings"
        self._principles_path   = VECTORSTORE_ROOT / "na2d" / "principles"

    # ------------------------------------------------------------------ build

    def build_laws(self, docs_filter: str = "latest") -> "LegalVectorStore":
        """
        Build FAISS index for law corpus.
        docs_filter: 'all' | 'latest' | 'original'
        """
        if self._laws_path.exists() and not self.force_rebuild:
            logger.info("Loading laws index from disk …")
            self._laws_store = self._load(self._laws_path)
            return self

        docs = CorpusChunker().get_chunks()
        docs = self._filter(docs, docs_filter)
        logger.info(f"Indexing {len(docs):,} law documents …")
        self._laws_docs  = docs
        self._laws_store = self._build(docs, "laws")
        self._save(self._laws_store, self._laws_path)
        return self

    def build_na2d(self) -> "LegalVectorStore":
        """Build FAISS indexes for rulings and principles."""
        if self._rulings_path.exists() and self._principles_path.exists() and not self.force_rebuild:
            logger.info("Loading na2d indexes from disk …")
            self._na2d_stores["rulings"]    = self._load(self._rulings_path)
            self._na2d_stores["principles"] = self._load(self._principles_path)
            return self

        corpus = CorpusChunker().get_na2d_chunks()
        for key, path in [("rulings", self._rulings_path), ("principles", self._principles_path)]:
            self._na2d_stores[key] = self._build(corpus[key], key)
            self._save(self._na2d_stores[key], path)
        return self

    # ------------------------------------------------------------------ search

    def search_laws(self, query: str, k: int = 5,
                    law_id: Optional[str] = None,
                    chunk_type: Optional[str] = None) -> List[Document]:
        self._require("laws")
        filters = {k: v for k, v in [("law_id", law_id), ("chunk_type", chunk_type)] if v}
        return self._laws_store.similarity_search(query, k=k, filter=filters or None)

    def search_na2d(self, query: str, k: int = 5,
                    source: str = "both",
                    crime_category: Optional[str] = None) -> List[Document]:
        """source: 'rulings' | 'principles' | 'both'"""
        self._require("na2d")
        results: List[Document] = []
        for key in (["rulings", "principles"] if source == "both" else [source]):
            hits = self._na2d_stores[key].similarity_search(query, k=k)
            if crime_category and key == "rulings":
                hits = [d for d in hits if d.metadata.get("crime_category") == crime_category]
            results.extend(hits)
        return results[:k]

    # ------------------------------------------------------------------ retrievers

    def laws_retriever(self, k: int = 5, law_id: Optional[str] = None, chunk_type: Optional[str] = None):
        self._require("laws")
        filters = {k: v for k, v in [("law_id", law_id), ("chunk_type", chunk_type)] if v}
        return self._laws_store.as_retriever(search_kwargs={"k": k, **({"filter": filters} if filters else {})})

    def na2d_retriever(self, k: int = 5, search_rulings: bool = True,
                       search_principles: bool = True, crime_category: Optional[str] = None) -> Na2dRetriever:
        self._require("na2d")
        return Na2dRetriever(stores=self._na2d_stores, k=k,
                             search_rulings=search_rulings,
                             search_principles=search_principles,
                             crime_category=crime_category)

    # ------------------------------------------------------------------ filter helpers

    def _filter(self, docs: List[Document], mode: str) -> List[Document]:
        if mode == "original":
            return [d for d in docs if d.metadata.get("chunk_type") == "article"]
        if mode == "latest":
            return self._latest_versions(docs)
        return docs  # 'all'

    @staticmethod
    def _latest_versions(docs: List[Document]) -> List[Document]:
        groups: Dict[tuple, List[Document]] = defaultdict(list)
        for doc in docs:
            m = doc.metadata
            groups[(m.get("law_id"), m.get("article_number"))].append(doc)
        latest = []
        for group in groups.values():
            amended = sorted(
                [d for d in group if d.metadata.get("chunk_type") == "amended_article" and d.metadata.get("amendment_date")],
                key=lambda d: d.metadata["amendment_date"], reverse=True,
            )
            if amended:
                latest.append(amended[0])
            else:
                originals = [d for d in group if d.metadata.get("sub_index", 0) == 0]
                latest.extend(originals or group[:1])
        return latest

    # ------------------------------------------------------------------ FAISS helpers

    def _build(self, docs: List[Document], label: str) -> FAISS:
        logger.info(f"  [{label}] Embedding {len(docs):,} chunks …")
        t0    = time.time()
        store = FAISS.from_documents(docs, self._embeddings)
        logger.info(f"  [{label}] Done in {time.time() - t0:.1f}s")
        return store

    def _save(self, store: FAISS, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        store.save_local(str(path))
        logger.info(f"  Saved → {path}")

    def _load(self, path: Path) -> FAISS:
        return FAISS.load_local(str(path), self._embeddings, allow_dangerous_deserialization=True)

    def _require(self, corpus: str):
        if corpus == "laws" and self._laws_store is None:
            raise RuntimeError("Laws index not built. Call build_laws() first.")
        if corpus == "na2d" and not self._na2d_stores:
            raise RuntimeError("Na2d indexes not built. Call build_na2d() first.")


# =============================================================================
# ENTRY POINTS
# =============================================================================

def get_retriever():
    """Laws retriever — latest versions, FAISS backed."""
    return LegalVectorStore().build_laws(docs_filter="latest").laws_retriever()

def get_na2d_retriever(**kwargs) -> Na2dRetriever:
    """Na2d retriever — rulings + principles, FAISS backed."""
    return LegalVectorStore().build_na2d().na2d_retriever(**kwargs)