from __future__ import annotations

from src.Config.config import DataPath
from src.Chunking.na2d_chunking import build_na2d_documents

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from src.Config.config import EMBEDDING_MODEL
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# SECTION 1: CONFIGURATION
# =============================================================================

# Local save paths for persisted FAISS indexes
VECTORSTORE_ROOT = Path(DataPath).parent / "VectorStores" / "na2d"
RULINGS_PATH     = VECTORSTORE_ROOT / "rulings"
PRINCIPLES_PATH  = VECTORSTORE_ROOT / "principles"


# =============================================================================
# SECTION 2: EMBEDDING MODEL
# =============================================================================

def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Returns a CPU-bound HuggingFace embedding model.
    Model is cached locally after the first download (~420 MB).
    """
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    return HuggingFaceEmbeddings(
        model_name       = EMBEDDING_MODEL,
        model_kwargs     = {"device": "cpu"},
        encode_kwargs    = {"normalize_embeddings": True,   # cosine sim ready
                            "batch_size": 64},              # tune to your RAM
    )


# =============================================================================
# SECTION 3: BUILD HELPERS
# =============================================================================

def _build_faiss(
    docs:       List[Document],
    embeddings: HuggingFaceEmbeddings,
    label:      str,
) -> FAISS:
    """Embed docs and return an in-memory FAISS index."""
    logger.info(f"  [{label}] Embedding {len(docs):,} chunks …")
    t0    = time.time()
    store = FAISS.from_documents(docs, embeddings)
    elapsed = time.time() - t0
    logger.info(f"  [{label}] Done in {elapsed:.1f}s")
    return store


def _save(store: FAISS, path: Path, label: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    store.save_local(str(path))
    logger.info(f"  [{label}] Saved → {path}")


def _load(path: Path, embeddings: HuggingFaceEmbeddings, label: str) -> FAISS:
    logger.info(f"  [{label}] Loading from {path} …")
    return FAISS.load_local(
        str(path),
        embeddings,
        allow_dangerous_deserialization=True,   # safe: our own files
    )


# =============================================================================
# SECTION 4: MAIN BUILDER
# =============================================================================

def build_na2d_vectorstores(
    force_rebuild: bool = False,
) -> Dict[str, FAISS]:
    """
    Build (or load) FAISS CPU vector stores for the na2d corpus.

    - If indexes already exist on disk and force_rebuild=False, they are
      loaded directly (fast path — no re-embedding needed).
    - Otherwise the full chunking + embedding pipeline runs and indexes
      are persisted to VECTORSTORE_ROOT.

    Returns::

        {
            "rulings":    FAISS,   # 11-folder court rulings
            "principles": FAISS,   # 3 principle books
        }

    Example::

        stores = build_na2d_vectorstores()

        # semantic search
        results = stores["rulings"].similarity_search(
            "ما هي عقوبة السرقة بالإكراه؟", k=5
        )

        # with score
        results = stores["rulings"].similarity_search_with_score(
            "مبدأ شخصية العقوبة", k=3
        )
    """
    embeddings  = get_embeddings()
    stores: Dict[str, FAISS] = {}

    already_built = (
        RULINGS_PATH.exists()
        and PRINCIPLES_PATH.exists()
        and not force_rebuild
    )

    if already_built:
        logger.info("✓ Found existing indexes — loading from disk")
        stores["rulings"]    = _load(RULINGS_PATH,    embeddings, "rulings")
        stores["principles"] = _load(PRINCIPLES_PATH, embeddings, "principles")
        return stores

    # ── Full build ────────────────────────────────────────────────────────
    logger.info("Building na2d corpus …")
    corpus = build_na2d_documents()

    for key, path in [("rulings", RULINGS_PATH), ("principles", PRINCIPLES_PATH)]:
        docs         = corpus[key]
        stores[key]  = _build_faiss(docs, embeddings, key)
        _save(stores[key], path, key)

    logger.info(
        f"\n✅ na2d vector stores ready | "
        f"rulings: {len(corpus['rulings']):,} chunks | "
        f"principles: {len(corpus['principles']):,} chunks"
    )
    return stores


# =============================================================================
# SECTION 5: FILTERED SEARCH HELPERS
# =============================================================================

def search_rulings(
    store:          FAISS,
    query:          str,
    k:              int            = 5,
    crime_category: Optional[str] = None,
) -> List[Document]:
    """
    Semantic search over rulings with optional crime_category filter.

    Args:
        store:          The rulings FAISS store.
        query:          Arabic query string.
        k:              Number of results to return.
        crime_category: Arabic folder name to filter by (e.g. 'أحكام_السرقة').
                        If None, searches all categories.
    """
    # Fetch more candidates when filtering, then trim
    fetch_k  = k * 4 if crime_category else k
    results  = store.similarity_search(query, k=fetch_k)

    if crime_category:
        results = [
            d for d in results
            if d.metadata.get("crime_category") == crime_category
        ]

    return results[:k]


def search_principles(
    store: FAISS,
    query: str,
    k:     int = 5,
) -> List[Document]:
    """Semantic search over principle books."""
    return store.similarity_search(query, k=k)


# =============================================================================
# SECTION 6: RETRIEVER
# =============================================================================

from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks  import CallbackManagerForRetrieverRun
from langchain_core.documents  import Document
from pydantic import Field


class Na2dRetriever(BaseRetriever):
    """
    Unified retriever over the na2d corpus.

    Searches rulings and/or principles stores and merges results,
    deduplicating by chunk_id and ranking by combined relevance.

    Args:
        stores:          Dict returned by get_na2d_stores().
        k:               Total number of documents to return.
        search_rulings:  Whether to include court rulings in results.
        search_principles: Whether to include principle books in results.
        crime_category:  Optional Arabic folder name to filter rulings.

    Usage::

        retriever = get_na2d_retriever()
        docs = retriever.invoke("ما هي عقوبة السرقة بالإكراه؟")

        # rulings only, filtered by category
        retriever = get_na2d_retriever(
            search_principles=False,
            crime_category="أحكام_السرقة",
        )
    """

    stores:             Dict[str, FAISS]    = Field(...)
    k:                  int                 = Field(default=5)
    search_rulings_:    bool                = Field(default=True,  alias="search_rulings")
    search_principles_: bool                = Field(default=True,  alias="search_principles")
    crime_category:     Optional[str]       = Field(default=None)

    class Config:
        arbitrary_types_allowed = True
        populate_by_name        = True

    def _get_relevant_documents(
        self,
        query:            str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:

        # How many to fetch from each store before merging
        fetch_k   = self.k * 3
        collected: List[Document] = []

        if self.search_rulings_ and "rulings" in self.stores:
            results = self.stores["rulings"].similarity_search(query, k=fetch_k)
            if self.crime_category:
                results = [
                    d for d in results
                    if d.metadata.get("crime_category") == self.crime_category
                ]
            collected.extend(results)

        if self.search_principles_ and "principles" in self.stores:
            results = self.stores["principles"].similarity_search(query, k=fetch_k)
            collected.extend(results)

        # Deduplicate by chunk_id, preserving insertion order (= relevance order)
        seen: set         = set()
        unique: List[Document] = []
        for doc in collected:
            cid = doc.metadata.get("chunk_id", id(doc))
            if cid not in seen:
                seen.add(cid)
                unique.append(doc)

        return unique[: self.k]


# =============================================================================
# SECTION 7: ENTRY POINTS
# =============================================================================

def get_na2d_stores(force_rebuild: bool = False) -> Dict[str, FAISS]:
    """Returns ready-to-query FAISS stores (loads from disk if available)."""
    return build_na2d_vectorstores(force_rebuild=force_rebuild)


def get_na2d_retriever(
    k:                 int            = 5,
    search_rulings:    bool           = True,
    search_principles: bool           = True,
    crime_category:    Optional[str]  = None,
    force_rebuild:     bool           = False,
) -> Na2dRetriever:
    """
    Build and return a ready-to-use Na2dRetriever.

    Args:
        k:                 Number of documents to return per query.
        search_rulings:    Include court rulings in results.
        search_principles: Include principle books in results.
        crime_category:    Arabic folder name to restrict rulings search
                           (e.g. ``'أحكام_السرقة'``). None = all categories.
        force_rebuild:     Force re-embedding even if indexes exist on disk.

    Returns:
        Na2dRetriever — compatible with any LangChain chain or pipeline.

    Examples::

        # Default — searches everything
        retriever = get_na2d_retriever(k=5)
        docs = retriever.invoke("مبدأ شخصية العقوبة")

        # Rulings only, specific crime category
        retriever = get_na2d_retriever(
            search_principles=False,
            crime_category="أحكام_التزوير",
        )

        # Plug into a RAG chain
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=get_na2d_retriever(k=6),
        )
    """
    stores = get_na2d_stores(force_rebuild=force_rebuild)

    return Na2dRetriever(
        stores            = stores,
        k                 = k,
        search_rulings    = search_rulings,
        search_principles = search_principles,
        crime_category    = crime_category,
    )