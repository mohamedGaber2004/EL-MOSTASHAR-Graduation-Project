from __future__ import annotations

# =============================================================================
# IMPORTS
# =============================================================================

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from src.Chunking.chunking import get_chunks, get_na2d_chunks
from src.Config import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# FILTER HELPERS
# =============================================================================

def filter_latest_versions(docs: List[Document]) -> List[Document]:
    """
    For each (law_id, article_number) pair keep only the most recent version.

    Priority:
        1. amended_article with the latest amendment_date
        2. original article (if no amendment exists)

    Chunks with no article_number (preambles, tables, rulings, principles)
    are always passed through unchanged.

    Use this at search-time to deduplicate results, NOT at index-build time.
    """
    no_number: List[Document] = [
        d for d in docs if not d.metadata.get("article_number")
    ]

    groups: Dict[tuple, List[Document]] = defaultdict(list)
    for doc in docs:
        article_no = doc.metadata.get("article_number")
        if not article_no:
            continue
        key = (doc.metadata.get("law_id"), article_no)
        groups[key].append(doc)

    latest: List[Document] = []
    for _, group in groups.items():
        amended = [
            d for d in group
            if d.metadata.get("chunk_type") == "amended_article"
            and d.metadata.get("amendment_date")
        ]
        if amended:
            amended.sort(key=lambda d: d.metadata["amendment_date"], reverse=True)
            latest.append(amended[0])
        else:
            originals = [d for d in group if d.metadata.get("chunk_type") == "article"]
            latest.extend(originals[:1] if originals else group[:1])

    return no_number + latest


# =============================================================================
# VECTOR STORE
# =============================================================================

class LegalVectorStore:
    """
    Wraps any LangChain vector store with legal-specific helpers.

    Supported backends: ``'faiss'``, ``'chroma'``

    Metadata keys — law chunks
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ``law_id``               e.g. ``'penal_code'``
    ``law_title``            human-readable Arabic title
    ``chunk_type``           ``'article'`` | ``'amended_article'`` |
                             ``'preamble'`` | ``'table'``
    ``article_number``       Western-digit string, e.g. ``'45'``
    ``source_file``          original filename
    ``amendment_date``       ISO date string  (amended_article only)
    ``amendment_law_number`` (amended_article only)
    ``amendment_type``       ``'modification'`` | ``'addition'`` | ``'deletion'``

    Metadata keys — Na2d chunks
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ``doc_type``             ``'ruling'`` | ``'principle'``
    ``crime_category``       folder name  (rulings only)
    ``book_title``           Arabic title (principles only)
    ``case_number``          (rulings only)
    ``case_year``            (rulings only)
    ``ruling_date``          (rulings only)
    ``chamber``              (rulings only)
    ``source_file``          original filename
    """

    def __init__(self, vectorstore, docs: List[Document]) -> None:
        self.vectorstore = vectorstore
        self._docs       = docs

    # ── constructors ──────────────────────────────────────────────────────

    @classmethod
    def from_documents(cls,docs:List[Document],embeddings,backend:str= "faiss",faiss_index_path:Optional[str] = None,chroma_persist_dir:Optional[str] = "./chroma_legal_db",chroma_collection:str= "egyptian_legal_corpus",verbose:bool= True,) -> "LegalVectorStore":
        if not docs:
            raise ValueError("No documents to index.")

        logger.info(f"  Indexing {len(docs):,} documents (backend='{backend}')")

        if backend == "faiss":
            vs = cls._build_faiss(docs, embeddings, faiss_index_path, verbose)
        elif backend == "chroma":
            vs = cls._build_chroma(
                docs, embeddings,
                chroma_persist_dir, chroma_collection, verbose,
            )
        else:
            raise ValueError(f"Unknown backend '{backend}'. Use 'faiss' or 'chroma'.")

        return cls(vs, docs)

    @classmethod
    def _build_faiss(cls, docs, embeddings, index_path, verbose):
        try:
            from langchain_community.vectorstores import FAISS
        except ImportError:
            raise ImportError("Run: pip install langchain-community faiss-cpu")
        vs = FAISS.from_documents(docs, embeddings)
        if index_path:
            vs.save_local(index_path)
            logger.info(f"  ✓ FAISS index saved → {index_path}")
        return vs

    @classmethod
    def _build_chroma(cls, docs, embeddings, persist_dir, collection, verbose):
        try:
            from langchain_community.vectorstores import Chroma
        except ImportError:
            raise ImportError("Run: pip install langchain-community chromadb")
        vs = Chroma.from_documents(
            docs, embeddings,
            collection_name   = collection,
            persist_directory = persist_dir,
        )
        return vs

    # ── load from disk ────────────────────────────────────────────────────

    @classmethod
    def load_faiss(
        cls,
        index_path: str,
        embeddings,
        docs: Optional[List[Document]] = None,
    ) -> "LegalVectorStore":
        """Load a previously saved FAISS index from disk."""
        from langchain_community.vectorstores import FAISS
        vs = FAISS.load_local(
            index_path, embeddings, allow_dangerous_deserialization=True
        )
        logger.info(f"✓ FAISS index loaded ← {index_path}")
        return cls(vs, docs or [])

    # ── search ────────────────────────────────────────────────────────────

    def search(
        self,
        query:      str,
        k:          int           = 5,
        law_id:     Optional[str] = None,
        chunk_type: Optional[str] = None,
        doc_type:   Optional[str] = None,
    ) -> List[Document]:
        """
        Similarity search with optional metadata filters.

        Args:
            query:      Arabic query string.
            k:          Number of results to return.
            law_id:     Restrict to one law         (e.g. ``'penal_code'``).
            chunk_type: Restrict by law chunk type   (``'article'`` |
                        ``'amended_article'`` | ``'preamble'`` | ``'table'``).
            doc_type:   Restrict to Na2d type        (``'ruling'`` | ``'principle'``).
        """
        filter_dict: Dict[str, Any] = {}
        if law_id:
            filter_dict["law_id"]     = law_id
        if chunk_type:
            filter_dict["chunk_type"] = chunk_type
        if doc_type:
            filter_dict["doc_type"]   = doc_type

        if filter_dict:
            return self.vectorstore.similarity_search(query, k=k, filter=filter_dict)
        return self.vectorstore.similarity_search(query, k=k)

    def search_with_scores(
        self,
        query:    str,
        k:        int           = 5,
        law_id:   Optional[str] = None,
        doc_type: Optional[str] = None,
    ) -> List[Tuple[Document, float]]:
        """Return ``(Document, score)`` pairs sorted by similarity."""
        filter_dict: Dict[str, Any] = {}
        if law_id:
            filter_dict["law_id"]   = law_id
        if doc_type:
            filter_dict["doc_type"] = doc_type
        if filter_dict:
            return self.vectorstore.similarity_search_with_score(
                query, k=k, filter=filter_dict
            )
        return self.vectorstore.similarity_search_with_score(query, k=k)

    def search_latest(
        self,
        query:  str,
        k:      int           = 5,
        law_id: Optional[str] = None,
    ) -> List[Document]:
        """
        Search then deduplicate results — for each (law_id, article_number)
        keep only the most recent version. Na2d chunks pass through unchanged.
        """
        raw     = self.search(query, k=k * 3, law_id=law_id)
        deduped = filter_latest_versions(raw)
        return deduped[:k]

    def as_retriever(
        self,
        k:          int           = 5,
        law_id:     Optional[str] = None,
        chunk_type: Optional[str] = None,
        doc_type:   Optional[str] = None,
    ):
        """Return a LangChain ``BaseRetriever`` for use in chains."""
        search_kwargs: Dict[str, Any] = {"k": k}
        filter_dict:   Dict[str, Any] = {}
        if law_id:
            filter_dict["law_id"]     = law_id
        if chunk_type:
            filter_dict["chunk_type"] = chunk_type
        if doc_type:
            filter_dict["doc_type"]   = doc_type
        if filter_dict:
            search_kwargs["filter"] = filter_dict
        return self.vectorstore.as_retriever(search_kwargs=search_kwargs)


# =============================================================================
# CONVENIENCE ENTRY POINT
# =============================================================================

def build_vector_store(
    docs:               List[Document],
    embeddings,
    backend:            str           = "faiss",
    faiss_index_path:   Optional[str] = None,
    chroma_persist_dir: Optional[str] = "./chroma_legal_db",
    chroma_collection:  str           = "egyptian_legal_corpus",
    verbose:            bool          = True,
) -> LegalVectorStore:

    return LegalVectorStore.from_documents(
        docs               = docs,
        embeddings         = embeddings,
        backend            = backend,
        faiss_index_path   = faiss_index_path,
        chroma_persist_dir = chroma_persist_dir,
        chroma_collection  = chroma_collection,
        verbose            = verbose,
    )


# =============================================================================
# RUN
# =============================================================================

law_chunks = get_chunks()
na2d       = get_na2d_chunks()
all_chunks = law_chunks + na2d["rulings"] + na2d["principles"]
embeddings = HuggingFaceEmbeddings(model_name=get_settings().EMBEDDING_MODEL)

lvs = build_vector_store(
    docs             = all_chunks,
    embeddings       = embeddings,
    backend          = "faiss",
    faiss_index_path = "legal_faiss_index",
)

def get_retriever():
    return lvs.as_retriever()