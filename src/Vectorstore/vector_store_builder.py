from src.Chunking.chunking import get_chunks
from src.Config.config import EMBEDDING_MODEL

import logging
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# =============================================================================
# SECTION 1: FILTER HELPERS
# =============================================================================

def filter_original_only(docs: List[Document]) -> List[Document]:
    """Keep only original law articles (exclude preambles and amendments)."""
    return [d for d in docs if d.metadata.get("chunk_type") == "article"]


def filter_amended_only(docs: List[Document]) -> List[Document]:
    """Keep only amended articles (from new* files)."""
    return [d for d in docs if d.metadata.get("chunk_type") == "amended_article"]


def filter_latest_versions(docs: List[Document]) -> List[Document]:
    """
    For each (law_id, article_number) pair keep only the most recent version.
    Priority order:
        1. amended_article with the latest amendment_date
        2. original article  (if no amendment exists)
    This gives a clean, deduplicated corpus for RAG.
    """
    from collections import defaultdict

    # group by (law_id, article_number)
    groups: Dict[tuple, List[Document]] = defaultdict(list)
    for doc in docs:
        m   = doc.metadata
        key = (m.get("law_id"), m.get("article_number"))
        groups[key].append(doc)

    latest: List[Document] = []
    for key, group in groups.items():
        # amended articles with a date
        amended = [
            d for d in group
            if d.metadata.get("chunk_type") == "amended_article"
            and d.metadata.get("amendment_date")
        ]
        if amended:
            # pick the most recent amendment date
            amended.sort(key=lambda d: d.metadata["amendment_date"], reverse=True)
            latest.append(amended[0])
        else:
            # fall back to original (keep sub_index == 0 to avoid duplicates)
            originals = [d for d in group if d.metadata.get("sub_index", 0) == 0]
            latest.extend(originals if originals else group[:1])

    return latest


# =============================================================================
# SECTION 2: VECTOR STORE BUILDER
# =============================================================================

class LegalVectorStore:
    """
    Wraps any LangChain vector store with legal-specific helpers.

    Usage:
        from src.Chunking.chunking import build_rag_documents
        from src.VS.vector_store_builder import LegalVectorStore

        docs  = build_rag_documents(ROOT_DIR)
        lvs   = LegalVectorStore.from_documents(docs, embeddings, backend="faiss")

        # simple search
        results = lvs.search("عقوبة السرقة", k=5)

        # search within one law
        results = lvs.search("الغرامة المالية", k=5, law_id="penal_code")

        # search latest versions only
        results = lvs.search_latest("نفقة الزوجة", k=5)
    """

    def __init__(self, vectorstore, docs: List[Document]):
        self.vectorstore = vectorstore
        self._docs       = docs

    # ── constructors ──────────────────────────────────────────────────────

    @classmethod
    def from_documents(
        cls,
        docs:      List[Document],
        embeddings,
        backend:   str  = "faiss",      # 'faiss' | 'chroma'
        chroma_persist_dir: Optional[str] = "./chroma_legal_db",
        chroma_collection:  str = "egyptian_legal_corpus",
        faiss_index_path:   Optional[str] = None,
        docs_filter:        Optional[str] = "all",   # 'all'|'latest'|'original'
        verbose:            bool = True,
    ) -> "LegalVectorStore":
        """
        Build the vector store from documents.

        Args:
            docs:       Output of build_rag_documents() from chunking.py.
            embeddings: Any LangChain Embeddings object.
            backend:    'faiss' or 'chroma'.
            docs_filter:
                'all'      — index everything (original + amended)
                'latest'   — deduplicated: amended version wins over original
                'original' — only original law articles (no amendments)
            faiss_index_path: if set, save FAISS index to this path.
        """
        # ── apply filter ──────────────────────────────────────────────────
        if docs_filter == "latest":
            index_docs = filter_latest_versions(docs)
            logger.info(f"  Filter='latest' → {len(index_docs):,} docs "
                        f"(from {len(docs):,})")
        elif docs_filter == "original":
            index_docs = filter_original_only(docs)
            logger.info(f"  Filter='original' → {len(index_docs):,} docs")
        else:
            index_docs = docs
            logger.info(f"  Filter='all' → {len(index_docs):,} docs")

        if not index_docs:
            raise ValueError("No documents to index after filtering.")

        # ── build vector store ────────────────────────────────────────────
        if backend == "faiss":
            vectorstore = cls._build_faiss(index_docs, embeddings,
                                           faiss_index_path, verbose)
        elif backend == "chroma":
            vectorstore = cls._build_chroma(index_docs, embeddings,
                                            chroma_persist_dir,
                                            chroma_collection, verbose)
        else:
            raise ValueError(f"Unknown backend: '{backend}'. Use 'faiss' or 'chroma'.")

        return cls(vectorstore, index_docs)

    @classmethod
    def _build_faiss(cls, docs, embeddings, index_path, verbose):
        try:
            from langchain_community.vectorstores import FAISS
        except ImportError:
            raise ImportError(
                "langchain-community not installed. "
                "Run: pip install langchain-community faiss-cpu"
            )
        if verbose:
            print(f"  Building FAISS index over {len(docs):,} documents …")
        vs = FAISS.from_documents(docs, embeddings)
        if index_path:
            vs.save_local(index_path)
            logger.info(f"  ✓ FAISS index saved → {index_path}")
        if verbose:
            print(f"  ✓ FAISS index ready")
        return vs

    @classmethod
    def _build_chroma(cls, docs, embeddings, persist_dir, collection, verbose):
        try:
            from langchain_community.vectorstores import Chroma
        except ImportError:
            raise ImportError(
                "langchain-community not installed. "
                "Run: pip install langchain-community chromadb"
            )
        if verbose:
            print(f"  Building Chroma index over {len(docs):,} documents …")
        vs = Chroma.from_documents(
            docs, embeddings,
            collection_name   = collection,
            persist_directory = persist_dir,
        )
        if verbose:
            print(f"  ✓ Chroma index ready  (persist_dir={persist_dir})")
        return vs

    # ── load from disk ────────────────────────────────────────────────────

    @classmethod
    def load_faiss(cls, index_path: str, embeddings,
                   docs: Optional[List[Document]] = None) -> "LegalVectorStore":
        """Load a previously saved FAISS index from disk."""
        from langchain_community.vectorstores import FAISS
        vs = FAISS.load_local(index_path, embeddings,
                              allow_dangerous_deserialization=True)
        logger.info(f"✓ FAISS index loaded ← {index_path}")
        return cls(vs, docs or [])

    # ── search ────────────────────────────────────────────────────────────

    def search(
        self,
        query:   str,
        k:       int = 5,
        law_id:  Optional[str] = None,
        chunk_type: Optional[str] = None,
    ) -> List[Document]:
        """
        Similarity search with optional metadata filters.

        Args:
            query:      Arabic query string.
            k:          Number of results.
            law_id:     Restrict to one law  (e.g. 'penal_code').
            chunk_type: Restrict by type     ('article'|'amended_article'|'preamble').
        """
        filter_dict: Dict[str, Any] = {}
        if law_id:
            filter_dict["law_id"] = law_id
        if chunk_type:
            filter_dict["chunk_type"] = chunk_type

        if filter_dict:
            return self.vectorstore.similarity_search(
                query, k=k, filter=filter_dict)
        return self.vectorstore.similarity_search(query, k=k)

    def search_with_scores(
        self,
        query:  str,
        k:      int = 5,
        law_id: Optional[str] = None,
    ) -> List[tuple[Document, float]]:
        """Returns (Document, score) pairs sorted by similarity."""
        filter_dict = {"law_id": law_id} if law_id else {}
        if filter_dict:
            return self.vectorstore.similarity_search_with_score(
                query, k=k, filter=filter_dict)
        return self.vectorstore.similarity_search_with_score(query, k=k)

    def search_latest(self, query: str, k: int = 5,
                      law_id: Optional[str] = None) -> List[Document]:
        """
        Search, then for each (law_id, article_number) keep only the
        most recent version among the returned hits.
        Useful when the index contains both original and amended chunks.
        """
        # over-fetch so dedup doesn't shrink below k
        raw     = self.search(query, k=k * 3, law_id=law_id)
        deduped = filter_latest_versions(raw)
        return deduped[:k]

    def as_retriever(self, k: int = 5,
                     law_id: Optional[str] = None,
                     chunk_type: Optional[str] = None):
        """Return a LangChain BaseRetriever for use in chains."""
        search_kwargs: Dict[str, Any] = {"k": k}
        filter_dict: Dict[str, Any] = {}
        if law_id:
            filter_dict["law_id"] = law_id
        if chunk_type:
            filter_dict["chunk_type"] = chunk_type
        if filter_dict:
            search_kwargs["filter"] = filter_dict
        return self.vectorstore.as_retriever(search_kwargs=search_kwargs)

    # ── info ──────────────────────────────────────────────────────────────

    def print_summary(self):
        import pandas as pd
        if not self._docs:
            print("No document metadata available.")
            return
        df = pd.DataFrame([d.metadata for d in self._docs])
        print("\n" + "=" * 70)
        print("VECTOR STORE SUMMARY")
        print("=" * 70)
        print(f"  Indexed documents : {len(df):,}")
        print(f"  Laws covered      : {df['law_id'].nunique()}")
        print(f"  Amended docs      : {int(df['is_amended'].sum()):,}")
        print("\n  Breakdown by law and type:")
        breakdown = (
            df.groupby(["law_id", "chunk_type"])["chunk_id"]
              .count().rename("count").reset_index().to_string(index=False)
        )
        for line in breakdown.split("\n"):
            print(f"    {line}")
        print("=" * 70 + "\n")


# =============================================================================
# SECTION 3: CONVENIENCE FUNCTION
# =============================================================================

def build_vector_store(
    docs:           List[Document],
    embeddings,
    backend:        str  = "faiss",
    docs_filter:    str  = "latest",
    faiss_index_path:   Optional[str] = None,
    chroma_persist_dir: Optional[str] = "./chroma_legal_db",
    chroma_collection:  str = "egyptian_legal_corpus",
    verbose:        bool = True,
) -> LegalVectorStore:

    lvs = LegalVectorStore.from_documents(
        docs                = docs,
        embeddings          = embeddings,
        backend             = backend,
        docs_filter         = docs_filter,
        faiss_index_path    = faiss_index_path,
        chroma_persist_dir  = chroma_persist_dir,
        chroma_collection   = chroma_collection,
        verbose             = verbose,
    )
    lvs.print_summary()
    return lvs


def get_retriever():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    lvs = build_vector_store(
        get_chunks(), embeddings,
        backend          = "faiss",
        docs_filter      = "latest",        # amended version wins over original
        faiss_index_path = "legal_faiss_index",
    )
    return lvs.as_retriever()