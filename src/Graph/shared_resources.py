import logging
from functools import lru_cache
from typing import Optional

from src.Config import get_settings
from src.Graphstore.KG_builder import LegalKnowledgeGraph

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_kg() -> Optional[LegalKnowledgeGraph]:
    """
    يفتح Neo4j connection مرة واحدة.
    لو فشل يرجع None والـ agents تشتغل بدونه.
    """
    cfg = get_settings()
    if not getattr(cfg, "NEO4J_URI", None):
        logger.warning("⚠️ NEO4J_URI مش موجود — KG معطل")
        return None
    try:
        kg = LegalKnowledgeGraph(cfg.NEO4J_URI, cfg.NEO4J_USERNAME, cfg.NEO4J_PASSWORD)
        kg.connect()
        logger.info("✅ Neo4j connected")
        return kg
    except Exception as e:
        logger.warning("⚠️ KG connection فشل: %s — هيشتغل بدون KG", e)
        return None


@lru_cache(maxsize=1)
def get_vector_store():
    cfg = get_settings()
    faiss_path  = getattr(cfg, "FAISS_INDEX_PATH", None)
    embed_model = getattr(cfg, "EMBEDDING_MODEL", None)

    if not faiss_path or not embed_model:
        logger.warning("⚠️ FAISS_INDEX_PATH أو EMBEDDING_MODEL مش موجود — VectorStore معطل")
        return None
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        from src.Vectorstore.vector_store_builder import build_vector_store, load_vector_store, filter_latest_versions
        from pathlib import Path

        embeddings = HuggingFaceEmbeddings(model_name=embed_model)
        index_faiss = Path(faiss_path) / "index.faiss"

        # ── لو الـ index موجود — حمّله ─────────────────────────────────
        if index_faiss.exists():
            vs = load_vector_store(str(faiss_path), embeddings)
            logger.info("✅ VectorStore loaded")
            return vs

        # ── لو مش موجود — ابنيه ────────────────────────────────────────
        logger.warning("⚠️ FAISS index مش موجود — بيتبنى دلوقتي...")
        from src.Chunking.chunking import get_chunks

        chunks = get_chunks()
        logger.info("📄 %d chunks loaded", len(chunks))

        chunks = filter_latest_versions(chunks)
        logger.info("📄 %d chunks after dedup", len(chunks))

        if not chunks:
            logger.error("❌ مفيش chunks — مش هيتبنى الـ index")
            return None

        vs = build_vector_store(
            docs=chunks,
            embeddings=embeddings,
            index_path=str(faiss_path),
        )
        logger.info("✅ VectorStore built and saved → %s", faiss_path)
        return vs

    except Exception as e:
        logger.error("❌ VectorStore فشل: %s", e, exc_info=True)
        return None


def cleanup():
    """أغلق الـ KG connection عند إنهاء البرنامج."""
    kg = get_kg()
    if kg:
        try:
            kg.close()
            logger.info("✅ Neo4j connection أُغلق")
        except Exception:
            pass