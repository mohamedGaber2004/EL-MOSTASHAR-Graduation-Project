from src.Config.log_config import logging
from functools import lru_cache
from typing import Optional

from src.Config import get_settings
from src.Graphstore.KG_builder import LegalKnowledgeGraph

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_kg() -> Optional[LegalKnowledgeGraph]:
    """
    Opens Neo4j connection once.
    If it fails, returns None and agents work without it.
    """
    cfg = get_settings()
    if not getattr(cfg, "NEO4J_URI", None):
        logger.warning("NEO4J_URI not found — Knowledge Graph is disabled.")
        return None
    try:
        kg = LegalKnowledgeGraph(cfg.NEO4J_URI, cfg.NEO4J_USERNAME, cfg.NEO4J_PASSWORD)
        kg.connect()
        logger.info("Neo4j connected successfully.")
        return kg
    except Exception as e:
        logger.warning("Knowledge Graph connection failed: %s — running without KG.", e)
        return None


@lru_cache(maxsize=1)
def get_vector_store():
    cfg = get_settings()
    faiss_path  = getattr(cfg, "FAISS_INDEX_PATH", None)
    embed_model = getattr(cfg, "ARABIC_NATIVE_EMBEDDING_MODEL", None)

    if not faiss_path or not embed_model:
        logger.warning("FAISS_INDEX_PATH or EMBEDDING_MODEL not found — VectorStore is disabled.")
        return None
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        from src.Vectorstore.vector_store_builder import build_vector_store, load_vector_store
        from pathlib import Path

        embeddings = HuggingFaceEmbeddings(model_name=embed_model)
        index_faiss = Path(faiss_path) / "index.faiss"

        # If index exists, load it
        if index_faiss.exists():
            vs = load_vector_store(str(faiss_path), embeddings)
            logger.info("VectorStore loaded — %d vectors", vs.index.ntotal)
            return vs

        # If it does not exist, build it
        logger.warning("FAISS index not found - building now...")
        from src.Chunking.chunking import get_na2d_chunks

        na2d = get_na2d_chunks()  # returns {"rulings": [...], "principles": [...]}

        chunks = na2d.get("rulings", []) + na2d.get("principles", [])

        logger.info(
            "📄 %d chunks loaded (rulings=%d, principles=%d)",
            len(chunks),
            len(na2d.get("rulings", [])),
            len(na2d.get("principles", [])),
        )

        if not chunks:
            logger.error("No chunks found - index will not be built")
            return None

        vs = build_vector_store(
            docs=chunks,
            embeddings=embeddings,
            index_path=str(faiss_path),
        )
        logger.info("VectorStore built and saved -> %s", faiss_path)
        return vs

    except Exception as e:
        logger.error("VectorStore failed: %s", e, exc_info=True)
        return None


def cleanup():
    """Close the KG connection on program exit."""
    kg = get_kg()
    if kg:
        try:
            kg.close()
            logger.info("Neo4j connection closed")
        except Exception:
            pass