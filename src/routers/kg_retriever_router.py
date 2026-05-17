from __future__ import annotations

import logging
import threading
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from langchain_huggingface import HuggingFaceEmbeddings

from src.Config import get_settings
from src.retriever.kg_retriever.kg_retriever import LegalRetriever, RetrievalResult

from src.routers.kg_router import get_graph


logger = logging.getLogger(__name__)

kg_retriever_router = APIRouter(prefix="/kg/retriever", tags=["KG Retriever"])


# =============================================================================
# LegalRetriever singleton
# =============================================================================
def get_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name=get_settings().ARABIC_NATIVE_EMBEDDING_MODEL)
    return embeddings

_retriever: Optional[LegalRetriever] = None
_RETRIEVER_LOCK = threading.Lock()

def get_retriever() -> LegalRetriever:
    """
    FastAPI dependency — lazily initialises the :class:`LegalRetriever`
    singleton (BM25 index is built on first call).

    Raises 503 if the underlying Neo4j graph is not yet connected.
    """
    global _retriever
    if _retriever is None:
        with _RETRIEVER_LOCK:
            if _retriever is None:
                graph      = get_graph()
                embeddings = get_embeddings()
                _retriever = LegalRetriever(graph=graph, embeddings=embeddings)
                logger.info("LegalRetriever singleton created")
    return _retriever


# =============================================================================
# Request / Response schemas
# =============================================================================

class RetrieveRequest(BaseModel):
    question:  str   = Field(...,  description="Natural-language legal question (Arabic)")
    k:         int   = Field(15,   description="ANN + BM25 candidate count",  ge=1, le=50)
    threshold: float = Field(0.5,  description="Minimum normalised RRF score", ge=0.0, le=1.0)


class SourceDoc(BaseModel):
    type:           str
    law_id:         str
    article_number: str
    law_title:      str
    score:          str
    text:           str


class RetrieveResponse(BaseModel):
    query:        str
    context_text: str
    sources:      list[SourceDoc]
    article_count: int
    table_count:   int


class IndexSetupResponse(BaseModel):
    message:    str
    dimensions: int


class EmbedResponse(BaseModel):
    message:       str
    nodes_written: int


class RetrieverStatusResponse(BaseModel):
    retriever_ready: bool
    bm25_article_count: Optional[int] = None


# =============================================================================
# Endpoints
# =============================================================================

# ── Status ────────────────────────────────────────────────────────────────────

@kg_retriever_router.get(
    "/status",
    response_model=RetrieverStatusResponse,
    summary="Check whether the LegalRetriever singleton is initialised",
)
def retriever_status() -> RetrieverStatusResponse:
    if _retriever is None:
        return RetrieverStatusResponse(retriever_ready=False)
    bm25_count = len(_retriever._bm25_index._records)
    return RetrieverStatusResponse(retriever_ready=True, bm25_article_count=bm25_count)


# ── Retrieve ─────────────────────────────────────────────────────────────────

@kg_retriever_router.post(
    "/retrieve",
    response_model=RetrieveResponse,
    summary="Hybrid legal retrieval: vector ANN + BM25 RRF + graph expansion",
)
def kg_retrieve(
    req:       RetrieveRequest,
    retriever: LegalRetriever = Depends(get_retriever),
) -> RetrieveResponse:
    """
    Full hybrid retrieval pipeline:

    1. Embed the question.
    2. ANN search over Article and TableChunk vector indexes in Neo4j.
    3. BM25 search over the in-memory article index.
    4. RRF merge of ANN and BM25 article hits.
    5. Graph expansion — enrich each article with penalties, amendments,
       definitions, cross-references, and topics.
    6. Threshold filtering and budget-aware context assembly.

    Returns the assembled ``context_text`` and a flat ``sources`` list for
    downstream RAG pipelines.
    """
    try:
        result: RetrievalResult = retriever.retrieve(
            question  = req.question,
            k         = req.k,
            threshold = req.threshold,
        )

        if not result.article_contexts and not result.table_contexts:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No relevant articles or tables found above the score threshold.",
            )

        return RetrieveResponse(
            query         = result.query,
            context_text  = result.context_text,
            sources       = [SourceDoc(**s) for s in result.sources],
            article_count = len(result.article_contexts),
            table_count   = len(result.table_contexts),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("kg_retrieve failed (question=%s)", req.question)
        raise HTTPException(status_code=500, detail=str(exc))


# ── Index management ──────────────────────────────────────────────────────────

@kg_retriever_router.post(
    "/index/setup",
    response_model=IndexSetupResponse,
    summary="Create Neo4j vector indexes for Article and TableChunk nodes",
)
def setup_index(
    dimensions: Optional[int]  = Query(None, description="Override embedding dimensions (auto-inferred if omitted)"),
    retriever:  LegalRetriever = Depends(get_retriever),
) -> IndexSetupResponse:
    """
    Idempotent — safe to call even when indexes already exist.
    Validates dimension consistency if the index already has entries.
    """
    try:
        dim = dimensions or retriever._embed_pipe.infer_dimension()
        retriever.setup_vector_index(dimensions=dim)
        return IndexSetupResponse(
            message    = "Vector indexes created (or already existed).",
            dimensions = dim,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
    except Exception as exc:
        logger.exception("setup_index failed")
        raise HTTPException(status_code=500, detail=str(exc))


@kg_retriever_router.post(
    "/index/embed",
    response_model=EmbedResponse,
    summary="Embed all un-embedded Article and TableChunk nodes",
)
def embed_nodes(
    batch_size:  int            = Query(128, description="Embedding batch size", ge=1),
    max_workers: int            = Query(4,   description="Parallel encoding threads", ge=1),
    retriever:   LegalRetriever = Depends(get_retriever),
) -> EmbedResponse:
    """
    Only processes nodes whose ``embedding`` property is currently ``NULL``.
    Safe to call repeatedly — already-embedded nodes are skipped.
    """
    try:
        n = retriever.index_nodes(batch_size=batch_size, max_workers=max_workers)
        return EmbedResponse(
            message       = f"Embedding complete." if n else "No new nodes to embed.",
            nodes_written = n,
        )
    except Exception as exc:
        logger.exception("embed_nodes failed")
        raise HTTPException(status_code=500, detail=str(exc))


@kg_retriever_router.post(
    "/index/reindex",
    response_model=EmbedResponse,
    summary="Force re-embed all Article nodes (overwrites existing embeddings)",
)
def reindex_articles(
    batch_size:  int            = Query(128, ge=1),
    max_workers: int            = Query(4,   ge=1),
    retriever:   LegalRetriever = Depends(get_retriever),
) -> EmbedResponse:
    """
    Use after swapping to a different embedding model or after bulk article
    text updates.
    """
    try:
        n = retriever.reindex_articles(batch_size=batch_size, max_workers=max_workers)
        return EmbedResponse(message="Article re-indexing complete.", nodes_written=n)
    except Exception as exc:
        logger.exception("reindex_articles failed")
        raise HTTPException(status_code=500, detail=str(exc))


@kg_retriever_router.post(
    "/index/rebuild",
    response_model=EmbedResponse,
    summary="Drop, recreate, and fully re-embed both Neo4j vector indexes",
)
def rebuild_index(
    batch_size:  int            = Query(128, ge=1),
    max_workers: int            = Query(4,   ge=1),
    dimensions:  Optional[int]  = Query(None, description="Override embedding dimensions"),
    retriever:   LegalRetriever = Depends(get_retriever),
) -> EmbedResponse:
    """
    Full rebuild — drops existing indexes, recreates them, and re-embeds
    every Article and TableChunk node.  Causes a brief period of degraded
    retrieval; call during a maintenance window.
    """
    try:
        n = retriever.rebuild_vector_index(
            batch_size  = batch_size,
            max_workers = max_workers,
            dimensions  = dimensions,
        )
        return EmbedResponse(message="Full vector index rebuild complete.", nodes_written=n)
    except Exception as exc:
        logger.exception("rebuild_index failed")
        raise HTTPException(status_code=500, detail=str(exc))