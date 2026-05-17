from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from src.Chunking.chunking import get_na2d_chunks
from src.retriever.vs_retriever.vs_reriever import (
    get_dense_retriever,
    get_hybrid_retriever,
    get_sparse_retriever,
)
from src.Vectorstore.vs_builder_enums import ChunkType, DocType
from src.routers.vs_router import get_vs
from src.Chunking.chunking_enums import Na2dOutputKey

logger = logging.getLogger(__name__)

vs_retriever_router = APIRouter(prefix="/vs/retriever", tags=["VS Retriever"])


# =============================================================================
# Request / Response schemas
# =============================================================================

class RetrieveRequest(BaseModel):
    query:      str            = Field(...,  description="Free-text query")
    k:          int            = Field(5,    description="Number of documents to retrieve", ge=1, le=50)

class HybridRetrieveRequest(RetrieveRequest):
    dense_weight:  float = Field(0.6, description="Relative weight for FAISS results", ge=0.0, le=1.0)
    sparse_weight: float = Field(0.4, description="Relative weight for BM25 results",  ge=0.0, le=1.0)

class RetrievedDoc(BaseModel):
    page_content: str
    metadata:     dict[str, Any] = Field(default_factory=dict)

class RetrieveResponse(BaseModel):
    query:    str
    strategy: str
    hits:     list[RetrievedDoc]


# =============================================================================
# Endpoints
# =============================================================================

# ── Dense (FAISS) ─────────────────────────────────────────────────────────────

@vs_retriever_router.post(
    "/dense",
    response_model=RetrieveResponse,
    summary="Semantic retrieval via FAISS dense vectors",
)
def dense_retrieve(
    req: RetrieveRequest,
    vs  = Depends(get_vs),
) -> RetrieveResponse:
    """
    Wraps the loaded FAISS store as a LangChain retriever and invokes it.
    Pure semantic (cosine / L2) search — no lexical component.
    """
    try:
        retriever = get_dense_retriever(vs,k= req.k,)
        docs = retriever.invoke(req.query)
        return RetrieveResponse(
            query    = req.query,
            strategy = "dense",
            hits     = [RetrievedDoc(page_content=d.page_content, metadata=d.metadata) for d in docs],
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("dense_retrieve failed")
        raise HTTPException(status_code=500, detail=str(exc))


# ── Sparse (BM25) ─────────────────────────────────────────────────────────────

@vs_retriever_router.post(
    "/sparse",
    response_model=RetrieveResponse,
    summary="Lexical retrieval via in-memory BM25 (Okapi)",
)
def sparse_retrieve(req: RetrieveRequest) -> RetrieveResponse:
    """
    Builds an in-memory BM25 index over the current chunk corpus (filtered by
    the supplied metadata constraints) and retrieves the top-k results.

    Note: BM25 is rebuilt per request from the live corpus.  For high-QPS
    scenarios consider caching the index at application startup.
    """
    try:
        na2d = get_na2d_chunks()
        docs = na2d[Na2dOutputKey.RULINGS] + na2d[Na2dOutputKey.PRINCIPLES]
        retriever = get_sparse_retriever(docs,k= req.k)
        results = retriever.invoke(req.query)
        return RetrieveResponse(
            query    = req.query,
            strategy = "sparse",
            hits     = [RetrievedDoc(page_content=d.page_content, metadata=d.metadata) for d in results],
        )
    except HTTPException:
        raise
    except ValueError as exc:
        # empty corpus after filtering
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    except Exception as exc:
        logger.exception("sparse_retrieve failed")
        raise HTTPException(status_code=500, detail=str(exc))


# ── Hybrid (FAISS + BM25 via RRF) ────────────────────────────────────────────

@vs_retriever_router.post(
    "/hybrid",
    response_model=RetrieveResponse,
    summary="Hybrid retrieval: RRF fusion of FAISS (dense) and BM25 (sparse)",
)
def hybrid_retrieve(
    req: HybridRetrieveRequest,
    vs  = Depends(get_vs),
) -> RetrieveResponse:
    """
    Combines FAISS semantic search and BM25 lexical search using
    Reciprocal Rank Fusion (RRF).  *dense_weight* and *sparse_weight* are
    normalised internally so they need not sum to 1.

    This is the recommended strategy for Arabic legal text, where exact
    term matches (article numbers, specific legal terms) complement
    semantic similarity.
    """
    try:
        if req.dense_weight <= 0 or req.sparse_weight <= 0:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Both dense_weight and sparse_weight must be > 0.",
            )

        na2d = get_na2d_chunks()
        docs = na2d[Na2dOutputKey.RULINGS] + na2d[Na2dOutputKey.PRINCIPLES]
        retriever = get_hybrid_retriever(
            vs,
            docs,
            k             = req.k,
            dense_weight  = req.dense_weight,
            sparse_weight = req.sparse_weight,
        )
        results = retriever.invoke(req.query)
        return RetrieveResponse(
            query    = req.query,
            strategy = f"hybrid(dense={req.dense_weight:.2f}, sparse={req.sparse_weight:.2f})",
            hits     = [RetrievedDoc(page_content=d.page_content, metadata=d.metadata) for d in results],
        )
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    except Exception as exc:
        logger.exception("hybrid_retrieve failed")
        raise HTTPException(status_code=500, detail=str(exc))


# =============================================================================
# Internal helpers
# =============================================================================

