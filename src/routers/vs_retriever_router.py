from __future__ import annotations

import logging
import threading
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from src.Chunking.chunking import get_na2d_chunks
from src.Chunking.chunking_enums import Na2dOutputKey
from src.retriever.vs_retriever.vs_reriever import (
    get_dense_retriever,
    get_hybrid_retriever,
    get_sparse_retriever,
)
from src.routers.vs_router import get_vs

logger = logging.getLogger(__name__)

vs_retriever_router = APIRouter(prefix="/vs/retriever", tags=["VS Retriever"])

_cached_bm25_retriever: Optional[Any] = None
_BM25_LOCK = threading.Lock()

def get_cached_sparse_retriever() -> Any:
    """Lazily loads and caches the BM25 index over the entire corpus."""
    global _cached_bm25_retriever
    if _cached_bm25_retriever is None:
        with _BM25_LOCK:
            if _cached_bm25_retriever is None:
                logger.info("Initializing global BM25 sparse corpus index...")
                na2d = get_na2d_chunks()
                docs = na2d[Na2dOutputKey.RULINGS] + na2d[Na2dOutputKey.PRINCIPLES]
                if not docs:
                    raise ValueError("Empty corpus returned from get_na2d_chunks.")
                
                # Build default index
                _cached_bm25_retriever = get_sparse_retriever(docs, k=5)
                logger.info("Global BM25 sparse corpus index established successfully.")
    return _cached_bm25_retriever


# =============================================================================
# Request / Response schemas
# =============================================================================

class RetrieveRequest(BaseModel):
    query:      str            = Field(...,  description="Free-text query")
    k:          int            = Field(5,    description="Number of documents to retrieve", ge=1, le=50)

class HybridRetrieveRequest(RetrieveRequest):
    query:      str            = Field(...,  description="Free-text query")
    k:          int            = Field(5,    description="Number of documents to retrieve", ge=1, le=50)
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
@vs_retriever_router.post("/dense",response_model=RetrieveResponse,summary="Semantic retrieval via FAISS dense vectors")
def dense_retrieve(req: RetrieveRequest,vs  = Depends(get_vs)) -> RetrieveResponse:
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
@vs_retriever_router.post("/sparse", response_model=RetrieveResponse, summary="Lexical retrieval via cached BM25 (Okapi)")
def sparse_retrieve(req: RetrieveRequest, sparse_retriever = Depends(get_cached_sparse_retriever)) -> RetrieveResponse:
    try:
        # Avoid rebuilding; update runtime parameter k dynamically
        sparse_retriever.k = req.k
        results = sparse_retriever.invoke(req.query)
        return RetrieveResponse(
            query    = req.query,
            strategy = "sparse",
            hits     = [RetrievedDoc(page_content=d.page_content, metadata=d.metadata) for d in results],
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    except Exception as exc:
        logger.exception("sparse_retrieve failed")
        raise HTTPException(status_code=500, detail=str(exc))


# ── Hybrid (FAISS + BM25 via RRF) ────────────────────────────────────────────
@vs_retriever_router.post("/hybrid", response_model=RetrieveResponse, summary="Hybrid retrieval: RRF fusion of FAISS and BM25")
def hybrid_retrieve(req: HybridRetrieveRequest, vs = Depends(get_vs), sparse_retriever = Depends(get_cached_sparse_retriever)) -> RetrieveResponse:
    try:
        if req.dense_weight <= 0 and req.sparse_weight <= 0:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="At least one weight (dense_weight or sparse_weight) must be strictly greater than 0.",
            )

        # Retrieve underlying document list directly from cached sparse index metadata to avoid extraction recalculation
        docs = sparse_retriever.docs if hasattr(sparse_retriever, 'docs') else []
        if not docs:
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
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    except Exception as exc:
        logger.exception("hybrid_retrieve failed")
        raise HTTPException(status_code=500, detail=str(exc))

