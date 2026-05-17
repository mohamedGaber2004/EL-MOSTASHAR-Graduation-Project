from __future__ import annotations

import logging
import threading
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel, Field

from src.Chunking.chunking import get_na2d_chunks
from src.Chunking.chunking_enums import Na2dOutputKey
from src.Config import get_settings
from src.Vectorstore.vector_store_builder import (
    build_vector_store,
    load_vector_store,
    search,
    search_with_scores,
)

logger = logging.getLogger(__name__)

vs_router = APIRouter(prefix="/vs", tags=["Vector Store"])


# =============================================================================

# Singleton
# =============================================================================

_vs      = None
_VS_LOCK = threading.Lock()


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=get_settings().ARABIC_NATIVE_EMBEDDING_MODEL)

def get_vs():
    """FastAPI dependency — raises 503 when no index is loaded yet."""
    if _vs is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector store not loaded. Call POST /vs/build or POST /vs/load first.",
        )
    return _vs


# =============================================================================
# Request / Response schemas
# =============================================================================

class VSStatusResponse(BaseModel):
    loaded: bool

class BuildVSResponse(BaseModel):
    message:          str
    rulings_count:    int
    principles_count: int
    total_count:      int
    index_path:       str

class LoadVSResponse(BaseModel):
    message:    str
    index_path: str

class SearchRequest(BaseModel):
    query: str = Field(..., description="Free-text search query (Arabic)")
    k:     int = Field(5,   description="Number of results", ge=1, le=50)

class SearchHit(BaseModel):
    page_content: str
    metadata:     dict[str, Any] = Field(default_factory=dict)

class ScoredHit(BaseModel):
    page_content: str
    metadata:     dict[str, Any] = Field(default_factory=dict)
    score:        float          = Field(..., description="L2 distance — lower means more similar")

class SearchResponse(BaseModel):
    query: str
    hits:  list[SearchHit]

class ScoredSearchResponse(BaseModel):
    query: str
    hits:  list[ScoredHit]


# =============================================================================
# Endpoints
# =============================================================================

# ── Status ────────────────────────────────────────────────────────────────────

@vs_router.get(
    "/status",
    response_model=VSStatusResponse,
    summary="Check whether a vector store is currently loaded",
)
def vs_status() -> VSStatusResponse:
    return VSStatusResponse(loaded=_vs is not None)


# ── Build ─────────────────────────────────────────────────────────────────────

@vs_router.post(
    "/build",
    response_model=BuildVSResponse,
    summary="Build a FAISS index from the Na2d corpus (rulings + principles)",
)
def build_vs_endpoint(
    index_path: str = Query("na2d_faiss_index", description="Directory to save the index"),
) -> BuildVSResponse:
    global _vs
    try:
        na2d       = get_na2d_chunks()
        rulings    = na2d[Na2dOutputKey.RULINGS]
        principles = na2d[Na2dOutputKey.PRINCIPLES]
        docs       = rulings + principles
        embeddings = get_embeddings()

        if not docs:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Na2d corpus is empty — nothing to index.",
            )

        with _VS_LOCK:
            _vs = build_vector_store(docs, embeddings, index_path=index_path)

        logger.info(
            "VS built: %d docs (%d rulings + %d principles) → %s",
            len(docs), len(rulings), len(principles), index_path,
        )
        return BuildVSResponse(
            message          = "Vector store built and loaded successfully.",
            rulings_count    = len(rulings),
            principles_count = len(principles),
            total_count      = len(docs),
            index_path       = index_path,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("build_vs_endpoint failed")
        raise HTTPException(status_code=500, detail=str(exc))


# ── Load ──────────────────────────────────────────────────────────────────────

@vs_router.post(
    "/load",
    response_model=LoadVSResponse,
    summary="Load a previously persisted FAISS index into memory",
)
def load_vs_endpoint(
    index_path: str = Query("na2d_faiss_index", description="Directory of the saved index"),
) -> LoadVSResponse:
    global _vs
    try:
        embeddings = get_embeddings()
        with _VS_LOCK:
            _vs = load_vector_store(index_path, embeddings)

        logger.info("VS loaded from %s", index_path)
        return LoadVSResponse(
            message    = f"Vector store loaded from '{index_path}'.",
            index_path = index_path,
        )

    except Exception as exc:
        logger.exception("load_vs_endpoint failed")
        raise HTTPException(status_code=500, detail=str(exc))


# ── Search ────────────────────────────────────────────────────────────────────

@vs_router.post(
    "/search",
    response_model=SearchResponse,
    summary="Similarity search over rulings and principles",
)
def vs_search(
    req: SearchRequest,
    vs  = Depends(get_vs),
) -> SearchResponse:
    try:
        docs = search(vs, req.query,k = req.k)
        return SearchResponse(
            query = req.query,
            hits  = [SearchHit(page_content=d.page_content, metadata=d.metadata) for d in docs],
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("vs_search failed")
        raise HTTPException(status_code=500, detail=str(exc))


# ── Search with scores ────────────────────────────────────────────────────────

@vs_router.post(
    "/search/scored",
    response_model=ScoredSearchResponse,
    summary="Similarity search returning L2 distance scores",
)
def vs_search_scored(
    req: SearchRequest,
    vs  = Depends(get_vs),
) -> ScoredSearchResponse:
    try:
        pairs = search_with_scores(vs, req.query, k=req.k)
        return ScoredSearchResponse(
            query=req.query,
            hits=[
                ScoredHit(page_content=doc.page_content, metadata=doc.metadata, score=score)
                for doc, score in pairs
            ],
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("vs_search_scored failed")
        raise HTTPException(status_code=500, detail=str(exc))
