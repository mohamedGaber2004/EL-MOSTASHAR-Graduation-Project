from __future__ import annotations

# =============================================================================
# IMPORTS
# =============================================================================

import logging
from functools import wraps
from typing import Any, Dict, List, Optional

from cachetools import TTLCache, cached
from cachetools.keys import hashkey
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from src.Config import get_settings
from src.Graphstore.KG_builder import LegalKnowledgeGraph

logger = logging.getLogger(__name__)

kg_router = APIRouter(prefix="/Services", tags=["Knowladge Graph"])


# =============================================================================
# CACHE CONFIGURATION
# =============================================================================
# Each endpoint family gets its own TTL bucket so hot (stats) vs cold (history)
# data can age independently.

_STATS_CACHE:      TTLCache = TTLCache(maxsize=1,   ttl=60)       # 1 min  — dashboard polling
_AMENDMENT_CACHE:  TTLCache = TTLCache(maxsize=256, ttl=300)      # 5 min  — per-law amendments
_HISTORY_CACHE:    TTLCache = TTLCache(maxsize=512, ttl=300)      # 5 min  — per-article history
_KEYWORD_CACHE:    TTLCache = TTLCache(maxsize=1024, ttl=120)     # 2 min  — search results


# =============================================================================
# GRAPH DEPENDENCY
# =============================================================================

def get_graph() -> LegalKnowledgeGraph:
    """
    FastAPI dependency — yields a *connected* graph instance.
    The driver is NOT closed between requests; Neo4j manages the pool.
    """
    cfg = get_settings()
    graph = LegalKnowledgeGraph(
        uri=cfg.NEO4J_URI,
        user=cfg.NEO4J_USERNAME,
        password=cfg.NEO4J_PASSWORD,
    )
    graph.connect()
    try:
        yield graph
    finally:
        graph.close()


# =============================================================================
# CACHE HELPER
# =============================================================================

def _cached_call(cache: TTLCache, fn, *args, **kwargs):
    """
    Generic thread-safe cache wrapper.
    Uses cachetools ``hashkey`` so mixed positional / keyword args hash correctly.
    """
    key = hashkey(*args, **kwargs)
    if key in cache:
        logger.debug("Cache HIT  | %s | key=%s", fn.__name__, key)
        return cache[key]
    logger.debug("Cache MISS | %s | key=%s", fn.__name__, key)
    result = fn(*args, **kwargs)
    cache[key] = result
    return result


# =============================================================================
# RESPONSE SCHEMAS
# =============================================================================

class StatisticsResponse(BaseModel):
    nodes:         Dict[str, int]
    relationships: Dict[str, int]

class AmendmentRow(BaseModel):
    amendment_id:         Optional[str] = None
    amendment_law_number: Optional[str] = None
    amendment_law_title:  Optional[str] = None
    amendment_date:       Optional[str] = None
    amendment_type:       Optional[str] = None
    description:          Optional[str] = None
    effective_date:       Optional[str] = None
    affected_articles:    List[str]     = Field(default_factory=list)
    law_id:               Optional[str] = None
    law_title:            Optional[str] = None

class ArticleHistoryRow(BaseModel):
    article:      Optional[str]       = None
    version:      Optional[str]       = None
    is_amended:   Optional[bool]      = None
    is_addition:  Optional[bool]      = None
    amendments:   List[Dict[str, Any]] = Field(default_factory=list)
    superseded_by: List[str]          = Field(default_factory=list)
    supersedes:   List[str]           = Field(default_factory=list)

class ArticleSearchRow(BaseModel):
    article_number: Optional[str] = None
    text:           Optional[str] = None
    law_id:         Optional[str] = None
    version:        Optional[str] = None
    is_amended:     Optional[bool] = None

class CacheInvalidateResponse(BaseModel):
    cleared: List[str]
    message: str


# =============================================================================
# ENDPOINTS
# =============================================================================

# ── Statistics ────────────────────────────────────────────────────────────────

@kg_router.get("/statistics",response_model=StatisticsResponse,summary="Node and relationship counts across the entire graph")
def get_statistics(graph: LegalKnowledgeGraph = Depends(get_graph)):
    try:
        data = _cached_call(_STATS_CACHE, graph.get_statistics)
        return StatisticsResponse(**data)
    except Exception as exc:
        logger.exception("get_statistics failed")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=str(exc))


# ── Amendments ────────────────────────────────────────────────────────────────

@kg_router.get("/amendments",response_model=List[AmendmentRow],summary="List amendments, optionally filtered by law",)
def query_amendments(law_id: Optional[str] = Query(None, description="Filter by law ID"),graph:  LegalKnowledgeGraph = Depends(get_graph)):
    try:
        rows = _cached_call(
            _AMENDMENT_CACHE, graph.query_amendments, law_id
        )
        return [AmendmentRow(**r) for r in rows]
    except Exception as exc:
        logger.exception("query_amendments failed (law_id=%s)", law_id)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=str(exc))


# ── Article history ───────────────────────────────────────────────────────────

@kg_router.get("/articles/{law_id}/{article_number}/history",response_model=List[ArticleHistoryRow],summary="Full amendment history for one article")
def query_article_history(law_id:         str,article_number: str,graph:          LegalKnowledgeGraph = Depends(get_graph)):
    try:
        rows = _cached_call(
            _HISTORY_CACHE, graph.query_article_history, law_id, article_number
        )
        return [ArticleHistoryRow(**r) for r in rows]
    except Exception as exc:
        logger.exception(
            "query_article_history failed (law_id=%s, article=%s)",
            law_id, article_number,
        )
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=str(exc))


# ── Keyword search ────────────────────────────────────────────────────────────

@kg_router.get("/articles/search",response_model=List[ArticleSearchRow],summary="Full-text keyword search across articles")
def search_articles(keyword: str = Query(..., min_length=2, description="Term to search in article text"),law_id:  Optional[str] = Query(None, description="Restrict search to one law"),k:int = Query(3, ge=1, le=50, description="Max results to return"),graph:   LegalKnowledgeGraph = Depends(get_graph)):
    try:
        rows = _cached_call(
            _KEYWORD_CACHE, graph.search_articles_by_keyword, keyword, law_id, k
        )
        return [ArticleSearchRow(**r) for r in rows]
    except Exception as exc:
        logger.exception("search_articles failed (keyword=%s)", keyword)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=str(exc))


# ── Cache management ──────────────────────────────────────────────────────────

_ALL_CACHES: Dict[str, TTLCache] = {
    "statistics": _STATS_CACHE,
    "amendments":  _AMENDMENT_CACHE,
    "history":     _HISTORY_CACHE,
    "keyword":     _KEYWORD_CACHE,
}


@kg_router.delete("/cache",response_model=CacheInvalidateResponse,summary="Invalidate one or all caches")
def invalidate_cache(name: Optional[str] = Query(None,description=("Cache bucket to clear: statistics | amendments | history | keyword. Omit to clear ALL buckets."))):
    if name and name not in _ALL_CACHES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown cache '{name}'. Valid names: {list(_ALL_CACHES)}",
        )

    targets = {name: _ALL_CACHES[name]} if name else _ALL_CACHES
    for cache in targets.values():
        cache.clear()

    cleared = list(targets.keys())
    logger.info("Cache(s) cleared: %s", cleared)
    return CacheInvalidateResponse(
        cleared=cleared,
        message=f"Cleared {len(cleared)} cache bucket(s) successfully.",
    )