from __future__ import annotations

# =============================================================================
# IMPORTS
# =============================================================================

import logging
from typing import Dict, List, Optional


from cachetools import TTLCache
import threading
from cachetools.keys import hashkey
from fastapi import BackgroundTasks ,APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from src.Graphstore.KG_builder import build_knowledge_graph
from src.Config import get_settings
from src.Graphstore.KG_builder import LegalKnowledgeGraph
from src.Chunking.chunking import get_chunks

logger = logging.getLogger(__name__)

kg_router = APIRouter(prefix="/kg", tags=["Knowledge Graph"])


# =============================================================================
# CACHE CONFIGURATION
# =============================================================================
# Each endpoint family gets its own TTL bucket so hot (stats) vs cold (history)
# data can age independently.

_STATS_CACHE:      TTLCache = TTLCache(maxsize=1,   ttl=60)
_AMENDMENT_CACHE:  TTLCache = TTLCache(maxsize=256, ttl=300)
_HISTORY_CACHE:    TTLCache = TTLCache(maxsize=512, ttl=300)


# =============================================================================
# GRAPH DEPENDENCY
# =============================================================================

_graph: Optional[LegalKnowledgeGraph] = None 

_GRAPH_LOCK = threading.Lock()

def get_graph() -> LegalKnowledgeGraph:
    global _graph
    if _graph is None:
        with _GRAPH_LOCK:
            if _graph is None:
                cfg = get_settings()
                _graph = LegalKnowledgeGraph(
                    uri=cfg.NEO4J_URI,
                    user=cfg.NEO4J_USERNAME,
                    password=cfg.NEO4J_PASSWORD,
                )
                _graph.connect()
                logger.info("Neo4j singleton created")
    return _graph

# =============================================================================
# CACHE HELPER
# =============================================================================

_CACHE_LOCK = threading.Lock()

def _cached_call(cache, fn, *args, **kwargs):
    key = hashkey(*args, **kwargs)
    with _CACHE_LOCK:
        if key in cache:
            return cache[key]
    result = fn(*args, **kwargs)
    with _CACHE_LOCK:
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
class ArticleResponse(BaseModel):
    """Schema for direct article text retrieval."""
    law_id:         str = Field(..., description="The unique ID of the law")
    article_number: str = Field(..., description="The article number")
    text:           str = Field(..., description="The full text of the article")
class CacheInvalidateResponse(BaseModel):
    cleared: List[str]
    message: str
class BuildKGResponse(BaseModel):
    message: str

# =============================================================================
# ENDPOINTS
# =============================================================================

# ── Statistics ────────────────────────────────────────────────────────────────

_STATS_KEY = "statistics"   # constant key — stats has no per-request arguments

@kg_router.get("/statistics", response_model=StatisticsResponse)
def get_statistics(graph: LegalKnowledgeGraph = Depends(get_graph)):
    try:
        # Use a fixed string key instead of hashkey(graph.get_statistics)
        if _STATS_KEY in _STATS_CACHE:
            logger.debug("Cache HIT | statistics")
            return StatisticsResponse(**_STATS_CACHE[_STATS_KEY])
        
        data = graph.get_statistics()
        _STATS_CACHE[_STATS_KEY] = data
        logger.debug("Cache MISS | statistics — fetched from Neo4j")
        return StatisticsResponse(**data)

    except Exception as exc:
        logger.exception("get_statistics failed")
        raise HTTPException(status_code=500, detail=str(exc))

# ── Amendments ────────────────────────────────────────────────────────────────

@kg_router.get("/amendments",response_model=List[AmendmentRow],summary="List amendments, optionally filtered by law")
def query_amendments(law_id: Optional[str] = Query(None, description="Filter by law ID"),graph:  LegalKnowledgeGraph = Depends(get_graph)):
    """
    Returns all amendments across every law, or only those belonging to
    ``law_id`` when provided.
    Response is cached per ``law_id`` for **5 minutes**.
    """
    try:
        rows = _cached_call(
            _AMENDMENT_CACHE, graph.query_amendments, law_id
        )
        return [AmendmentRow(**r) for r in rows]
    except Exception as exc:
        logger.exception("query_amendments failed (law_id=%s)", law_id)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=str(exc))

# ── Query Articel ────────────────────────────────────────────────────────────────

@kg_router.get("/{law_id}/{article_number}", response_model=ArticleResponse, summary="Get Article Text")
def fetch_article_text(law_id: str, article_number: str, graph: LegalKnowledgeGraph = Depends(get_graph)):
    """
    Fetches the latest text for a specific article within a law.
    If multiple versions exist, it returns the most recent one.
    """
    try:
        text = _cached_call(
            _HISTORY_CACHE, graph.get_article, law_id, article_number
        )
        
        if text is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Article {article_number} not found in law {law_id}"
            )
            
        return ArticleResponse(
            law_id=law_id,
            article_number=article_number,
            text=text
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception(
            "get_article failed (law_id=%s, article=%s)",
            law_id, article_number,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred while retrieving the article."
        )

# ── Build KG ────────────────────────────────────────────────────────────────

@kg_router.post("/build",response_model=BuildKGResponse,summary="Build or rebuild the Knowledge Graph")
def build_kg_endpoint(background_tasks: BackgroundTasks,drop_existing: bool = Query(True, description="Drop existing graph before rebuilding")):

    cfg = get_settings()

    def _run_build():
        try:
            logger.info("KG build started...")
            build_knowledge_graph(
                neo4j_uri=cfg.NEO4J_URI,
                neo4j_user=cfg.NEO4J_USERNAME,
                neo4j_password=cfg.NEO4J_PASSWORD,
                drop_existing=drop_existing,
                verbose=True,
                chunks=get_chunks()
            )

            # invalidate cache after rebuild
            for cache in _ALL_CACHES.values():
                cache.clear()

            logger.info("KG build completed successfully")

        except Exception as e:
            logger.exception("KG build failed: %s", e)

    background_tasks.add_task(_run_build)

    return BuildKGResponse(
        message="Knowledge Graph build started in background."
    )

# ── Cache management ──────────────────────────────────────────────────────────

_ALL_CACHES: Dict[str, TTLCache] = {
    "statistics": _STATS_CACHE,
    "amendments":  _AMENDMENT_CACHE,
    "history":     _HISTORY_CACHE,
}

@kg_router.delete("/cache",response_model=CacheInvalidateResponse,summary="Invalidate one or all caches")
def invalidate_cache(
    name: Optional[str] = Query(
        None,
        description=(
            "Cache bucket to clear: statistics | amendments | history | keyword. "
            "Omit to clear ALL buckets."
        ),
    ),
):
    """
    Immediately evicts all entries from the named cache bucket, or from every
    bucket when ``name`` is not provided.

    Use after importing new data so the next request fetches fresh results.
    """
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