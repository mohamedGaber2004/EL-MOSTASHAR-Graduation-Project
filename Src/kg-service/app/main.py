from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager
from dotenv import load_dotenv

load_dotenv()

from kg_service.app.Graphstore.KG_builder import LegalKnowledgeGraph, build_knowledge_graph
from kg_service.app.config import get_settings

kg: Optional[LegalKnowledgeGraph] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global kg
    try:
        kg = build_knowledge_graph(
            neo4j_uri=get_settings().NEO4J_URI,
            neo4j_user=get_settings().NEO4J_USERNAME,
            neo4j_password=get_settings().NEO4J_PASSWORD,
            drop_existing=True,
            verbose=True,
        )
    except Exception as e:
        print(f"[WARNING] Could not build knowledge graph on startup: {e}")
    yield
    if kg is not None:
        kg.close()


app = FastAPI(
    title="Knowledge Graph Service",
    description="Manages the legal knowledge graph (Neo4j)",
    version="0.1",
    lifespan=lifespan,
)


# ── Request / Response Models ──────────────────────────────────────────────────

class AmendmentsQueryRequest(BaseModel):
    law_id: Optional[str] = None  # if omitted, returns amendments for all laws


class ArticleHistoryRequest(BaseModel):
    law_id: str
    article_number: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

def _require_kg() -> LegalKnowledgeGraph:
    if kg is None:
        raise HTTPException(status_code=503, detail="Knowledge graph not connected")
    return kg


@app.get("/health")
async def health():
    return {"status": "ok", "kg_connected": kg is not None}


@app.get("/statistics")
async def get_statistics() -> Dict[str, Any]:
    """
    Return node and relationship counts for every label/type in the graph.
    """
    graph = _require_kg()
    try:
        return graph.get_statistics()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/amendments")
async def query_amendments(req: AmendmentsQueryRequest) -> List[Dict[str, Any]]:
    """
    Return all amendments, optionally filtered to a single law.

    - Leave `law_id` empty to get every amendment across all laws.
    - Pass a `law_id` to scope results to that law.
    """
    graph = _require_kg()
    try:
        return graph.query_amendments(law_id=req.law_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/article-history")
async def query_article_history(req: ArticleHistoryRequest) -> List[Dict[str, Any]]:
    """
    Return the full amendment / version history for a specific article.
    """
    graph = _require_kg()
    try:
        results = graph.query_article_history(
            law_id=req.law_id,
            article_number=req.article_number,
        )
        if not results:
            raise HTTPException(
                status_code=404,
                detail=f"No history found for article {req.article_number} in law {req.law_id}",
            )
        return results
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8003, reload=True)