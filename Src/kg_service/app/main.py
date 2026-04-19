from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import os

load_dotenv()

# Your actual graphstore logic (moved from src/Graphstore)
from app.graphstore import connect_kg, query_entity, add_entity

kg = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global kg
    try:
        kg = connect_kg(
            uri=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USERNAME"),
            password=os.getenv("NEO4J_PASSWORD"),
        )
    except Exception as e:
        print(f"[WARNING] Could not connect to Neo4j on startup: {e}")
    yield
    if kg:
        kg.close()

app = FastAPI(
    title="Knowledge Graph Service",
    description="Manages the legal knowledge graph (Neo4j)",
    version="0.1",
    lifespan=lifespan,
)


class KGQueryRequest(BaseModel):
    entity: str
    depth: int = 1


class KGQueryResponse(BaseModel):
    relations: List[Dict[str, Any]]


class KGAddRequest(BaseModel):
    entity: str
    relations: List[Dict[str, Any]]


@app.get("/health")
async def health():
    return {"status": "ok", "kg_connected": kg is not None}


@app.post("/query", response_model=KGQueryResponse)
async def query_kg(req: KGQueryRequest) -> KGQueryResponse:
    """
    Query the knowledge graph for an entity and its relations.
    """
    if kg is None:
        raise HTTPException(status_code=503, detail="Knowledge graph not connected")

    try:
        relations = query_entity(kg, req.entity, req.depth)
        return KGQueryResponse(relations=relations)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/add")
async def add_to_kg(req: KGAddRequest) -> Dict[str, str]:
    """
    Add an entity and its relations to the knowledge graph.
    """
    if kg is None:
        raise HTTPException(status_code=503, detail="Knowledge graph not connected")

    try:
        add_entity(kg, req.entity, req.relations)
        return {"status": "added", "entity": req.entity}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8003, reload=True)