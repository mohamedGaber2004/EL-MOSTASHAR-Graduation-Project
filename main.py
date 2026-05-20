from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from src.Config.config import get_settings
from src.routers.case_router import router as case_router
from src.routers.chunking_router import chunking_router
from src.routers.data_ingestion_router import data_ingestion_router
from src.routers.kg_router import get_graph, kg_router
# Updated dependency names to prevent explicit initialization ImportErrors
from src.routers.kg_retriever_router import get_kg_retriever, kg_retriever_router
from src.routers.vs_retriever_router import get_hybrid_retriever, vs_retriever_router
from src.routers.vs_router import vs_router

logger = logging.getLogger(__name__)

# =============================================================================
# Environment & Observability Configuration
# =============================================================================
cfg = get_settings()
os.environ["LANGCHAIN_TRACING_V2"]  = str(cfg.LANGSMITH_TRACING).lower()
os.environ["LANGCHAIN_API_KEY"]     = cfg.LANGSMITH_API_KEY or ""
os.environ["LANGCHAIN_ENDPOINT"]    = cfg.LANGSMITH_ENDPOINT or ""
os.environ["LANGCHAIN_PROJECT"]     = cfg.LANGSMITH_PROJECT or ""


# =============================================================================
# Application Lifespan Control
# =============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Warming up engines and pre-building indices...")
    graph = get_graph() 
    retriever = get_kg_retriever()
    logger.info("✅ All retrievers and structural indices are online!")
    
    yield

    logger.info("🛑 Shutting down server, cleaning up structural resources...")
    try:
        graph = get_graph()
        if graph is not None and hasattr(graph, "close"):
            graph.close()
            logger.info("🔌 Neo4j Graph connection pools drained and closed.")
    except Exception as e:
        logger.error(f"⚠️ Error during graph resource cleanup: {e}", exc_info=True)


# =============================================================================
# FastAPI Initialization & Route Routing
# =============================================================================
app = FastAPI(
    title="Egyptian Legal Multi-Agent System API",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/", tags=["Root"])
def read_root():
    return {"message": "Welcome to the Graduation Project API!"}


# Mount application routers cleanly
app.include_router(case_router)
app.include_router(data_ingestion_router)
app.include_router(chunking_router)
app.include_router(kg_router)
app.include_router(kg_retriever_router)
app.include_router(vs_router)
app.include_router(vs_retriever_router)


# =============================================================================
# Local Execution Entry Point
# =============================================================================
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)