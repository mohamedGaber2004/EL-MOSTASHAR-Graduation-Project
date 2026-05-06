import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException, status ,Depends
from pydantic import BaseModel, Field
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

from src.Config.config import get_settings
from src.Graphstore.graph_rag import LegalGraphRAG
from src.Graphstore.KG_builder import LegalKnowledgeGraph
from src.routers.kg_router import  get_graph
from functools import lru_cache

# Configure logging for the router
logger = logging.getLogger(__name__)


graph_rag_router = APIRouter(prefix="/graph_rag", tags=["Graph RAG"])

@lru_cache(maxsize=1)
def _get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=get_settings().ARABIC_NATIVE_EMBEDDING_MODEL)

@lru_cache(maxsize=1)
def _get_llm() -> ChatGroq:
    cfg = get_settings()
    return ChatGroq(model=cfg.JUDGE_MODEL, temperature=cfg.JUDGE_TEMP, api_key=cfg.GROQ_API_KEY)

# =============================================================================
# SCHEMAS (Models)
# =============================================================================

class GraphRAGQueryRequest(BaseModel):
    question:      str           = Field(..., description="Legal question to ask the Graph RAG pipeline")
    k:             Optional[int] = Field(5, ge=1, le=20, description="Number of top articles to retrieve")
    system_prompt: Optional[str] = Field(None, description="Optional override for the LLM system prompt")

class GraphRAGSource(BaseModel):
    law_id: str
    law_title: Optional[str] = None
    article_number: str
    score: str
    text: str

class GraphRAGQueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[GraphRAGSource]

class GraphRAGActionResponse(BaseModel):
    success: bool
    message: str
    count: Optional[int] = 0

# =============================================================================
# DEPENDENCY / UTILITY
# =============================================================================

def get_rag_pipeline(graph: LegalKnowledgeGraph = Depends(get_graph)) -> LegalGraphRAG:
    """
    Constructs the RAG pipeline using the SHARED graph instance.
    """
    return LegalGraphRAG(graph=graph, embeddings=_get_embeddings(), llm=_get_llm())

# =============================================================================
# ENDPOINTS
# =============================================================================

@graph_rag_router.post("/query", response_model=GraphRAGQueryResponse)
def query_graph_rag(payload: GraphRAGQueryRequest, rag: LegalGraphRAG = Depends(get_rag_pipeline)):
    try:
        result = rag.query(
            question=payload.question,
            k=payload.k,
            system_prompt=payload.system_prompt,
        )
        return GraphRAGQueryResponse(
            query=result["query"],
            answer=result["answer"],
            sources=[GraphRAGSource(**src) for src in result["sources"]],
        )
    except Exception as exc:
        logger.exception("Error during Graph RAG query")
        raise HTTPException(status_code=500, detail=str(exc))

@graph_rag_router.post("/setup-index", response_model=GraphRAGActionResponse)
def setup_graph_rag_index(rag: LegalGraphRAG = Depends(get_rag_pipeline)):
    try:
        rag.setup_vector_index()
        return GraphRAGActionResponse(success=True, message="Vector index is ready.")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@graph_rag_router.post("/index-articles", response_model=GraphRAGActionResponse)
def index_graph_rag_articles(
    batch_size: int = 128, 
    max_workers: int = 4, 
    rag: LegalGraphRAG = Depends(get_rag_pipeline)
):
    try:
        count = rag.index_articles(batch_size=batch_size, max_workers=max_workers)
        return GraphRAGActionResponse(success=True, message="Indexing complete.", count=count)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@graph_rag_router.post("/rebuild-index", response_model=GraphRAGActionResponse)
def rebuild_graph_rag_index(
    batch_size: int = 128, 
    max_workers: int = 4, 
    rag: LegalGraphRAG = Depends(get_rag_pipeline)
):
    try:
        count = rag.rebuild_vector_index(batch_size=batch_size, max_workers=max_workers)
        return GraphRAGActionResponse(success=True, message="Rebuild complete.", count=count)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))