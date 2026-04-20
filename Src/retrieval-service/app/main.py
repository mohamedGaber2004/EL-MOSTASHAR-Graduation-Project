from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import os

load_dotenv()

# Your actual vectorstore logic (moved from src/Vectorstore)
from app.Vectorstore.vector_store_builder import load_vector_store, search, search_with_scores
from app.config import get_settings
from langchain_huggingface import HuggingFaceEmbeddings

vectorstore = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global vectorstore
    # Warm up FAISS index on startup
    try:
        cfg = get_settings()
        embeddings = HuggingFaceEmbeddings(model_name=cfg.EMBEDDING_MODEL)
        index_path = os.getenv("FAISS_INDEX_PATH", "app/faiss_index")
        vectorstore = load_vector_store(index_path, embeddings)
    except Exception as e:
        print(f"[WARNING] Could not load vectorstore on startup: {e}")
    yield

app = FastAPI(
    title="Retrieval Service",
    description="Handles vector search over legal documents",
    version="0.1",
    lifespan=lifespan,
)


class RetrieveRequest(BaseModel):
    query: str
    top_k: int = 5
    doc_type: Optional[str] = None


class RetrieveResponse(BaseModel):
    chunks: List[str]
    scores: List[float] = []


@app.get("/health")
async def health():
    return {"status": "ok", "vectorstore_loaded": vectorstore is not None}


@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(req: RetrieveRequest) -> RetrieveResponse:
    """
    Search the FAISS index and return the top_k most relevant chunks.
    """
    if vectorstore is None:
        raise HTTPException(status_code=503, detail="Vectorstore not loaded yet")

    try:
        results = search_with_scores(vectorstore, req.query, req.top_k, doc_type=req.doc_type)
        return RetrieveResponse(
            chunks=[doc.page_content for doc, _score in results],
            scores=[float(_score) for _doc, _score in results]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8002, reload=True)