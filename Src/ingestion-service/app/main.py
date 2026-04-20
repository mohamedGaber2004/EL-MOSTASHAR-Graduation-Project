import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from app.Chunking import get_chunks, get_na2d_chunks

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield

app = FastAPI(
    title="Ingestion Service",
    description="Handles chunking for legal documents",
    version="0.1",
    lifespan=lifespan,
)

class ChunkData(BaseModel):
    content: str
    metadata: dict

class IngestResponse(BaseModel):
    chunks: List[ChunkData]
    total_chunks: int

class Na2dResponse(BaseModel):
    rulings: List[str]
    principles: List[str]
    total_rulings: int
    total_principles: int

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/ingest/laws", response_model=IngestResponse)
async def ingest_laws() -> IngestResponse:
    try:
        docs = get_chunks()
        chunks = [ChunkData(content=doc.page_content, metadata=doc.metadata) for doc in docs]
        return IngestResponse(chunks=chunks, total_chunks=len(chunks))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest/na2d", response_model=Na2dResponse)
async def ingest_na2d() -> Na2dResponse:
    try:
        result = get_na2d_chunks()
        rulings    = [doc.page_content for doc in result["rulings"]]
        principles = [doc.page_content for doc in result["principles"]]
        return Na2dResponse(
            rulings=rulings,
            principles=principles,
            total_rulings=len(rulings),
            total_principles=len(principles)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8004, reload=True)