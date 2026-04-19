from fastapi import FastAPI
from contextlib import asynccontextmanager
import httpx
import os
from dotenv import load_dotenv

from app.routers import case_router

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # nothing heavy to warm up in gateway
    yield

app = FastAPI(
    title="EL-Mosta4ar Gateway",
    description="API Gateway for the Legal AI System",
    version="0.1",
    lifespan=lifespan,
)

app.include_router(case_router.router)

@app.get("/")
async def root():
    return {"message": "Welcome to the Legal AI System API"}

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8080, reload=True)