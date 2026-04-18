from fastapi import FastAPI
from src.routers.case_router import router as case_router

from src.pipeline import default_pipeline_manager
from contextlib import asynccontextmanager


# Create pipeline manager (lazy init)
pipeline = default_pipeline_manager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up heavy components (FAISS, embeddings) if available.
    # Keep warmup optional: if it fails, the pipeline will initialize on demand.
    try:
        pipeline.init_components()
    except Exception:
        pass
    yield


app = FastAPI(
    title="Legal AI System",
    description="AI-powered legal case analysis system",
    version="1.0.0",
    lifespan=lifespan,
)


# Include routers
app.include_router(case_router)


@app.get("/")
async def root():
    return {"message": "Welcome to the Legal AI System API"}


if __name__ == "__main__":
    import uvicorn
    # When using `reload=True` or multiple workers, uvicorn requires the
    # application to be provided as an import string so it can re-import
    # the module in the child process. Use "main:app" here.
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
