# Import necessary modules
import uvicorn , os
from fastapi import FastAPI
from contextlib import asynccontextmanager
from src.Config.config import get_settings
from src.routers.case_router import router as case_router
from src.routers.data_ingestion_router import data_ingestion_router
from src.routers.chunking_router import chunking_router
from src.routers.graph_rag_router import graph_rag_router
from src.routers.kg_router import kg_router , get_graph


cfg = get_settings()
os.environ["LANGCHAIN_TRACING_V2"]  = str(cfg.LANGSMITH_TRACING).lower()
os.environ["LANGCHAIN_API_KEY"]     = cfg.LANGSMITH_API_KEY or ""
os.environ["LANGCHAIN_ENDPOINT"]    = cfg.LANGSMITH_ENDPOINT or ""
os.environ["LANGCHAIN_PROJECT"]     = cfg.LANGSMITH_PROJECT or ""





@asynccontextmanager
async def lifespan(app: FastAPI):
    yield                        # app is running
    graph = get_graph()
    if graph is not None:
        graph.close()

app = FastAPI(lifespan=lifespan)

# Define a simple root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Graduation Project API!"}

# Include the case router in the app
app.include_router(case_router)
app.include_router(data_ingestion_router)
app.include_router(chunking_router)
app.include_router(graph_rag_router)
app.include_router(kg_router)



# Entry point for running the app
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)