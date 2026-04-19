from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Any
from contextlib import asynccontextmanager
from dotenv import load_dotenv

load_dotenv()

# Your actual LangGraph imports (moved from src/Graph)
from app.graph import run_case
from app.graph.state import AgentState

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up any heavy graph components here
    yield

app = FastAPI(
    title="Orchestrator Service",
    description="Runs the LangGraph legal pipeline",
    version="0.1",
    lifespan=lifespan,
)


class RunPipelineRequest(BaseModel):
    case_id: str
    source_documents: Optional[List[str]] = None


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/pipeline/run")
async def run_pipeline(payload: RunPipelineRequest) -> Any:
    """
    Entry point for the LangGraph agent.
    Receives a case, runs the full graph, returns the final state.
    """
    try:
        state = AgentState(
            case_id=payload.case_id,
            source_documents=payload.source_documents or []
        )
        final = run_case(state)

        if hasattr(final, "model_dump"):
            return final.model_dump()
        return final

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)