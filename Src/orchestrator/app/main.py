from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Any
from contextlib import asynccontextmanager
from dotenv import load_dotenv

load_dotenv()

# Your actual LangGraph imports (moved from src/Graph)
from app.Graph import run_case
from app.Graph.state import AgentState
from app.config import get_settings

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
        from pathlib import Path
        cfg = get_settings()
        docs = payload.source_documents or []

        # Auto-resolve: if no documents given, look in CASES_DIR/<case_id>
        if not docs:
            cases_dir = Path(getattr(cfg, "CASES_DIR", "Datasets/User_Cases"))
            case_path = cases_dir / payload.case_id
            if case_path.is_dir():
                # Include all .txt and .pdf files in that folder
                docs = [str(f) for f in sorted(case_path.iterdir())
                        if f.suffix.lower() in (".txt", ".pdf")]
            elif case_path.with_suffix(".txt").exists():
                docs = [str(case_path.with_suffix(".txt"))]
            # If still empty, pass the path itself so the agent reports a clear error
            if not docs:
                docs = [str(case_path)]

        state = AgentState(
            case_id=payload.case_id,
            source_documents=docs
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