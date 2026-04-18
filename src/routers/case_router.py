from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Any
from threading import Lock

from src.Graph import run_case
from src.Graph.state import AgentState


# Create router
router = APIRouter(prefix="/cases", tags=["cases"])


# Simple in-memory cache for invoked case results (per-process)
_CASE_CACHE: dict = {}
_CACHE_LOCK = Lock()


class InvokeCaseRequest(BaseModel):
    case_id: str
    source_documents: Optional[List[str]] = None


@router.post("/invoke_case")
async def invoke_case(payload: InvokeCaseRequest) -> Any:
    """
    Run the legal pipeline for a case.

    Request body: { "case_id": "case_1", "source_documents": ["path/to/docs"] }
    Returns the final state (as JSON) produced by the graph.
    """
    try:
        state = AgentState(case_id=payload.case_id, source_documents=payload.source_documents or [])
        final = run_case(state)

        # Normalize to JSON-serializable dict
        if hasattr(final, "model_dump"):
            out = final.model_dump()
        else:
            out = final

        # Store in cache (thread-safe)
        with _CACHE_LOCK:
            _CASE_CACHE[payload.case_id] = out

        return out
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/get_case_result")
async def get_case_result(case_id: Optional[str] = None) -> Any:
    """
    Retrieve a previously-run case result.

    Note: persistence is not implemented. This endpoint returns 404 unless
    you first invoke the pipeline via `/cases/invoke_case` and wire a
    storage layer. For now it provides a helpful message.
    """
    if not case_id:
        raise HTTPException(status_code=400, detail="Missing query parameter: case_id")

    with _CACHE_LOCK:
        res = _CASE_CACHE.get(case_id)

    if res is None:
        raise HTTPException(status_code=404, detail=f"No cached result for case_id={case_id}")

    return res
