from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Any
from threading import Lock
import os , logging

logger = logging.getLogger(__name__)

from src.Graph.graph_builder import run_case
from src.Graph.states_and_schemas.state import AgentState


# Create router
router = APIRouter(prefix="/cases", tags=["Full Case"])


# Simple in-memory cache for invoked case results (per-process)
_CASE_CACHE: dict = {}
_CACHE_LOCK = Lock()


class InvokeCaseRequest(BaseModel):
    case_id:          str
    source_documents: Optional[List[str]] = None


@router.post("/invoke_case")
async def invoke_case(payload: InvokeCaseRequest) -> Any:
    source_docs = payload.source_documents or []
    if not source_docs:
        case_dir = os.path.join("Datasets", "User_Cases", payload.case_id)
        if os.path.isdir(case_dir):
            source_docs = [case_dir]

    state = AgentState(case_id=payload.case_id, source_documents=source_docs)

    try:
        final = run_case(state)
    except Exception as e:
        # str(e) can be empty for some exception types — always include the class name
        detail = f"{type(e).__name__}: {str(e)}" if str(e) else type(e).__name__
        logger.error("invoke_case failed for case_id=%s: %s", payload.case_id, detail, exc_info=True)
        raise HTTPException(status_code=500, detail=detail)

    # Normalize to JSON-serializable dict
    if hasattr(final, "model_dump"):
        out = final.model_dump()
    elif isinstance(final, dict):
        out = final
    else:
        out = {"error": "unexpected output type", "type": str(type(final))}

    with _CACHE_LOCK:
        _CASE_CACHE[payload.case_id] = out

    return out


@router.get("/get_case_result")
async def get_case_result(case_id: Optional[str] = None) -> Any:

    if not case_id:
        raise HTTPException(status_code=400, detail="Missing query parameter: case_id")

    with _CACHE_LOCK:
        res = _CASE_CACHE.get(case_id)

    if res is None:
        raise HTTPException(status_code=404, detail=f"No cached result for case_id={case_id}")

    return res
