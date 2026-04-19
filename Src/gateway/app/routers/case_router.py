from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Any
from threading import Lock
import httpx
import os

router = APIRouter(prefix="/cases", tags=["cases"])

ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://orchestrator:8000")

# In-memory cache (same as before)
_CASE_CACHE: dict = {}
_CACHE_LOCK = Lock()


class InvokeCaseRequest(BaseModel):
    case_id: str
    source_documents: Optional[List[str]] = None


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.post("/invoke_case")
async def invoke_case(payload: InvokeCaseRequest) -> Any:
    """
    Forward the case to the orchestrator service which runs the LangGraph pipeline.
    """
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{ORCHESTRATOR_URL}/pipeline/run",
                json={
                    "case_id": payload.case_id,
                    "source_documents": payload.source_documents or []
                }
            )
            response.raise_for_status()
            out = response.json()

        # Store in cache (thread-safe)
        with _CACHE_LOCK:
            _CASE_CACHE[payload.case_id] = out

        return out

    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/get_case_result")
async def get_case_result(case_id: Optional[str] = None) -> Any:
    """
    Retrieve a previously-run case result from cache.
    """
    if not case_id:
        raise HTTPException(status_code=400, detail="Missing query parameter: case_id")

    with _CACHE_LOCK:
        res = _CASE_CACHE.get(case_id)

    if res is None:
        raise HTTPException(
            status_code=404,
            detail=f"No cached result for case_id={case_id}. Run /invoke_case first."
        )

    return res