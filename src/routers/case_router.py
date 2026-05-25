from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import List, Optional, Any
from threading import Lock
import shutil
import os , logging

logger = logging.getLogger(__name__)

from src.Graph.graph_builder import run_case
from src.Graph.state import AgentState


# Create router
router = APIRouter(prefix="/cases", tags=["Full Case"])


# Simple in-memory cache for invoked case results (per-process)
_CASE_CACHE: dict = {}
_CACHE_LOCK = Lock()


@router.post("/invoke_case")
async def invoke_case(
    case_id: str = Form(...),
    files: List[UploadFile] = File(...),
):
    """
    Endpoint to invoke AI processing on a case by receiving files from Spring Boot.
    """
    if not case_id or not case_id.strip():
        raise HTTPException(status_code=400, detail="case_id is required")

    # Create directory
    case_dir = os.path.join("Datasets", "User_Cases", case_id.strip())
    os.makedirs(case_dir, exist_ok=True)

    saved_files_count = 0

    try:
        # Save uploaded files
        for uploaded_file in files:
            if not uploaded_file.filename:
                continue

            file_path = os.path.join(case_dir, uploaded_file.filename)

            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(uploaded_file.file, buffer)

            saved_files_count += 1

        if saved_files_count == 0:
            raise HTTPException(status_code=400, detail="No valid files were uploaded")

        # Run the AI graph
        state = AgentState(case_id=case_id, source_documents=[case_dir])

        final = run_case(state)

        # Prepare response
        if hasattr(final, "model_dump"):
            result = final.model_dump()
        elif isinstance(final, dict):
            result = final
        else:
            result = {"raw_result": str(final)}

        return {
            "success": True,
            "message": "Case processed successfully",
            "case_id": case_id,
            "files_received": saved_files_count,
            "result": result
        }

    except Exception as e:
        logger.error(f"Failed to process case {case_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/get_case_result")
async def get_case_result(case_id: Optional[str] = None) -> Any:

    if not case_id:
        raise HTTPException(status_code=400, detail="Missing query parameter: case_id")

    with _CACHE_LOCK:
        res = _CASE_CACHE.get(case_id)

    if res is None:
        raise HTTPException(status_code=404, detail=f"No cached result for case_id={case_id}")

    return res
