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
async def invoke_case(case_id: str = Form(...),files: List[UploadFile] = File(...)):
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

@router.post(
    "/invoke_case_from_device",
    openapi_extra={
        "requestBody": {
            "content": {
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "required": ["case_id", "files"],
                        "properties": {
                            "case_id": {
                                "type": "string",
                                "description": "Unique case identifier, e.g. CASE-2024-001"
                            },
                            "files": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "format": "binary"      # ← this triggers the real OS file picker
                                },
                                "description": "One or more files (.pdf .png .jpg .jpeg .docx .txt .csv) — max 20MB each"
                            }
                        }
                    }
                }
            },
            "required": True
        }
    }
)
async def invoke_case_from_device(case_id: str = Form(...),files: List[UploadFile] = File(...)):
    """
    Upload files directly from your device for AI processing.
    - Allowed: `.pdf` `.png` `.jpg` `.jpeg` `.docx` `.txt` `.csv`
    - Max **20MB** per file
    """

    ALLOWED_EXTENSIONS = {".txt"}
    MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024  # 20MB

    if not case_id or not case_id.strip():
        raise HTTPException(status_code=400, detail="case_id is required")

    case_dir = os.path.join("Datasets", "User_Cases", case_id.strip())
    os.makedirs(case_dir, exist_ok=True)

    saved_files: List[str] = []
    rejected_files: List[dict] = []

    try:
        for uploaded_file in files:
            filename = uploaded_file.filename

            # --- Validation ---
            if not filename:
                rejected_files.append({"file": "(unnamed)", "reason": "Missing filename"})
                continue

            ext = os.path.splitext(filename)[-1].lower()
            if ext not in ALLOWED_EXTENSIONS:
                rejected_files.append({"file": filename, "reason": f"Extension '{ext}' not allowed"})
                continue

            file_bytes = await uploaded_file.read()

            if len(file_bytes) == 0:
                rejected_files.append({"file": filename, "reason": "File is empty"})
                continue

            if len(file_bytes) > MAX_FILE_SIZE_BYTES:
                rejected_files.append({"file": filename, "reason": "Exceeds 20MB size limit"})
                continue

            # --- Save file ---
            # Sanitize filename to prevent path traversal attacks
            safe_filename = os.path.basename(filename)
            file_path = os.path.join(case_dir, safe_filename)

            with open(file_path, "wb") as buffer:
                buffer.write(file_bytes)

            saved_files.append(safe_filename)

        if not saved_files:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "No valid files were saved",
                    "rejected_files": rejected_files,
                },
            )

        # --- Run AI graph ---
        state = AgentState(case_id=case_id, source_documents=[case_dir])
        final = run_case(state)

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
            "files_saved": saved_files,
            "files_rejected": rejected_files,
            "result": result,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to process device-uploaded case {case_id}: {str(e)}", exc_info=True)
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