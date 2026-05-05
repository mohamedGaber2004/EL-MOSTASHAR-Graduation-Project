from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from src.Graph.state import AgentState

from src.agents.data_ingestion_agent import DataIngestionAgent

# Create router
data_ingestion_router = APIRouter(prefix="/agents", tags=["Agents Pipeline"])

class InvokeDataIngestion(BaseModel):
    case_id: str
    source_documents: Optional[List[str]] = None


@data_ingestion_router.post("/ingest_data")
async def ingest_data(payload: InvokeDataIngestion):
    try:
        source_docs = payload.source_documents or []
        if not source_docs:
            import os
            case_dir = os.path.join("Datasets", "User_Cases", payload.case_id)
            if os.path.isdir(case_dir):
                source_docs = [case_dir]

        state = AgentState(case_id=payload.case_id, source_documents=source_docs)
        agent = DataIngestionAgent()
        final = agent.run(state)

        # Normalize to JSON-serializable dict
        if hasattr(final, "model_dump"):
            out = final.model_dump()
        else:
            out = final

        return out
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))