from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from src.Graph import run_case

# Create router
router = APIRouter(prefix="/cases", tags=["cases"])



@router.post("/invoke_case")
async def invoke_case():
    """
    invoke case by case id.
    invoke graph that takes state : AgentState()
    """
    pass


@router.get("/get_case_result")
async def get_sample_case():
    """
    Get the case by case id
    """
    pass