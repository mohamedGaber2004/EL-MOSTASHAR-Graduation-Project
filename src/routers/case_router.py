from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from src.Graph import create_sample_case

# Create router
router = APIRouter(prefix="/cases", tags=["cases"])


@router.post("/upload_case")
async def create_case():
    """
    upload case to "Datasets/User_Cases"
    """
    pass


@router.post("/invoke_case")
async def create_case():
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