from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from src.Graph import create_sample_case

# Create router
router = APIRouter(prefix="/cases", tags=["cases"])

# Simple Case model for API
class Case(BaseModel):
    case_id: str
    title: str
    description: str
    defendants: List[str]
    charges: List[str]
    status: str
    created_at: datetime
    verdict: Optional[str] = None
    confidence_score: Optional[float] = None
    reasoning: Optional[str] = None

def agent_state_to_case(state) -> Case:
    """Convert AgentState to Case model"""
    # Extract reasoning from agent_outputs or reasoning_trace
    reasoning = None
    if state.reasoning_trace and len(state.reasoning_trace) > 0:
        reasoning = " ".join([item.get("reasoning", "") for item in state.reasoning_trace if isinstance(item, dict)])
    elif state.agent_outputs:
        # Try to get reasoning from agent outputs
        for agent_output in state.agent_outputs.values():
            if isinstance(agent_output, dict) and "reasoning" in agent_output:
                reasoning = agent_output["reasoning"]
                break
    
    return Case(
        case_id=state.case_id or "UNKNOWN",
        title=f"Case {state.case_number or state.case_id}",
        description=f"Legal case with {len(state.defendants)} defendants and {len(state.charges)} charges",
        defendants=[d.name for d in state.defendants] if state.defendants else [],
        charges=[c.statute for c in state.charges] if state.charges else [],
        status="Active",
        created_at=datetime.now(),
        verdict=state.suggested_verdict,
        confidence_score=state.confidence_score,
        reasoning=reasoning
    )

@router.post("/sample", response_model=Case)
async def create_case():
    """
    Create and return a sample case for testing.
    """
    sample_state = create_sample_case()
    return agent_state_to_case(sample_state)

@router.get("/sample", response_model=Case)
async def get_sample_case():
    """
    Get the sample case with analysis results.
    """
    sample_state = create_sample_case()
    case = agent_state_to_case(sample_state)
    
    # Add sample analysis results for demonstration
    case.verdict = "Guilty"
    case.confidence_score = 0.85
    case.reasoning = "Based on the evidence analysis, the defendant is found guilty of the charge of assault resulting in death. The prosecution has proven all required elements of the crime beyond reasonable doubt, including the act of assault, the resulting death, the causal relationship, and criminal intent. The defendant's confession and witness statements corroborate the physical evidence found at the scene."
    
    return case