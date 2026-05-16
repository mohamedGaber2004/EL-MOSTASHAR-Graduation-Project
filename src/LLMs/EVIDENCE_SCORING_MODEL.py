from dotenv import load_dotenv
from src.Config import get_settings
from .MODEL_BASE import BaseModel

load_dotenv()

class EvidenceScoringModel(BaseModel):
    def __init__(self, model_name: str, temperature: float):
        super().__init__(model_name, temperature) 

def get_evidence_scoring_model():
    cfg = get_settings()
    model = cfg.EVIDENCE_SCORING_MODEL
    temp = cfg.EVIDENCE_SCORING_TEMP
    llm = EvidenceScoringModel(model, temp)
    return llm.get_as_llm()