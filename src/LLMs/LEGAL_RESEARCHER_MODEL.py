from dotenv import load_dotenv
from src.Config import get_settings
from .MODEL_BASE import BaseModel

load_dotenv()

class LegalResearcherModel(BaseModel):
    def __init__(self, model_name: str, temperature: float):
        super().__init__(model_name, temperature) 

def get_legal_reasearcher_model():
    cfg = get_settings()
    model = cfg.LEGAL_RESEARCHER_MODEL
    temp = cfg.LEGAL_RESEARCHER_TEMP
    llm = LegalResearcherModel(model, temp)
    return llm.get_as_llm()