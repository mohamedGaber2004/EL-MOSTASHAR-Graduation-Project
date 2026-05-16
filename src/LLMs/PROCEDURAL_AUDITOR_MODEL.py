from dotenv import load_dotenv
from src.Config import get_settings
from .MODEL_BASE import BaseModel

load_dotenv()

class ProceduralModel(BaseModel):
    def __init__(self, model_name: str, temperature: float):
        super().__init__(model_name, temperature) 

def get_procedural_model():
    cfg = get_settings()
    model = cfg.PROCEDURAL_AUDITOR_MODEL
    temp = cfg.PROCEDURAL_AUDITOR_TEMP
    llm = ProceduralModel(model, temp)
    return llm.get_as_llm()