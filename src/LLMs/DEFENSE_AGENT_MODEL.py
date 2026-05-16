from dotenv import load_dotenv
from src.Config import get_settings
from .MODEL_BASE import BaseModel

load_dotenv()

class DefenseAgentModel(BaseModel):
    def __init__(self, model_name: str, temperature: float):
        super().__init__(model_name, temperature) 

def get_defense_model():
    cfg = get_settings()
    model = cfg.DEFENSE_AGENT_MODEL
    temp = cfg.DEFENSE_AGENT_TEMP
    llm = DefenseAgentModel(model, temp)
    return llm.get_as_llm()