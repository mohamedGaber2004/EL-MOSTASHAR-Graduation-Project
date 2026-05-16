from dotenv import load_dotenv
from src.Config import get_settings
from .MODEL_BASE import BaseModel

load_dotenv()

class ProsecutionAnalystModel(BaseModel):
    def __init__(self, model_name: str, temperature: float):
        super().__init__(model_name, temperature) 


def get_prosecution_analyst_model():
    cfg = get_settings()
    model = cfg.PROSECUTION_ANALYST_MODEL
    temp  = cfg.PROSECUTION_ANALYST_TEMP
    llm   = ProsecutionAnalystModel(model, temp)
    return llm.get_as_llm()