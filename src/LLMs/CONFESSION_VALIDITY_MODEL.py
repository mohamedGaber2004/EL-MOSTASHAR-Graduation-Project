from dotenv import load_dotenv
from src.Config import get_settings
from .MODEL_BASE import BaseModel

load_dotenv()

class ConfessionValidityModel(BaseModel):
    def __init__(self, model_name: str, temperature: float):
        super().__init__(model_name, temperature) 


def get_confession_validity_model():
    cfg = get_settings()
    model = cfg.CONFESSION_VALIDITY_MODEL
    temp  = cfg.CONFESSION_VALIDITY_TEMP
    llm   = ConfessionValidityModel(model, temp)
    return llm.get_as_llm()