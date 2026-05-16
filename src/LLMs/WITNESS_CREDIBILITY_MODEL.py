from dotenv import load_dotenv
from src.Config import get_settings
from .MODEL_BASE import BaseModel

load_dotenv()

class WitnessCredibilityModel(BaseModel):
    def __init__(self, model_name: str, temperature: float):
        super().__init__(model_name, temperature) 


def get_witness_credibility_model():
    cfg = get_settings()
    model = cfg.WITNESS_CREDIBILITY_MODEL
    temp  = cfg.WITNESS_CREDIBILITY_TEMP
    llm   = WitnessCredibilityModel(model, temp)
    return llm.get_as_llm()