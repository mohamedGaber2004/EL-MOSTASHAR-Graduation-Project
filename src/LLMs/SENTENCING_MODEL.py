from dotenv import load_dotenv
from src.Config import get_settings
from .MODEL_BASE import BaseModel

load_dotenv()

class SentencingModel(BaseModel):
    def __init__(self, model_name: str, temperature: float):
        super().__init__(model_name, temperature) 


def get_sentencing_model():
    cfg = get_settings()
    model = cfg.SENTENCING_MODEL
    temp  = cfg.SENTENCING_TEMP
    llm   = SentencingModel(model, temp)
    return llm.get_as_llm()