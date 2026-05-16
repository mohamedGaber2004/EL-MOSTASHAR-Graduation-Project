from dotenv import load_dotenv
from src.Config import get_settings
from .MODEL_BASE import BaseModel

load_dotenv()

class DataIngestionModel(BaseModel):
    def __init__(self, model_name: str, temperature: float):
        super().__init__(model_name, temperature) 

def get_ingesion_model():
    cfg = get_settings()
    model = cfg.DATA_INGESTION_MODEL
    temp = cfg.DATA_INGESTION_TEMP
    llm = DataIngestionModel(model, temp)
    return llm.get_as_llm()