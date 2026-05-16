from dotenv import load_dotenv
from src.Config import get_settings
from .MODEL_BASE import BaseModel

load_dotenv()

class JudgeModel(BaseModel):
    def __init__(self, model_name: str, temperature: float):
        super().__init__(model_name, temperature) 

def get_judge_model():
    cfg = get_settings()
    model = cfg.JUDGE_MODEL
    temp = cfg.JUDGE_TEMP
    llm = JudgeModel(model, temp)
    return llm.get_as_llm()