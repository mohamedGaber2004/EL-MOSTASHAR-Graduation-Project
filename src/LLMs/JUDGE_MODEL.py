from langchain_groq import ChatGroq
from dotenv import load_dotenv
from src.Config import get_settings

load_dotenv()

class JudgeModel : 
    def __init__(self, model_name: str, temperature: float):
        self.model_name = model_name
        self.temperature = temperature
        self.as_llm = ChatGroq(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=2000
        )

    def get_as_llm(self):
        return self.as_llm

def get_judge_model():
    cfg = get_settings()
    model = cfg.JUDGE_MODEL
    temp = cfg.JUDGE_TEMP
    llm = JudgeModel(model, temp)
    return llm.get_as_llm()