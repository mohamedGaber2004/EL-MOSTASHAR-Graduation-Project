from langchain_groq import ChatGroq
from dotenv import load_dotenv

from src.Config import get_settings

load_dotenv()

class AssistantModel : 
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

# New: config-driven getter
def get_llm_oss(model_name: str = None, temperature: float = None):
    cfg = get_settings()
    model = model_name or getattr(cfg, "EVIDENCE_ANALYST_MODEL", "openai/gpt-oss-120b")
    temp = temperature if temperature is not None else getattr(cfg, "EVIDENCE_ANALYST_TEMP", 0.0)
    llm = AssistantModel(model, temp)
    return llm.get_as_llm()