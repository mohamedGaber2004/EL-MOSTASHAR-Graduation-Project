from langchain_groq import ChatGroq
from dotenv import load_dotenv
from src.Config import get_settings

load_dotenv()

class ReasoningModel : 
    def __init__(self, model_name: str, temperature: float):
        self.model_name = model_name
        self.temperature = temperature
        self.re_llm = ChatGroq(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=2000
        )

    def get_re_llm(self):
        return self.re_llm

# New: config-driven getter
def get_llm_llama(model_name: str = None, temperature: float = None):
    cfg = get_settings()
    model = model_name or getattr(cfg, "JUDGE_MODEL", "llama-3.3-70b-versatile")
    temp = temperature if temperature is not None else getattr(cfg, "JUDGE_TEMP", 0.0)
    llm = ReasoningModel(model, temp)
    return llm.get_re_llm()