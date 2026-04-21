from langchain_groq import ChatGroq
from dotenv import load_dotenv
from src.Config import get_settings

load_dotenv()

class ArabicModel: 
    def __init__(self, model_name: str, temperature: float):
        self.model_name = model_name
        self.temperature = temperature
        self.ar_llm = ChatGroq(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=2000
        )

    def get_ar_llm(self):
        return self.ar_llm

# New: config-driven getter
def get_llm_qwn(model_name: str = None, temperature: float = None):
    cfg = get_settings()
    model = model_name or getattr(cfg, "DATA_INGESTION_MODEL", "Qwen/Qwen3-32B")
    temp = temperature if temperature is not None else getattr(cfg, "DATA_INGESTION_TEMP", 0.0)
    llm = ArabicModel(model, temp)
    return llm.get_ar_llm()