from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
from src.Config import get_settings

load_dotenv()

class BaseModel : 
    def __init__(self, model_name: str, temperature: float):
        self.cfg = get_settings()
        self.model_name = model_name
        self.temperature = temperature
        self.as_groq_llm = ChatGroq(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=2048
        )
        self.as_open_router_llm = ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            max_completion_tokens=2048,
            base_url=self.cfg.OPENAI_BASE_URL,
            api_key=self.cfg.OPENAI_BASE_ROUTER_API_KEY
        )
        self.as_mistral_llm = ChatMistralAI(
            model=self.model_name,
            temperature=self.temperature,
            api_key=self.cfg.MISTRAL_API_KEY,
            max_tokens=2048
        )

    def get_as_llm(self):
        return {
            "groq_llm":self.as_groq_llm,
            "open_router_llm": self.as_open_router_llm,
            "mistral_llm": self.as_mistral_llm
        }