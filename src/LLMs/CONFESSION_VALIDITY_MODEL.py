from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from src.Config import get_settings

load_dotenv()


class ConfessionValidityModel:
    def __init__(self, model_name: str, temperature: float):
        self.cfg = get_settings()
        self.model_name = model_name
        self.temperature = temperature
        self.as_groq_llm = ChatGroq(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=2000,
        )
        self.as_open_router_llm = ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            max_completion_tokens=2000,
            base_url=self.cfg.OPENAI_BASE_URL,
            api_key=self.cfg.OPENAI_BASE_ROUTER_API_KEY,
        )
        self.as_google_llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=self.temperature,
            google_api_key=self.cfg.GOOGLE_API_KEY,
        )

    def get_as_llm(self):
        return {
            "groq_llm":        self.as_groq_llm,
            "open_router_llm": self.as_open_router_llm,
            "google_llm":      self.as_google_llm,
        }


def get_confession_validity_model():
    cfg = get_settings()
    model = cfg.CONFESSION_VALIDITY_MODEL
    temp  = cfg.CONFESSION_VALIDITY_TEMP
    llm   = ConfessionValidityModel(model, temp)
    return llm.get_as_llm()