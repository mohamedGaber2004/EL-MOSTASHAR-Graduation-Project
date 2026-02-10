from langchain_groq import ChatGroq
from dotenv import load_dotenv

from Config.config import ARABIC_MODEL
load_dotenv()

class ArabicModel: 
    def __init__(self, model_name: str = ARABIC_MODEL, temperature: int = 0):
        self.model_name = model_name
        self.temperature = temperature
        self.ar_llm = ChatGroq(
            model=self.model_name,
            temperature=self.temperature
        )

    def get_ar_llm(self):
        return self.ar_llm