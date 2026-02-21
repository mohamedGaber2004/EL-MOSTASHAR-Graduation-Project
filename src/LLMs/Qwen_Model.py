from langchain_groq import ChatGroq
from dotenv import load_dotenv

from src.Config.config import ARABIC_MODEL , ARABIC_MODEL_TEMP

load_dotenv()

class ArabicModel: 
    def __init__(self, model_name: str = ARABIC_MODEL, temperature: float = ARABIC_MODEL_TEMP):
        self.model_name = model_name
        self.temperature = temperature
        self.ar_llm = ChatGroq(
            model=self.model_name,
            temperature=self.temperature
        )

    def get_ar_llm(self):
        return self.ar_llm
    

def get_llm_qwn():
    llm = ArabicModel()
    my_llm = llm.get_ar_llm()
    return my_llm