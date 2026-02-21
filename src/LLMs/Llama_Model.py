from langchain_groq import ChatGroq
from dotenv import load_dotenv

from src.Config.config import REASONING_MODEL , REASONING_MODEL_TEMP

load_dotenv()

class ReasoningModel : 
    def __init__(self,model_name:str = REASONING_MODEL,temprature:float=REASONING_MODEL_TEMP):
        self.model_name = model_name
        self.temprature = temprature
        self.re_llm = ChatGroq(
            model=self.model_name,
            temperature=self.temprature
        )

    def get_re_llm(self):
        return self.re_llm
    

def get_llm_llama():
    llm = ReasoningModel()
    my_llm = llm.get_re_llm()
    return my_llm