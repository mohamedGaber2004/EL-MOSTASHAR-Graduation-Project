from langchain_groq import ChatGroq
from dotenv import load_dotenv

from src.Config import get_settings

load_dotenv()

class ReasoningModel : 
    def __init__(self,model_name:str = get_settings().REASONING_MODEL,temprature:float=get_settings().REASONING_MODEL_TEMP):
        self.model_name = model_name
        self.temprature = temprature
        self.re_llm = ChatGroq(
            model=self.model_name,
            temperature=self.temprature,
            max_tokens=2000
        )

    def get_re_llm(self):
        return self.re_llm
    

def get_llm_llama():
    llm = ReasoningModel()
    my_llm = llm.get_re_llm()
    return my_llm