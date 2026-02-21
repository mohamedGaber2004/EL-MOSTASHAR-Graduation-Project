from langchain_groq import ChatGroq
from dotenv import load_dotenv

from src.Config.config import ASSISTANT_MODEL , ASSISTANT_MODEL_TEMP

load_dotenv()

class AssistantModel : 
    def __init__(self,model_name:str = ASSISTANT_MODEL,temprature:float=ASSISTANT_MODEL_TEMP):
        self.model_name = model_name
        self.temprature = temprature
        self.as_llm = ChatGroq(
            model=self.model_name,
            temperature=self.temprature
        )

    def get_as_llm(self):
        return self.as_llm
    

def get_llm_oss():
    llm = AssistantModel()
    my_llm = llm.get_as_llm()
    return my_llm