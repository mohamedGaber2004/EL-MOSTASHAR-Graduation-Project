from langchain_groq import ChatGroq
import sys
import os

# This adds the root "Graduation Project" directory to your path
root_path = os.path.abspath(os.path.join(os.getcwd(), "../../"))
if root_path not in sys.path:
    sys.path.append(root_path)

from Config.config import ASSISTANT_MODEL

class ArabicModel : 
    def __init__(self,model_name:str = ASSISTANT_MODEL,temprature:float=0):
        self.model_name = model_name
        self.temprature = temprature
        self.as_llm = ChatGroq(model=self.model_name,temperature=self.temprature)