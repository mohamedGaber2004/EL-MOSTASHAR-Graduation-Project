from dotenv import load_dotenv
from src.Config import get_settings
from .MODEL_BASE import BaseModel

load_dotenv()

class JudgeModel(BaseModel):
    def __init__(self, model_name: str, temperature: float):
        super().__init__(model_name, temperature) 

    self.as_open_router_llm = ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=4096,
            model_kwargs={
            "extra_body": {
                "reasoning": {"max_tokens": 1024}
                }
            },
            base_url=self.cfg.OPENAI_BASE_URL,
            api_key=self.cfg.OPENAI_BASE_ROUTER_API_KEY
        )
    
def get_judge_model():
    cfg = get_settings()
    model = cfg.JUDGE_MODEL
    temp = cfg.JUDGE_TEMP
    llm = JudgeModel(model, temp)
    return llm.get_as_llm()
