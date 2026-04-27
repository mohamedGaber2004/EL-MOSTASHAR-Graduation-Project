from typing import Optional
from functools import lru_cache
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # APP 
    APP_NAME: str 
    APP_VERSION: str 

    # Data Paths
    DataPath: str 
    na2d_data_path: str 
    FAISS_INDEX_PATH: str 

    # API Keys
    GROQ_API_KEY: Optional[str] = None
    HUGGINGFACE_API_TOKEN: Optional[str] = None
    LANGCHAIN_API_KEY: Optional[str] = None

    # LLMs (raw model config for legacy code)
    REASONING_MODEL: str
    REASONING_MODEL_TEMP: float
    ARABIC_MODEL: str
    ARABIC_MODEL_TEMP: float
    ASSISTANT_MODEL: str
    ASSISTANT_MODEL_TEMP: float
    EMBEDDING_MODEL : str
    
    # Free/open LLM model config for each agent node
    DATA_INGESTION_MODEL: str 
    PROCEDURAL_AUDITOR_MODEL: str 
    LEGAL_RESEARCHER_MODEL: str 
    EVIDENCE_ANALYST_MODEL: str 
    DEFENSE_ANALYST_MODEL: str 
    JUDGE_MODEL: str 

    # Optional: temperature per node
    DATA_INGESTION_TEMP: float
    PROCEDURAL_AUDITOR_TEMP: float
    LEGAL_RESEARCHER_TEMP: float
    EVIDENCE_ANALYST_TEMP: float
    DEFENSE_ANALYST_TEMP: float
    JUDGE_TEMP: float
    
    # Neo4j KG
    NEO4J_URI: Optional[str] = None
    NEO4J_USERNAME: Optional[str] = None
    NEO4J_PASSWORD: Optional[str] = None

    # Agents
    INVOKATION_MAX_RETRIES: int 
    INTER_AGENTS_DELAY: float
    INGESTION_AGENT_MAX_CHARS: int


    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()