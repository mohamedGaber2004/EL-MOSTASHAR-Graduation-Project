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
    GOOGLE_API_KEY: Optional[str] = None
    COHERE_API_KEY: Optional[str] = None
    GROQ_API_KEY: Optional[str] = None
    HUGGINGFACE_API_TOKEN: Optional[str] = None
    LANGSMITH_API_KEY: Optional[str] = None
    LANGSMITH_TRACING: Optional[bool] = True
    LANGSMITH_ENDPOINT: Optional[str] = None
    LANGSMITH_PROJECT: Optional[str] = None
    OPENAI_BASE_URL: Optional[str] = None
    OPENAI_BASE_ROUTER_API_KEY: Optional[str] = None
    
    # Embedding Model
    EMBEDDING_MODEL: str
    ARABIC_NATIVE_EMBEDDING_MODEL: str

    # GraphRAG Model
    GRPAH_RAG_MODEL : str
    
    # Free/open LLM model config for each agent node
    DATA_INGESTION_MODEL: str 
    PROCEDURAL_AUDITOR_MODEL: str 
    LEGAL_RESEARCHER_MODEL: str 
    EVIDENCE_SCORING_MODEL: str 
    DEFENSE_AGENT_MODEL: str 
    JUDJICAL_PRINCIPLE_AGENT: str
    JUDGE_MODEL: str 

    # Optional: temperature per agent
    DATA_INGESTION_TEMP: float
    PROCEDURAL_AUDITOR_TEMP: float
    LEGAL_RESEARCHER_TEMP: float
    EVIDENCE_SCORING_TEMP: float
    DEFENSE_AGENT_TEMP: float
    JUDJICAL_PRINCIPLE_TEMP: float
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