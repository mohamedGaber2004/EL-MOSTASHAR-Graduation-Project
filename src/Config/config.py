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
    DATA_INGESTION_TEMP: float
    DATA_INGESTION_PROVIDER: str = "open_router"

    PROCEDURAL_AUDITOR_MODEL: str 
    PROCEDURAL_AUDITOR_TEMP: float
    PROCEDURAL_AUDITOR_PROVIDER: str = "open_router"

    LEGAL_RESEARCHER_MODEL: str 
    LEGAL_RESEARCHER_TEMP: float
    LEGAL_RESEARCHER_PROVIDER: str = "open_router"

    EVIDENCE_SCORING_MODEL: str 
    EVIDENCE_SCORING_TEMP: float
    EVIDENCE_SCORING_PROVIDER: str = "open_router"

    DEFENSE_AGENT_MODEL: str 
    DEFENSE_AGENT_TEMP: float
    DEFENSE_AGENT_PROVIDER: str = "open_router"

    JUDJICAL_PRINCIPLE_AGENT: str
    JUDJICAL_PRINCIPLE_TEMP: float
    JUDJICAL_PRINCIPLE_PROVIDER: str = "open_router"

    CONFESSION_VALIDITY_MODEL: str
    CONFESSION_VALIDITY_TEMP: float
    CONFESSION_VALIDITY_PROVIDER: str = "open_router"

    WITNESS_CREDIBILITY_MODEL: str
    WITNESS_CREDIBILITY_TEMP: float
    WITNESS_CREDIBILITY_PROVIDER: str = "open_router"

    PROSECUTION_ANALYST_MODEL: str
    PROSECUTION_ANALYST_TEMP: float
    PROSECUTION_ANALYST_PROVIDER: str = "open_router"

    SENTENCING_MODEL: str
    SENTENCING_TEMP: float
    SENTENCING_PROVIDER: str = "open_router"

    JUDGE_MODEL: str
    JUDGE_TEMP: float
    JUDGE_PROVIDER: str = "open_router"

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