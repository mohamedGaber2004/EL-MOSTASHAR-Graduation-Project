from typing import Optional
from functools import lru_cache
from pydantic_settings import BaseSettings , SettingsConfigDict

class Settings(BaseSettings):
    # API Keys
    GROQ_API_KEY: Optional[str] = None
    TAVILY_API_KEY: Optional[str] = None
    LANGSMITH_API_KEY: Optional[str] = None

    # LLMs (raw model config for legacy code)
    REASONING_MODEL: str = "llama-3.3-70b-versatile"
    REASONING_MODEL_TEMP: float = 0.0
    ARABIC_MODEL: str = "Qwen/Qwen3-32B"
    ARABIC_MODEL_TEMP: float = 0.0
    ASSISTANT_MODEL: str = "openai/gpt-oss-120b"
    ASSISTANT_MODEL_TEMP: float = 0.0
    APP_NAME : str = "EL-Mosta4ar"
    APP_VERSION : str = "0.1"
    HUGGINGFACE_API_TOKEN : Optional[str] = None
    LANGCHAIN_API_KEY : Optional[str] = None
    DataPath : str = "Datasets/Unstructured_Data"
    na2d_data_path : str = "Datasets/Unstructured_Data/na2d"
    # Backwards-compatible alias expected by some modules
    FAISS_INDEX_PATH : str = "legal_faiss_index"
    EMBEDDING_MODEL : str = "silma-ai/silma-embeddding-sts-v0.1"
    NEO4J_URI : Optional[str] = None
    NEO4J_USERNAME : Optional[str] = None
    NEO4J_PASSWORD : Optional[str] = None

    # Free/open LLM model config for each agent node
    DATA_INGESTION_MODEL: str = "openai/gpt-oss-120b"
    PROCEDURAL_AUDITOR_MODEL: str = "llama-3.3-70b-versatile"
    LEGAL_RESEARCHER_MODEL: str = "Qwen/Qwen3-32B"
    EVIDENCE_ANALYST_MODEL: str = "openai/gpt-oss-120b"
    DEFENSE_ANALYST_MODEL: str = "Qwen/Qwen3-32B"
    JUDGE_MODEL: str = "llama-3.3-70b-versatile"

    # Optional: temperature per node
    DATA_INGESTION_TEMP: float = 0.0
    PROCEDURAL_AUDITOR_TEMP: float = 0.0
    LEGAL_RESEARCHER_TEMP: float = 0.0
    EVIDENCE_ANALYST_TEMP: float = 0.0
    DEFENSE_ANALYST_TEMP: float = 0.0
    JUDGE_TEMP: float = 0.0

    class Config:
        env_file = ".env"

    # Use `na2d_data_path` directly; do not shadow with property to avoid BaseSettings conflicts.


@lru_cache()
def get_settings():
    return Settings()