from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

class OrchestratorSettings(BaseSettings):
    TAVILY_API_KEY: str | None = None
    GROQ_API_KEY: str | None = None  # for nodes that call LLM directly

    # Internal service URLs
    LLM_SERVICE_URL: str
    RETRIEVAL_SERVICE_URL: str
    KG_SERVICE_URL: str
    INGESTION_SERVICE_URL: str
    CASES_DIR: str = "../../Datasets/User_Cases"  # path to user case files, relative to CWD

    # Agent node models
    DATA_INGESTION_MODEL: str
    DATA_INGESTION_TEMP: float
    PROCEDURAL_AUDITOR_MODEL: str
    PROCEDURAL_AUDITOR_TEMP: float
    LEGAL_RESEARCHER_MODEL: str
    LEGAL_RESEARCHER_TEMP: float
    EVIDENCE_ANALYST_MODEL: str
    EVIDENCE_ANALYST_TEMP: float
    DEFENSE_ANALYST_MODEL: str
    DEFENSE_ANALYST_TEMP: float
    JUDGE_MODEL: str
    JUDGE_TEMP: float

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

@lru_cache()
def get_settings() -> OrchestratorSettings:
    return OrchestratorSettings()