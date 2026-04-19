from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

class RetrievalSettings(BaseSettings):
    HUGGINGFACE_API_TOKEN: str | None = None
    EMBEDDING_MODEL: str
    FAISS_INDEX_PATH: str

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

@lru_cache()
def get_settings() -> RetrievalSettings:
    return RetrievalSettings()