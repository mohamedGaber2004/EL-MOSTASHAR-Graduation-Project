from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

load_dotenv() 

ENV_FILE = Path(__file__).parent / ".env"
class KGSettings(BaseSettings):
    NEO4J_URI: str | None = None
    NEO4J_USERNAME: str | None = None
    NEO4J_PASSWORD: str | None = None
    INGESTION_SERVICE_URL: str = "http://127.0.0.1:8004"

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

@lru_cache()
def get_settings() -> KGSettings:
    return KGSettings()