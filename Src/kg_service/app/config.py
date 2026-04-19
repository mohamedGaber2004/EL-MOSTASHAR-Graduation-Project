from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

class KGSettings(BaseSettings):
    NEO4J_URI: str | None = None
    NEO4J_USERNAME: str | None = None
    NEO4J_PASSWORD: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

@lru_cache()
def get_settings() -> KGSettings:
    return KGSettings()