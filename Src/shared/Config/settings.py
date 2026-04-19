from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

class SharedSettings(BaseSettings):
    APP_NAME: str = "EL-Mosta4ar"
    APP_VERSION: str = "0.1"
    LANGSMITH_API_KEY: str | None = None
    LANGCHAIN_API_KEY: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",  # ignore variables that belong to other services
    )

@lru_cache()
def get_shared_settings() -> SharedSettings:
    return SharedSettings()