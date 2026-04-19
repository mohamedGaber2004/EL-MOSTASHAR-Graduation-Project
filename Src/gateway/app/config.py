from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

class GatewaySettings(BaseSettings):
    ORCHESTRATOR_URL: str
    INGESTION_SERVICE_URL: str

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

@lru_cache()
def get_settings() -> GatewaySettings:
    return GatewaySettings()