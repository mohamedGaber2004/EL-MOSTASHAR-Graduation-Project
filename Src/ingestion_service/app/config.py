from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_FILE = Path(__file__).parent / ".env"

class IngestionSettings(BaseSettings):
    DataPath: str
    na2d_data_path: str

    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        extra="ignore",
    )

@lru_cache()
def get_settings() -> IngestionSettings:
    return IngestionSettings()