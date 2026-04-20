from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

load_dotenv() 

ENV_FILE = Path(__file__).parent / ".env"

class IngestionSettings(BaseSettings):
    DataPath: str = None 
    na2d_data_path: str

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

@lru_cache()
def get_settings() -> IngestionSettings:
    return IngestionSettings()