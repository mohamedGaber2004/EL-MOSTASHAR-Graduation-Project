from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

class LLMSettings(BaseSettings):
    GROQ_API_KEY: str | None = None
    HUGGINGFACE_API_TOKEN: str | None = None

    # Model names + temps
    REASONING_MODEL: str
    REASONING_MODEL_TEMP: float
    ARABIC_MODEL: str
    ARABIC_MODEL_TEMP: float
    ASSISTANT_MODEL: str
    ASSISTANT_MODEL_TEMP: float

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

@lru_cache()
def get_settings() -> LLMSettings:
    return LLMSettings()