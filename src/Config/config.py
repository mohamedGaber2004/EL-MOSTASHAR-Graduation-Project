from pydantic_settings import BaseSettings , SettingsConfigDict



class Settings(BaseSettings):
    APP_NAME : str
    APP_VERSION : str
    GROQ_API_KEY : str
    TAVILY_API_KEY : str
    LANGSMITH_API_KEY : str
    HUGGINGFACE_API_TOKEN : str
    LANGCHAIN_API_KEY : str
    DataPath : str
    na2d_data_path : str
    REASONING_MODEL : str
    REASONING_MODEL_TEMP : float
    ARABIC_MODEL : str
    ARABIC_MODEL_TEMP : float
    ASSISTANT_MODEL : str
    ASSISTANT_MODEL_TEMP : float
    EMBEDDING_MODEL : str
    NEO4J_URI : str
    NEO4J_USERNAME : str
    NEO4J_PASSWORD : str

    class Config:
        env_file = ".env"


def get_settings():
    return Settings()