from pydantic_settings import BaseSettings, SettingsConfigDict

class Config(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
    
    openai_api_key: str = ""  # Default empty, will be loaded from environment
    model_name: str = "gpt-3.5-turbo"
