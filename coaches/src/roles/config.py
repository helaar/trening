
import dotenv

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

class Config(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
    
    openai_api_key: SecretStr = Field(..., description="OpenAI API key")
    model_name: str = "gpt-3.5-turbo"

config = Config() # type: ignore
dotenv.load_dotenv()