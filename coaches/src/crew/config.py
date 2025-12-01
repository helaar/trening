
import dotenv

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

class Config(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
    
    openai_api_key: SecretStr = Field(..., description="OpenAI API key")
    crewai_storage_dir: str = Field(..., description="CrewAI storage directory")
    model_name: str = "gpt-3.5-turbo"
    coaches: str = "./coaches.yaml"
    tasks: str = "./tasks.yaml"
    athletes: str = "./athletes.yaml"
    exchange_dir: str = "../ENV/exchange"
    @property
    def workouts(self) -> str:
        return f"{self.exchange_dir}/" # TODO: May change to workouts/
    
    @property
    def output(self) -> str:
        return f"{self.exchange_dir}/output/"

config = Config() # type: ignore
dotenv.load_dotenv()