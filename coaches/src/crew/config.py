
import dotenv

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

class Config(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
    
    openai_api_key: SecretStr = Field(..., description="OpenAI API key")
    anthropic_api_key: SecretStr = Field(..., description="Anthropic API key")
    crewai_storage_dir: str = Field(..., description="CrewAI storage directory")
    crewai_tracing_enabled: bool = Field(..., description="Enable CrewAI tracing")
    
    # LLM Model Configuration
    dev_llm_model: str = Field(default="gpt-5-mini", description="Development LLM model")
    prod_llm_model: str = Field(default="claude-sonnet-4-5", description="Production LLM model")
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._current_mode = "dev"
    
    def set_mode(self, mode: str) -> None:
        """Set the current LLM mode.
        
        Args:
            mode: Either 'dev' for development or 'prod' for production
        """
        if mode.lower() not in ["dev", "prod"]:
            raise ValueError(f"Invalid mode '{mode}'. Must be 'dev' or 'prod'.")
        self._current_mode = mode.lower()
    
    def get_model(self, mode: str | None = None) -> str:
        """Get the LLM model based on the specified or current mode.
        
        Args:
            mode: Either 'dev' for development or 'prod' for production.
                 If None, uses the current mode set by set_mode()
            
        Returns:
            The model name to use
        """
        effective_mode = mode or self._current_mode
        if effective_mode.lower() == "prod":
            return self.prod_llm_model
        return self.dev_llm_model
    coaches: str = "./coaches.yaml"
    tasks: str = "./tasks.yaml"
    athletes: str = "./athletes.yaml"
    knowledge: str = "./knowledge.yaml"
    plans: str = "./plans.yaml"
    exchange_dir: str = "../ENV/exchange"
    @property
    def workouts(self) -> str:
        return f"{self.exchange_dir}/" # TODO: May change to workouts/
    
    @property
    def output(self) -> str:
        return f"{self.exchange_dir}/output/"

config = Config() # type: ignore
dotenv.load_dotenv()