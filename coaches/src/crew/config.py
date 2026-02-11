
import dotenv
import yaml
from pathlib import Path
from typing import Any

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
        self._app_settings = self._load_app_settings()
        self._athlete_settings = self._load_athlete_settings()
    
    def _load_app_settings(self) -> dict[str, Any]:
        """Load application settings from app-settings.yaml."""
        app_settings_path = Path("../app-settings.yaml")
        if app_settings_path.exists():
            with app_settings_path.open("r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}
    
    def _load_athlete_settings(self) -> dict[str, Any]:
        """Load athlete settings from athlete-settings.yaml."""
        athlete_settings_path = Path("../athlete-settings.yaml")
        if athlete_settings_path.exists():
            with athlete_settings_path.open("r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}
    
    def get_athlete_settings(self, athlete: str) -> dict[str, Any]:
        """Get settings for a specific athlete."""
        athletes = self._athlete_settings.get("athletes", {})
        return athletes.get(athlete.lower(), {})
    
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
    
    # Configuration file paths
    coaches: str = "./coaches.yaml"
    tasks: str = "./tasks.yaml"
    athletes: str = "./athletes.yaml"
    athlete_settings: str = "../athlete-settings.yaml"
    knowledge: str = "./knowledge.yaml"
    plans: str = "./plans.yaml"
    
    @property
    def base_output_dir(self) -> str:
        """Get the base output directory from app settings."""
        app_config = self._app_settings.get("application", {})
        output_dir_str = app_config.get("output-dir", "./ENV/exchange/athletes")
        
        # Resolve output path relative to parent directory (project root)
        base_output_dir = (Path("..") / output_dir_str).resolve()
        return str(base_output_dir)
    
    def get_athlete_base_dir(self, athlete: str) -> Path:
        """Get the base directory for a specific athlete."""
        return Path(self.base_output_dir) / athlete.lower()
    
    def get_athlete_analyses_dir(self, athlete: str) -> Path:
        """Get the analyses directory for a specific athlete."""
        return self.get_athlete_base_dir(athlete) / "analyses"
    
    def get_athlete_comments_dir(self, athlete: str) -> Path:
        """Get the comments directory for a specific athlete."""
        return self.get_athlete_base_dir(athlete) / "comments"
    
    def get_athlete_self_assessments_dir(self, athlete: str) -> Path:
        """Get the self-assessments directory for a specific athlete."""
        return self.get_athlete_base_dir(athlete) / "self-assessments"
    
    def get_athlete_daily_dir(self, athlete: str) -> Path:
        """Get the daily analysis directory for a specific athlete."""
        return self.get_athlete_base_dir(athlete) / "daily"
    
    def get_athlete_load_dir(self, athlete: str) -> Path:
        """Get the load directory for a specific athlete."""
        return self.get_athlete_base_dir(athlete) / "load"
    
    def get_athlete_long_term_dir(self, athlete: str) -> Path:
        """Get the long term analysis directory for a specific athlete."""
        return self.get_athlete_base_dir(athlete) / "long_term"
    
    def get_athlete_planning_dir(self, athlete: str) -> Path:
        """Get the planning directory for a specific athlete."""
        return self.get_athlete_base_dir(athlete) / "planning"
    
    @property
    def exchange_dir(self) -> str:
        """Legacy property for backward compatibility."""
        return self.base_output_dir
    
    def workouts(self, athlete: str) -> str:
        """Get the workouts directory for a specific athlete (analyses directory)."""
        return str(self.get_athlete_analyses_dir(athlete))
    
    def output(self, athlete: str) -> str:
        """Get the output directory for a specific athlete (daily directory)."""
        return str(self.get_athlete_daily_dir(athlete))

config = Config() # type: ignore
dotenv.load_dotenv()