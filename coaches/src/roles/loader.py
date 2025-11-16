"""Coach loader utility for parsing YAML and creating CrewAI agents."""

import yaml
from pathlib import Path
from crewai import Agent
from .models import CoachConfig
from .config import Config


class CoachLoader:
    """Utility class for loading coach definitions from YAML and creating agents."""
    
    def __init__(self, config: Config):
        """Initialize the loader with configuration."""
        self.config = config
    
    def load_coaches_config(self, yaml_file_path: str | Path) -> CoachConfig:
        """Load and parse the coaches YAML configuration file."""
        yaml_path = Path(yaml_file_path)
        
        if not yaml_path.exists():
            raise FileNotFoundError(f"Coaches config file not found: {yaml_path}")
        
        with open(yaml_path, 'r', encoding='utf-8') as file:
            yaml_data = yaml.safe_load(file)
        
        return CoachConfig(**yaml_data)
    
    def create_agent_from_coach(self, coach_name: str, coaches_config: CoachConfig, tools: list | None = None) -> Agent:
        """Create a CrewAI Agent from a coach definition."""
        try:
            # Extract CrewAI parameters
            agent_params = coaches_config.extract_crewai_params(coach_name)
            
            # Create and return the agent
            return Agent(
                role=agent_params["role"],
                goal=agent_params["goal"],
                backstory=agent_params["backstory"],
                verbose=True,
                allow_delegation=False,
                llm=self.config.model_name,
                tools=tools or []
            )
            
        except ValueError as e:
            raise ValueError(f"Failed to create agent for coach '{coach_name}': {e}")
    
    def load_agent_from_file(self, yaml_file_path: str | Path, coach_name: str, tools: list | None = None) -> Agent:
        """Load a specific coach agent directly from YAML file."""
        coaches_config = self.load_coaches_config(yaml_file_path)
        return self.create_agent_from_coach(coach_name, coaches_config, tools)
    
    def list_available_coaches(self, yaml_file_path: str | Path) -> list[str]:
        """List all available coach names in the YAML file."""
        coaches_config = self.load_coaches_config(yaml_file_path)
        return coaches_config.list_coaches()