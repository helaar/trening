"""Coach loader utility for parsing YAML and creating CrewAI agents."""

from pydantic import BaseModel
import yaml
from pathlib import Path
from crewai import Agent, Task
from .models import Coach, TaskDescription
from .config import Config


class YamlLoader[T: BaseModel]:
    """Abstract base class for loading configuration from YAML files."""
    def __init__(self, yaml_path: Path, model_class: type[T]) -> None:
        self.config = self._load_yaml(yaml_path)
        self.model_class = model_class

    def _load_yaml(self, yaml_path: Path) -> dict[str, object]:
        with yaml_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data
    
    def find(self, key: str) -> T | None:
        maybe_dict = self.config.get(key)
        if maybe_dict and isinstance(maybe_dict, dict) :
            return self.model_class(**maybe_dict) 
        return None 


class CoachLoader(YamlLoader[Coach]):
    """Utility class for loading coach definitions from YAML and creating agents."""
    
    def __init__(self, config: Config):
        super().__init__(Path(config.coaches), Coach)
        self.model_name = config.model_name

    def create_coach_agent(self, coach_name: str, tools: list | None = None) -> Agent:
        """Create a CrewAI Agent from a coach definition."""
        try:
            # Extract CrewAI parameters
            coach = self.find(coach_name)
            if not coach:
                raise ValueError(f"Coach '{coach_name}' not found in configuration.")
            
            # Create and return the agent
            return Agent(
                role=coach.role,
                goal=coach.goal,
                backstory=coach.backstory,
                verbose=True,
                allow_delegation=False,
                llm=self.model_name,
                tools=tools or []
            )
            
        except ValueError as e:
            raise ValueError(f"Failed to create agent for coach '{coach_name}': {e}")
    
class TaskLoader(YamlLoader[TaskDescription]):

    def __init__(self, config: Config) -> None:
        super().__init__(Path(config.tasks), TaskDescription)
    
    def create_task(self, task_name: str, **kwargs) -> Task | None:
        task = self.find(task_name)
        if not task:
            return None
        
        return Task(
            name=task_name,
            description=task.description,
            expected_output=task.expected_output,
            markdown=task.markdown or False,
            output_file=task.output_file,
            agent=kwargs.get("agent"),
            context=kwargs.get("context")
        )
        