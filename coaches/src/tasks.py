


from pathlib import Path

from crewai import Agent, Task
from pydantic import BaseModel
import yaml

class TaskDescription(BaseModel):
    """Class representing a task description loaded from YAML."""
    description: str
    expected_output: str 
    markdown: bool | None = False
    output_file: str | None = None

class TaskLoader:

    def __init__(self, task_file : Path ) -> None:
        self.tasks: dict[str, object] = self._load_yaml(Path(task_file))


    def _load_yaml(self, filename: Path ) -> dict[str, object]:
        with   filename.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data
    
    def create_task(self, task_name: str, **kwargs) -> Task | None:
        data = self.tasks.get(task_name)
        if not data:
            return None
        
        task_data = TaskDescription(**data) # type: ignore
        task = Task(
            name=task_name,
            description=task_data.description,
            expected_output=task_data.expected_output,
            markdown=task_data.markdown or False,
            output_file=task_data.output_file,
            agent=kwargs.get("agent"),
            context=kwargs.get("context")
        )
        return task