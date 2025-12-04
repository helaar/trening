"""Coach loader utility for parsing YAML and creating CrewAI agents."""
from datetime import date, timedelta
from typing import override
from pydantic import BaseModel
import yaml
from pathlib import Path
from crewai import Agent, Task
from .models import Coach, CommonKnowledge, Plan, TaskDescription
from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource
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
    
    def find(self, key: str) -> list[T] :
        maybe_dict = self.config.get(key)
        if maybe_dict and isinstance(maybe_dict, dict) :
            return [self.model_class(**maybe_dict)] 
        if maybe_dict and isinstance(maybe_dict, list):
            return [ self.model_class(**i) for i in maybe_dict]
        return []
    
    def get(self, key: str) -> T | None:
        items = self.find(key)
        if items:
            return items[0]
        return None
    



class CoachLoader(YamlLoader[Coach]):
    """Utility class for loading coach definitions from YAML and creating agents."""
    
    def __init__(self, config: Config):
        super().__init__(Path(config.coaches), Coach)
        self.model_name = config.model_name

    def create_coach_agent(self, coach_name: str, memory, reasoning:bool | None=False, tools: list | None = None) -> Agent:
        """Create a CrewAI Agent from a coach definition."""
        try:
            # Extract CrewAI parameters
            coach = self.get(coach_name)
            if not coach:
                raise ValueError(f"Coach '{coach_name}' not found in configuration.")
            
            reasoning_steps = 3 if reasoning else 0
            # Create and return the agent
            return Agent(
                role=coach.role,
                goal=coach.goal,
                backstory=coach.backstory,
                verbose=True,
                allow_delegation=False,
                llm=self.model_name,
                tools=tools or [],
                memory=memory,
                reasoning=reasoning,
                reasoning_max_steps=reasoning_steps
            )
            
        except ValueError as e:
            raise ValueError(f"Failed to create agent for coach '{coach_name}': {e}")
    
class TaskLoader(YamlLoader[TaskDescription]):

    def __init__(self, config: Config) -> None:
        super().__init__(Path(config.tasks), TaskDescription)
    
    def create_task(self, task_name: str, **kwargs) -> Task | None:
        task = self.get(task_name)
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
        
class KnowledgeLoader(YamlLoader[CommonKnowledge]):
    """Utility class for loading knowledge source definitions from YAML."""
    
    def __init__(self, config: Config) -> None:
        super().__init__(Path(config.knowledge), CommonKnowledge)

    def get_knowledge(self) -> StringKnowledgeSource:
        ck = self.find("common_knowledge")

        return StringKnowledgeSource(content="\n\n".join([f"Rule: {k.rule}\nAccept: {k.accept}\nReject: {k.reject}" for k in ck])) 
            
class PlansLoader(YamlLoader[Plan]):
    def __init__(self, config: Config) -> None:
        super().__init__(Path(config.plans), Plan)

    @override
    def find(self, key: str) -> list[Plan]:
        """Override find to transform date keys into Plan instances."""
        athlete_plans = self.config.get(key)
        if not athlete_plans or not isinstance(athlete_plans, dict):
            return []
        
        plans = []
        for date_key, activities in athlete_plans.items():
            if isinstance(activities, list):
                # Remove 'd' prefix from date key (e.g., 'd2025-12-04' -> '2025-12-04')
                clean_date = date_key.lstrip('d') if date_key.startswith('d') else date_key
                plans.append(Plan(date=clean_date, activities=activities))
        return plans

    def get_plan(self, athlete:str, plan_date:str) -> list[str] | None:
        plans = self.find(athlete)
        if not plans:
            return None

        activities = []
        for p in plans:
            
            if p.date == plan_date:
                activities.extend(p.activities)
        
        if not activities:
            return None
        return activities
    
    def get_plans(self, athlete: str, start:date|str, end: date|str) -> dict[str, list[str]]:

        start = self._date(start)
        end = self._date(end)
        s = min(start,end)
        e = max(start,end)

        plan:dict[str,list[str]] = {}
        while s <= e:
            d = s.isoformat()
            p = self.get_plan(athlete, d)
            if p:
                print( f"Found plan for {d}: {p}")
                plan[d] = p
            s = s + timedelta(days=1)
        return plan

    @staticmethod
    def _date(dt: str | date) -> date:
        if isinstance(dt, date):
            return dt
        return date.fromisoformat(dt)