""" Tool to read a athlete's profile from a file """

from datetime import date
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
import yaml

from crewai.tools import BaseTool

from crew.loaders import PlansLoader


class AthleteLookupTool(BaseTool):
    """Tool that retrieves a single athlete's information from a YAML file."""
    name: str = "athlete_lookup"
    description: str = (
        "Use this tool to retrieve information about an athlete by unique athlete identificator (athlete). "
        "Returns data in a textual, JSON-like form."
    )
    yaml_path: Path
    _cache: dict[str,Any] | None = None

    def _run(self, athlete: str) -> str:
        data = self._load_yaml()
        athletes = data.get("athletes", {})
        athlete_dict = athletes.get(athlete.lower())
        if not athlete_dict:
            return f"Athlete {athlete} not found."
        
        return f"{athlete_dict}"

    def athlete_as_dict(self, athlete: str) -> dict[str, Any] | None:
        data = self._load_yaml()
        athletes = data.get("athletes", {})
        athlete_dict = athletes.get(athlete.lower())
        return athlete_dict

    def _load_yaml(self) -> dict[str, Any]:
        if self._cache is not None:
            return self._cache

        with self.yaml_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        self._cache = data
        return data

class AthletePlanArgs(BaseModel):
    athlete: str = Field(description="Unique identifier of the athlete")
    start_date: date = Field(description="Start date of the plan range")
    end_date: date = Field(description="End date of the plan range")

class AthletePlanTool(BaseTool):
    """Tool that retrieves an athlete's training plan from a YAML file."""
    args_schema: type = AthletePlanArgs
    def __init__(self, loader: PlansLoader) -> None:
        super().__init__(
            name="athlete_plan_lookup",
            description="Use this tool to retrieve an athlete's training plan for a given date range. Returns data in a textual, JSON-like form."
        )
        self._loader = loader

    def _run(self, athlete: str, start_date: date, end_date: date) -> str:
        plans_dict = self._loader.get_plans(athlete, start_date, end_date)
        
        return f"{plans_dict}\n\nHint: Unplanned workouts may be transportation/commute rides or runs not included in the plan." if plans_dict else f"No plans found for athlete {athlete} between {start_date} and {end_date}."


