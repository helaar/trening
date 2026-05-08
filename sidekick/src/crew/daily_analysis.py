"""Daily LLM analysis crew for a single training day."""
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Type

import yaml
from crewai import Agent, Crew, Task
from crewai.tools import BaseTool
from pydantic import BaseModel

from config import settings
from models.athlete import Athlete
from models.plan import PlannedActivity

logger = logging.getLogger(__name__)

_CREW_DIR = Path(__file__).parent


@dataclass
class DailyAnalysisInput:
    athlete: Athlete
    workout_analyses: list[dict[str, Any]]
    planned_activities: list[PlannedActivity]
    date: str


def _athlete_settings_summary(athlete: Athlete) -> dict[str, Any]:
    s = athlete.settings
    result: dict[str, Any] = {}
    if s.heart_rate:
        result["heart_rate"] = {
            "max_hr": s.heart_rate.max,
            "lactate_threshold_hr": s.heart_rate.lt,
            "zones": [{"name": z.name, "min": z.min, "max": z.max} for z in s.heart_rate.hr_zones],
        }
    if s.cycling:
        result["cycling"] = {
            "ftp_watts": s.cycling.ftp,
            "zones": [{"name": z.name, "min": z.min, "max": z.max} for z in s.cycling.power_zones],
        }
    if s.running:
        result["running"] = {
            "ftp_watts": s.running.ftp,
            "zones": [{"name": z.name, "min": z.min, "max": z.max} for z in s.running.power_zones],
        }
    return result


class _WorkoutDataTool(BaseTool):
    name: str = "get_workout_data"
    description: str = (
        "Retrieve workout analyses and athlete threshold/zone settings for today. "
        "Returns JSON with 'athlete_settings' and 'workouts' keys. Call this first."
    )
    _payload: str = ""

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, payload: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        object.__setattr__(self, "_payload", payload)

    def _run(self, **kwargs: Any) -> str:
        return self._payload


class _PlansDataTool(BaseTool):
    name: str = "get_plans_data"
    description: str = (
        "Retrieve today's planned activities and athlete settings. "
        "Returns JSON with 'planned_activities' and 'athlete_settings' keys."
    )
    _payload: str = ""

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, payload: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        object.__setattr__(self, "_payload", payload)

    def _run(self, **kwargs: Any) -> str:
        return self._payload


def _load_yaml(filename: str) -> dict[str, Any]:
    path = _CREW_DIR / filename
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _make_agent(agent_def: dict[str, Any], tools: list, llm: str) -> Agent:
    return Agent(
        role=agent_def["role"],
        goal=agent_def["goal"],
        backstory=agent_def["backstory"].strip(),
        verbose=True,
        allow_delegation=False,
        llm=llm,
        tools=tools,
        memory=False,
    )


def _make_task(task_def: dict[str, Any], agent: Agent, context: list[Task] | None = None) -> Task:
    return Task(
        description=task_def["description"].strip(),
        expected_output=task_def["expected_output"].strip(),
        agent=agent,
        context=context or [],
    )


def run_daily_analysis(input: DailyAnalysisInput) -> dict[str, Any]:
    """Build and run the two-agent daily analysis crew (synchronous — use asyncio.to_thread)."""
    if settings.anthropic_api_key:
        os.environ.setdefault("ANTHROPIC_API_KEY", settings.anthropic_api_key)

    agents_cfg = _load_yaml("agents.yaml")
    tasks_cfg = _load_yaml("tasks.yaml")

    weekday = datetime.fromisoformat(input.date).strftime("%A")
    athlete_name = f"{input.athlete.firstname or ''} {input.athlete.lastname or ''}".strip() or "athlete"

    athlete_settings = _athlete_settings_summary(input.athlete)

    workout_payload = json.dumps({
        "athlete_settings": athlete_settings,
        "workouts": input.workout_analyses,
    }, default=str)

    plans_payload = json.dumps({
        "athlete_settings": athlete_settings,
        "planned_activities": [p.model_dump(mode="json") for p in input.planned_activities],
    }, default=str)

    workout_tool = _WorkoutDataTool(payload=workout_payload)
    plans_tool = _PlansDataTool(payload=plans_payload)

    llm = settings.llm_model

    analyst = _make_agent(agents_cfg["workout_performance_analyst"], tools=[workout_tool], llm=llm)
    coach = _make_agent(agents_cfg["daily_coach"], tools=[plans_tool], llm=llm)

    inputs = {"date": input.date, "weekday": weekday, "athlete_name": athlete_name}

    analysis_task = _make_task(
        {
            "description": tasks_cfg["workout_analysis_task"]["description"].format(**inputs),
            "expected_output": tasks_cfg["workout_analysis_task"]["expected_output"].format(**inputs),
        },
        agent=analyst,
    )
    coaching_task = _make_task(
        {
            "description": tasks_cfg["daily_coaching_task"]["description"].format(**inputs),
            "expected_output": tasks_cfg["daily_coaching_task"]["expected_output"].format(**inputs),
        },
        agent=coach,
        context=[analysis_task],
    )

    crew = Crew(
        agents=[analyst, coach],
        tasks=[analysis_task, coaching_task],
        verbose=True,
    )

    logger.info("Starting daily analysis crew for %s on %s", athlete_name, input.date)
    result = crew.kickoff()

    workout_analysis = ""
    coaching_feedback = ""
    if result.tasks_output and len(result.tasks_output) >= 2:
        workout_analysis = result.tasks_output[0].raw
        coaching_feedback = result.tasks_output[1].raw
    elif result.tasks_output:
        coaching_feedback = result.tasks_output[0].raw
    else:
        coaching_feedback = str(result.raw)

    return {
        "workout_analysis": workout_analysis,
        "coaching_feedback": coaching_feedback,
    }
