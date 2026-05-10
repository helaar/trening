"""Long-term restitution analysis crew."""
import json
import logging
import os
from dataclasses import dataclass
from datetime import date
from typing import Any

from crewai import Crew

from config import settings
from models.athlete import Athlete
from models.daily_entry import DailyEntry

from .daily_analysis import (
    _RestitutionDataTool,
    _build_timeline,
    _load_yaml,
    _make_agent,
    _make_task,
)

logger = logging.getLogger(__name__)


@dataclass
class LongTermAnalysisInput:
    athlete: Athlete
    start_date: str                        # YYYY-MM-DD
    end_date: str                          # YYYY-MM-DD
    daily_entries: list[DailyEntry]        # from DailyEntryRepository.get_range()
    workout_analyses: list[dict[str, Any]] # from WorkoutRepository.get_analyses_for_range()


def run_long_term_analysis(input: LongTermAnalysisInput) -> dict[str, Any]:
    """Build and run the restitution analyst crew (synchronous — use asyncio.to_thread)."""
    if settings.anthropic_api_key:
        os.environ.setdefault("ANTHROPIC_API_KEY", settings.anthropic_api_key)
    if settings.openai_api_key:
        os.environ.setdefault("OPENAI_API_KEY", settings.openai_api_key)

    agents_cfg = _load_yaml("agents.yaml")
    tasks_cfg = _load_yaml("tasks.yaml")

    athlete_name = (
        f"{input.athlete.firstname or ''} {input.athlete.lastname or ''}".strip()
        or "athlete"
    )
    days = (
        date.fromisoformat(input.end_date) - date.fromisoformat(input.start_date)
    ).days + 1

    timeline = _build_timeline(
        input.start_date,
        input.end_date,
        input.daily_entries,
        input.workout_analyses,
    )
    restitution_tool = _RestitutionDataTool(payload=json.dumps(timeline, default=str))

    llm = settings.llm_model
    analyst = _make_agent(
        agents_cfg["restitution_analyst"],
        tools=[restitution_tool],
        default_llm=llm,
    )

    task_inputs = {
        "athlete_name": athlete_name,
        "start_date": input.start_date,
        "end_date": input.end_date,
        "days": days,
    }
    task_cfg = tasks_cfg["restitution_analysis_task"]
    analysis_task = _make_task(
        {
            "description": task_cfg["description"].format(**task_inputs),
            "expected_output": task_cfg["expected_output"].format(**task_inputs),
        },
        agent=analyst,
    )

    crew = Crew(agents=[analyst], tasks=[analysis_task], verbose=True)

    logger.info(
        "Starting restitution analysis crew for %s (%s to %s)",
        athlete_name,
        input.start_date,
        input.end_date,
    )
    result = crew.kickoff()

    restitution_analysis = (
        result.tasks_output[0].raw if result.tasks_output else str(result.raw)
    )
    return {"restitution_analysis": restitution_analysis}
