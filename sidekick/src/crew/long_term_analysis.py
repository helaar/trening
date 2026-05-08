"""Long-term restitution analysis crew."""
import json
import logging
import os
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

from crewai import Crew
from crewai.tools import BaseTool

from config import settings
from models.athlete import Athlete
from models.daily_entry import DailyEntry

from .daily_analysis import _load_yaml, _make_agent, _make_task

logger = logging.getLogger(__name__)


@dataclass
class LongTermAnalysisInput:
    athlete: Athlete
    start_date: str                        # YYYY-MM-DD
    end_date: str                          # YYYY-MM-DD
    daily_entries: list[DailyEntry]        # from DailyEntryRepository.get_range()
    workout_analyses: list[dict[str, Any]] # from WorkoutRepository.get_analyses_for_range()


def _build_timeline(
    start_date: str,
    end_date: str,
    daily_entries: list[DailyEntry],
    workout_analyses: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge restitution entries and workout analyses into a per-day timeline."""
    restitution_by_date: dict[str, dict[str, Any]] = {}
    for entry in daily_entries:
        if entry.restitution:
            r = entry.restitution
            restitution_by_date[entry.date] = {
                "sleep_hours": r.sleep_hours,
                "sleep_quality": r.sleep_quality,
                "hrv": r.hrv,
                "resting_hr": r.resting_hr,
                "readiness": r.readiness,
            }

    training_by_date: dict[str, list[dict[str, Any]]] = {}
    for analysis in workout_analyses:
        session = analysis.get("session", {})
        start_time = session.get("start_time")
        if not start_time:
            continue
        # start_time may be a datetime object or ISO string
        if hasattr(start_time, "date"):
            day_str = start_time.date().isoformat()
        else:
            day_str = str(start_time)[:10]
        training_by_date.setdefault(day_str, []).append(analysis)

    def _aggregate_training(analyses: list[dict[str, Any]]) -> dict[str, Any]:
        tss_values = [
            a.get("metrics", {}).get("training_stress_score")
            for a in analyses
            if a.get("metrics", {}).get("training_stress_score") is not None
        ]
        if_values = [
            a.get("metrics", {}).get("intensity_factor")
            for a in analyses
            if a.get("metrics", {}).get("intensity_factor") is not None
        ]
        total_minutes = sum(
            (a.get("session", {}).get("duration_sec") or 0) / 60
            for a in analyses
        )
        sports = list({
            a.get("session", {}).get("category") or a.get("session", {}).get("sport")
            for a in analyses
            if a.get("session", {}).get("category") or a.get("session", {}).get("sport")
        })
        return {
            "activity_count": len(analyses),
            "total_tss": round(sum(tss_values), 1) if tss_values else None,
            "avg_if": round(sum(if_values) / len(if_values), 3) if if_values else None,
            "total_minutes": round(total_minutes),
            "sports": sports,
        }

    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    timeline = []
    current = start
    while current <= end:
        day_str = current.isoformat()
        training_analyses = training_by_date.get(day_str)
        timeline.append({
            "date": day_str,
            "restitution": restitution_by_date.get(day_str),
            "training": _aggregate_training(training_analyses) if training_analyses else None,
        })
        current += timedelta(days=1)

    return timeline


class _RestitutionDataTool(BaseTool):
    name: str = "get_restitution_data"
    description: str = (
        "Retrieve the daily recovery and training load timeline for the analysis period. "
        "Returns a JSON array with one entry per calendar day, each containing 'date', "
        "'restitution' (HRV, resting HR, sleep, readiness — null if not recorded), and "
        "'training' (TSS, IF, duration — null if no workouts that day). Call this first."
    )
    _payload: str = ""

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, payload: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        object.__setattr__(self, "_payload", payload)

    def _run(self, **kwargs: Any) -> str:
        return self._payload


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
