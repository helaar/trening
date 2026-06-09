"""Daily LLM analysis crew for a single training day."""
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import yaml
from crewai import Agent, Crew, Task
from crewai.tools import BaseTool
from pydantic import BaseModel

from config import settings
from crew.prompt_logging import capture_prompt_log, drain_prompt_log
from models.athlete import Athlete
from utils.datetime_utils import convert_datetimes_in_obj
from models.crew_outputs import CoachingOutput, MemoryExtractionOutput, RestitutionAnalysisOutput, WorkoutAnalysisOutput
from models.memory import Memory, MemoryCategory, MemoryScope
from models.daily_entry import DailyEntry
from models.plan import PlannedActivity

logger = logging.getLogger(__name__)

_CREW_DIR = Path(__file__).parent

_RESTITUTION_WINDOW_DAYS = 14


@dataclass
class DailyAnalysisInput:
    athlete: Athlete
    workout_analyses: list[dict[str, Any]]
    planned_activities: list[PlannedActivity]
    date: str
    daily_entries: list[DailyEntry] = field(default_factory=list)
    recent_workout_analyses: list[dict[str, Any]] = field(default_factory=list)
    active_memories: list[Memory] = field(default_factory=list)
    upcoming_races: list[PlannedActivity] = field(default_factory=list)
    prompt_overrides: dict[str, str] = field(default_factory=dict)


_PHILOSOPHY_SUB_KEYS = ("name", "intensity_targets", "coach_guidance", "analyst_guidance")


def _load_philosophy(athlete_id: int, overrides: dict[str, str]) -> dict[str, str] | None:
    """Load philosophy for athlete by slug selection."""
    slug = overrides.get(f"philosophy.{athlete_id}.selected")
    if not slug:
        return None
    result = {k: overrides.get(f"philosophy.{slug}.{k}", "") for k in _PHILOSOPHY_SUB_KEYS}
    return result if result.get("name") else None


def _compute_intensity_distribution(analysis: dict[str, Any]) -> dict[str, Any]:
    """Compute low/moderate/high % from zone list positions (philosophy-neutral).

    Zone position mapping (0-based):
      0-1 → low (Zone 1 Recovery + Zone 2 Endurance)
        2 → moderate (Zone 3 Tempo)
       3+ → high (Zone 4 Threshold and above)
    """
    zones_data = analysis.get("zones", {})

    def _extract(zone_list: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not zone_list:
            return None
        total_seconds = sum(z.get("seconds", 0) for z in zone_list)
        if not total_seconds:
            return None
        low = sum(z.get("seconds", 0) for z in zone_list[:2])
        moderate = zone_list[2].get("seconds", 0) if len(zone_list) > 2 else 0
        high = sum(z.get("seconds", 0) for z in zone_list[3:])
        return {
            "low_pct": round(low / total_seconds * 100, 1),
            "moderate_pct": round(moderate / total_seconds * 100, 1),
            "high_pct": round(high / total_seconds * 100, 1),
        }

    power_zones = (zones_data.get("power_zones") or {}).get("zones") if zones_data else None
    if power_zones:
        dist = _extract(power_zones)
        if dist:
            return {**dist, "basis": "power"}

    hr_zones = (zones_data.get("hr_zones") or {}).get("zones") if zones_data else None
    if hr_zones:
        dist = _extract(hr_zones)
        if dist:
            return {**dist, "basis": "hr"}

    return {"low_pct": None, "moderate_pct": None, "high_pct": None, "basis": "unavailable"}


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
    result["timezone"] = s.timezone
    return result


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
        day_str = start_time.date().isoformat() if hasattr(start_time, "date") else str(start_time)[:10]
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

        low_min = moderate_min = high_min = 0.0
        for a in analyses:
            dist = _compute_intensity_distribution(a)
            if dist.get("basis") == "unavailable":
                continue
            duration_min = (a.get("session", {}).get("duration_sec") or 0) / 60
            low_min += duration_min * (dist.get("low_pct") or 0) / 100
            moderate_min += duration_min * (dist.get("moderate_pct") or 0) / 100
            high_min += duration_min * (dist.get("high_pct") or 0) / 100

        return {
            "activity_count": len(analyses),
            "total_tss": round(sum(tss_values), 1) if tss_values else None,
            "avg_if": round(sum(if_values) / len(if_values), 3) if if_values else None,
            "total_minutes": round(total_minutes),
            "sports": sports,
            "intensity_distribution_minutes": {
                "low": round(low_min, 1),
                "moderate": round(moderate_min, 1),
                "high": round(high_min, 1),
            },
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


class _RacesDataTool(BaseTool):
    name: str = "get_upcoming_races"
    description: str = (
        "Retrieve the athlete's season goal race and other upcoming races from "
        "the day being analyzed onward. Returns JSON with 'season_goal' (the "
        "race tagged 'seasongoal', or null if none is planned) and "
        "'upcoming_races' (a list of races tagged 'race', sorted by date, each "
        "with 'date', 'name', 'sport', and 'days_until' — the number of days "
        "from the day being analyzed to the race)."
    )
    _payload: str = ""

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, payload: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        object.__setattr__(self, "_payload", payload)

    def _run(self, **kwargs: Any) -> str:
        return self._payload


class _MemoryFilterInput(BaseModel):
    category: MemoryCategory | None = None
    scope: MemoryScope | None = None


class _MemoryContextTool(BaseTool):
    name: str = "get_athlete_memories"
    description: str = (
        "Retrieve the active memory bank for this athlete — durable observations about "
        "patterns, habits, risks, and goals built up over previous sessions. "
        "Returns JSON with an 'active_memories' list, each entry containing scope, "
        "category, content, and confidence. Use this to personalise "
        "your coaching and avoid repeating observations the athlete already knows. "
        "Optionally filter by category (recovery, habit, performance, risk, goal) "
        "or scope (recent, long_term) to focus on a specific theme."
    )
    args_schema: type[BaseModel] = _MemoryFilterInput
    _payload: str = ""

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, payload: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        object.__setattr__(self, "_payload", payload)

    def _run(self, category: MemoryCategory | None = None, scope: MemoryScope | None = None) -> str:
        data = json.loads(self._payload)
        memories = data.get("active_memories", [])
        if category is not None:
            memories = [m for m in memories if m.get("category") == category]
        if scope is not None:
            memories = [m for m in memories if m.get("scope") == scope]
        return json.dumps({"active_memories": memories})


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


class _MemoryDataTool(BaseTool):
    name: str = "get_memory_data"
    description: str = (
        "Retrieve the current active memories for the athlete and today's coaching output. "
        "Returns JSON with 'active_memories' and 'coaching_output' keys. Call this first."
    )
    _payload: str = ""

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, payload: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        object.__setattr__(self, "_payload", payload)

    def _run(self, **kwargs: Any) -> str:
        return self._payload


_MAX_MEMORIES = 25


def _format_memories(memories: list[Memory]) -> list[dict[str, Any]]:
    ranked = sorted(memories, key=lambda m: (m.confidence, m.updated_at), reverse=True)[:_MAX_MEMORIES]
    return [
        {
            "scope": m.scope,
            "category": m.category,
            "content": m.content,
            "confidence": m.confidence,
        }
        for m in ranked
    ]


def _load_yaml(filename: str) -> dict[str, Any]:
    path = _CREW_DIR / filename
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _apply_prompt_overrides(cfg: dict[str, Any], prefix: str, overrides: dict[str, str]) -> dict[str, Any]:
    """Return a copy of cfg with string values replaced by matching overrides.

    Override keys use dot-notation relative to the YAML stem, e.g.
    "agents.daily_coach.backstory" -> prefix="agents", key="daily_coach.backstory".
    """
    if not overrides:
        return cfg
    import copy
    result = copy.deepcopy(cfg)
    for full_key, value in overrides.items():
        if not full_key.startswith(f"{prefix}."):
            continue
        relative = full_key[len(prefix) + 1:]
        parts = relative.split(".")
        target = result
        for part in parts[:-1]:
            if not isinstance(target, dict) or part not in target:
                break
            target = target[part]
        else:
            leaf = parts[-1]
            if isinstance(target, dict) and leaf in target:
                target[leaf] = value
    return result


_KNOWN_LLM_PREFIXES = ("anthropic/", "openai/", "azure/", "bedrock/", "vertex_ai/")


def _normalize_llm(model: str) -> str:
    """Add a provider prefix when the model string lacks one.

    CrewAI's router only recognises claude-* models that appear in its
    bundled registry. Newer models (e.g. claude-sonnet-4-6) are absent
    and fall through to the OpenAI provider, causing 404s. An explicit
    prefix bypasses the registry lookup entirely.
    """
    if any(model.startswith(p) for p in _KNOWN_LLM_PREFIXES):
        return model
    if model.startswith("claude-"):
        return f"anthropic/{model}"
    if model.startswith(("gpt-", "o1-", "o3-", "o4-")):
        return f"openai/{model}"
    return model


def _make_agent(agent_def: dict[str, Any], tools: list, default_llm: str) -> Agent:
    llm = _normalize_llm(agent_def.get("llm_model") or default_llm)
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


def _make_task(
    task_def: dict[str, Any],
    agent: Agent,
    context: list[Task] | None = None,
    async_execution: bool = False,
    output_pydantic: type[BaseModel] | None = None,
) -> Task:
    return Task(
        description=task_def["description"].strip(),
        expected_output=task_def["expected_output"].strip(),
        agent=agent,
        context=context or [],
        async_execution=async_execution,
        output_pydantic=output_pydantic,
    )


def run_daily_analysis(input: DailyAnalysisInput) -> dict[str, Any]:
    """Build and run the three-agent daily analysis crew (synchronous — use asyncio.to_thread).

    The workout performance analyst and restitution analyst run in parallel; their
    outputs are both passed as context to the daily coach.
    """
    if settings.anthropic_api_key:
        os.environ.setdefault("ANTHROPIC_API_KEY", settings.anthropic_api_key)
    if settings.openai_api_key:
        os.environ.setdefault("OPENAI_API_KEY", settings.openai_api_key)

    agents_cfg = _apply_prompt_overrides(_load_yaml("agents.yaml"), "agents", input.prompt_overrides)
    tasks_cfg = _apply_prompt_overrides(_load_yaml("tasks.yaml"), "tasks", input.prompt_overrides)

    from datetime import datetime
    weekday = datetime.fromisoformat(input.date).strftime("%A")
    athlete_name = f"{input.athlete.firstname or ''} {input.athlete.lastname or ''}".strip() or "athlete"

    athlete_settings = _athlete_settings_summary(input.athlete)

    # Build a lookup of athlete assessments per activity_id from the day's daily entry
    from models.daily_entry import ActivityAssessment as _ActivityAssessment
    assessment_by_id: dict[int, _ActivityAssessment] = {}
    for entry in input.daily_entries:
        if entry.date == input.date:
            for a in entry.activity_assessments:
                assessment_by_id[a.activity_id] = a

    def _enrich_workout(w: dict) -> dict:
        aid = w.get("activity_id")
        assessment = assessment_by_id.get(aid) if aid else None
        user_tags = assessment.tags if assessment else []
        session_tags = w.get("session", {}).get("tags", [])
        all_tags = list(dict.fromkeys(session_tags + user_tags))

        enriched = dict(w)
        if all_tags:
            enriched["tags"] = all_tags
        if assessment:
            enriched["athlete_rpe"] = assessment.rpe
            if assessment.notes:
                enriched["athlete_notes"] = assessment.notes
        enriched["intensity_distribution"] = _compute_intensity_distribution(w)
        return enriched

    philosophy = _load_philosophy(input.athlete.athlete_id, input.prompt_overrides)

    athlete_tz = input.athlete.settings.timezone
    workout_payload_data: dict[str, Any] = {
        "athlete_settings": athlete_settings,
        "workouts": [_enrich_workout(w) for w in input.workout_analyses],
    }
    if philosophy:
        workout_payload_data["training_philosophy"] = philosophy
    workout_payload = json.dumps(convert_datetimes_in_obj(workout_payload_data, athlete_tz), default=str)

    plans_payload_data: dict[str, Any] = {
        "athlete_settings": athlete_settings,
        "planned_activities": [p.model_dump(mode="json") for p in input.planned_activities],
    }
    if philosophy:
        plans_payload_data["training_philosophy"] = philosophy
    plans_payload = json.dumps(plans_payload_data, default=str)

    def _race_entry(activity: PlannedActivity) -> dict[str, Any]:
        # days_until is relative to the calendar day being analyzed (input.date),
        # not the date the analysis happens to run on.
        race_date = date.fromisoformat(activity.date)
        calendar_day = date.fromisoformat(input.date)
        days_until = (race_date - calendar_day).days
        return {
            "date": activity.date,
            "name": activity.name,
            "sport": activity.sport,
            "days_until": days_until,
        }

    season_goal = next(
        (a for a in input.upcoming_races if "seasongoal" in a.labels), None
    )
    races_payload = json.dumps(
        {
            "season_goal": _race_entry(season_goal) if season_goal else None,
            "upcoming_races": [
                _race_entry(a) for a in input.upcoming_races if "race" in a.labels
            ],
        },
        default=str,
    )

    restitution_start = (
        date.fromisoformat(input.date) - timedelta(days=_RESTITUTION_WINDOW_DAYS - 1)
    ).isoformat()
    timeline = _build_timeline(
        restitution_start,
        input.date,
        input.daily_entries,
        input.recent_workout_analyses,
    )
    restitution_payload = json.dumps(timeline, default=str)

    memory_payload = json.dumps(
        {"active_memories": _format_memories(input.active_memories)},
        default=str,
    )

    workout_tool = _WorkoutDataTool(payload=workout_payload)
    plans_tool = _PlansDataTool(payload=plans_payload)
    races_tool = _RacesDataTool(payload=races_payload)
    restitution_tool = _RestitutionDataTool(payload=restitution_payload)
    memory_context_tool = _MemoryContextTool(payload=memory_payload)

    llm = settings.llm_model

    analyst = _make_agent(agents_cfg["workout_performance_analyst"], tools=[workout_tool], default_llm=llm)
    restitution_analyst = _make_agent(agents_cfg["restitution_analyst"], tools=[restitution_tool], default_llm=llm)
    coach = _make_agent(
        agents_cfg["daily_coach"], tools=[plans_tool, races_tool, memory_context_tool], default_llm=llm
    )

    shared_inputs = {"date": input.date, "weekday": weekday, "athlete_name": athlete_name}
    restitution_inputs = {
        "athlete_name": athlete_name,
        "start_date": restitution_start,
        "end_date": input.date,
        "days": _RESTITUTION_WINDOW_DAYS,
    }

    analysis_task = _make_task(
        {
            "description": tasks_cfg["workout_analysis_task"]["description"].format(**shared_inputs),
            "expected_output": tasks_cfg["workout_analysis_task"]["expected_output"].format(**shared_inputs),
        },
        agent=analyst,
        async_execution=True,
        output_pydantic=WorkoutAnalysisOutput,
    )
    restitution_task = _make_task(
        {
            "description": tasks_cfg["restitution_analysis_task"]["description"].format(**restitution_inputs),
            "expected_output": tasks_cfg["restitution_analysis_task"]["expected_output"].format(**restitution_inputs),
        },
        agent=restitution_analyst,
        async_execution=True,
        output_pydantic=RestitutionAnalysisOutput,
    )
    coaching_task = _make_task(
        {
            "description": tasks_cfg["daily_coaching_task"]["description"].format(**shared_inputs),
            "expected_output": tasks_cfg["daily_coaching_task"]["expected_output"].format(**shared_inputs),
        },
        agent=coach,
        context=[analysis_task, restitution_task],
        output_pydantic=CoachingOutput,
    )

    memory_tool = _MemoryDataTool(payload=memory_payload)
    memory_extractor = _make_agent(agents_cfg["memory_extractor"], tools=[memory_tool], default_llm=llm)
    memory_task = _make_task(
        {
            "description": tasks_cfg["memory_extraction_task"]["description"].format(**shared_inputs),
            "expected_output": tasks_cfg["memory_extraction_task"]["expected_output"].format(**shared_inputs),
        },
        agent=memory_extractor,
        context=[coaching_task],
        output_pydantic=MemoryExtractionOutput,
    )

    crew = Crew(
        agents=[analyst, restitution_analyst, coach, memory_extractor],
        tasks=[analysis_task, restitution_task, coaching_task, memory_task],
        verbose=True,
    )

    logger.info("Starting daily analysis crew for %s on %s", athlete_name, input.date)
    with capture_prompt_log(input.athlete.athlete_id, "daily_analysis", crew) as prompt_log_run_id:
        result = crew.kickoff()
    prompt_log_entries = drain_prompt_log(prompt_log_run_id)

    workout_output: WorkoutAnalysisOutput | None = None
    restitution_output: RestitutionAnalysisOutput | None = None
    coaching_output: CoachingOutput | None = None
    memory_extraction_output: MemoryExtractionOutput | None = None

    if result.tasks_output and len(result.tasks_output) >= 4:
        t0, t1, t2, t3 = result.tasks_output[:4]
        workout_output = t0.pydantic if isinstance(t0.pydantic, WorkoutAnalysisOutput) else None
        restitution_output = t1.pydantic if isinstance(t1.pydantic, RestitutionAnalysisOutput) else None
        coaching_output = t2.pydantic if isinstance(t2.pydantic, CoachingOutput) else None
        memory_extraction_output = t3.pydantic if isinstance(t3.pydantic, MemoryExtractionOutput) else None
        if workout_output is None:
            logger.warning("workout_analysis_task pydantic output missing, raw=%r", t0.raw[:200])
        if restitution_output is None:
            logger.warning("restitution_analysis_task pydantic output missing, raw=%r", t1.raw[:200])
        if coaching_output is None:
            logger.warning("daily_coaching_task pydantic output missing, raw=%r", t2.raw[:200])
        if memory_extraction_output is None:
            logger.warning("memory_extraction_task pydantic output missing, raw=%r", t3.raw[:200])
    else:
        logger.warning("Unexpected tasks_output length: %d", len(result.tasks_output or []))

    return {
        "workout_analysis": workout_output,
        "restitution_analysis": restitution_output,
        "coaching_feedback": coaching_output,
        "memory_extraction": memory_extraction_output,
        "prompt_log_entries": prompt_log_entries,
    }
