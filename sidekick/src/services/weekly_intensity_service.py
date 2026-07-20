"""Weekly intensity-distribution assessment, computed directly from synced workouts.

Unlike the daily LLM coaching crew, this requires no agent run — it aggregates
already-stored workout analyses, so it is available as soon as a day's workouts have
been synced from Strava.
"""

from datetime import date, datetime, timedelta
from typing import Any

from analysis.weekly_intensity import compute_weekly_philosophy_assessment
from database.athlete_repository import AthleteRepository
from database.crew_definition_repository import CrewDefinitionRepository
from database.workout_repository import WorkoutRepository

# Philosophies the weekly low/moderate/high chart is shown for. The underlying
# aggregation is philosophy-neutral zone-band math, so any philosophy whose model maps
# onto a low/moderate/high split can be added here for the chart alone.
#
# This is intentionally a superset of crew/daily_analysis.py's `_POLARIZED_SLUG`: that
# gate feeds `weekly_philosophy_assessment` into the LLM prompt, where the status labels
# ("polarized"/"mild_drift"/"gray_zone_week") are only explained to the model for the
# polarized 80/20 philosophy's coach_guidance/analyst_guidance. Adding a philosophy here
# only affects the standalone /weekly-intensity chart, not what the LLM crew sees.
_WEEKLY_ASSESSMENT_PHILOSOPHIES = {"polarized_80_20", "incrementel_321"}


async def get_weekly_intensity(
    athlete_repo: AthleteRepository,
    workout_repo: WorkoutRepository,
    crew_def_repo: CrewDefinitionRepository,
    athlete_id: int,
    date_str: str,
) -> dict[str, Any] | None:
    """Weekly philosophy assessment for the 7 days ending on date_str, or None.

    Returns None when the athlete has no training philosophy set, or it isn't one of
    the philosophies this chart is defined for (_WEEKLY_ASSESSMENT_PHILOSOPHIES).
    """
    athlete = await athlete_repo.get_athlete(athlete_id)
    if not athlete or not athlete.settings.training_philosophy:
        return None

    philosophy = await crew_def_repo.get_philosophy(athlete.settings.training_philosophy)
    if not philosophy or philosophy.name not in _WEEKLY_ASSESSMENT_PHILOSOPHIES:
        return None

    end = date.fromisoformat(date_str)
    start = end - timedelta(days=6)
    recent_analyses = await workout_repo.get_analyses_for_range(
        athlete_id,
        datetime.combine(start, datetime.min.time()),
        datetime.combine(end, datetime.min.time()),
    )
    return compute_weekly_philosophy_assessment(recent_analyses, date_str)
