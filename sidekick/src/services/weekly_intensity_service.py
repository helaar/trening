"""Weekly polarized-intensity assessment, computed directly from synced workouts.

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

# Must match the philosophy slug the daily analysis crew gates on
# (crew/daily_analysis.py's `_POLARIZED_SLUG`) so both call sites agree on when a
# weekly assessment is meaningful.
_POLARIZED_SLUG = "polarized_80_20"


async def get_weekly_intensity(
    athlete_repo: AthleteRepository,
    workout_repo: WorkoutRepository,
    crew_def_repo: CrewDefinitionRepository,
    athlete_id: int,
    date_str: str,
) -> dict[str, Any] | None:
    """Weekly philosophy assessment for the 7 days ending on date_str, or None.

    Returns None when the athlete has no training philosophy set, or it isn't the
    polarized 80/20 philosophy this assessment is defined for.
    """
    athlete = await athlete_repo.get_athlete(athlete_id)
    if not athlete or not athlete.settings.training_philosophy:
        return None

    philosophy = await crew_def_repo.get_philosophy(athlete.settings.training_philosophy)
    if not philosophy or philosophy.name != _POLARIZED_SLUG:
        return None

    end = date.fromisoformat(date_str)
    start = end - timedelta(days=6)
    recent_analyses = await workout_repo.get_analyses_for_range(
        athlete_id,
        datetime.combine(start, datetime.min.time()),
        datetime.combine(end, datetime.min.time()),
    )
    return compute_weekly_philosophy_assessment(recent_analyses, date_str)
