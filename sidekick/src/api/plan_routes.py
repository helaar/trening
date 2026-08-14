import logging
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from pymongo.asynchronous.database import AsyncDatabase

from auth.dependencies import get_current_athlete_id
from database.athlete_repository import AthleteRepository
from database.intervals_activity_repository import IntervalsActivityRepository
from database.mongodb import get_db
from database.workout_repository import WorkoutRepository
from models.plan import PlannedActivity
from services import intervals_calendar
from services.plan_matching import attach_matches, attach_matches_for_range, group_workouts_by_date

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/athlete", tags=["plans"])


async def get_athlete_repository(db: AsyncDatabase = Depends(get_db)) -> AthleteRepository:
    return AthleteRepository(db)


@router.get("/{athlete_id}/plans", response_model=list[PlannedActivity])
async def get_plans(
    athlete_id: int,
    date: str | None = Query(None, description="Single date (YYYY-MM-DD)"),
    start: str | None = Query(None, description="Range start (YYYY-MM-DD)"),
    end: str | None = Query(None, description="Range end (YYYY-MM-DD)"),
    current_athlete_id: int = Depends(get_current_athlete_id),
    athlete_repo: AthleteRepository = Depends(get_athlete_repository),
    db: AsyncDatabase = Depends(get_db),
) -> list[PlannedActivity]:
    """Planned workouts, read live from the athlete's Intervals.icu calendar.

    Read-only: sidekick has no plan-authoring UI of its own — the athlete
    creates/edits planned workouts directly in Intervals.icu. Each plan's
    `matched_activity_id` is filled in from that date's completed workouts,
    via Intervals.icu's own planned/actual pairing (see services/plan_matching.py).
    """
    if athlete_id != current_athlete_id:
        raise HTTPException(status_code=403, detail="You can only access your own data")
    settings = await athlete_repo.get_athlete_settings(athlete_id)
    workout_repo = WorkoutRepository(db)
    intervals_activity_repo = IntervalsActivityRepository(db)

    if date:
        plans = await intervals_calendar.get_for_date(athlete_id, settings, date)
        activity_date = datetime.strptime(date, "%Y-%m-%d")
        workouts = await workout_repo.get_analyses_for_date(athlete_id, activity_date)
        await attach_matches(athlete_id, plans, workouts, intervals_activity_repo)
        return plans
    if start and end:
        plans = await intervals_calendar.get_for_range(athlete_id, settings, start, end)
        start_dt = datetime.strptime(start, "%Y-%m-%d")
        end_dt = datetime.strptime(end, "%Y-%m-%d")
        workouts = await workout_repo.get_analyses_for_range(athlete_id, start_dt, end_dt)

        plans_by_date: dict[str, list[PlannedActivity]] = {}
        for plan in plans:
            plans_by_date.setdefault(plan.date, []).append(plan)
        workouts_by_date = group_workouts_by_date(workouts)
        await attach_matches_for_range(
            athlete_id, plans_by_date, workouts_by_date, intervals_activity_repo
        )
        return plans
    raise HTTPException(status_code=400, detail="Provide either 'date' or both 'start' and 'end'")
