import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from pymongo.asynchronous.database import AsyncDatabase

from auth.dependencies import get_current_athlete_id
from database.athlete_repository import AthleteRepository
from database.mongodb import get_db
from models.plan import PlannedActivity
from services import intervals_calendar

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
) -> list[PlannedActivity]:
    """Planned workouts, read live from the athlete's Intervals.icu calendar.

    Read-only: sidekick has no plan-authoring UI of its own — the athlete
    creates/edits planned workouts directly in Intervals.icu.
    """
    if athlete_id != current_athlete_id:
        raise HTTPException(status_code=403, detail="You can only access your own data")
    settings = await athlete_repo.get_athlete_settings(athlete_id)
    if date:
        return await intervals_calendar.get_for_date(athlete_id, settings, date)
    if start and end:
        return await intervals_calendar.get_for_range(athlete_id, settings, start, end)
    raise HTTPException(status_code=400, detail="Provide either 'date' or both 'start' and 'end'")
