from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pymongo.asynchronous.database import AsyncDatabase

from auth.dependencies import get_current_athlete_id
from database.athlete_repository import AthleteRepository
from database.crew_definition_repository import CrewDefinitionRepository
from database.daily_analysis_repository import DailyAnalysisRepository
from database.mongodb import get_db
from database.workout_repository import WorkoutRepository
from models.daily_analysis import DailyAnalysisResult
from services.weekly_intensity_service import get_weekly_intensity

router = APIRouter(prefix="/athlete", tags=["daily-analysis"])


async def get_daily_analysis_repository(
    db: AsyncDatabase = Depends(get_db),
) -> DailyAnalysisRepository:
    return DailyAnalysisRepository(db)


@router.get("/{athlete_id}/daily-analysis", response_model=DailyAnalysisResult | None)
async def get_daily_analysis(
    athlete_id: int,
    date: str = Query(..., description="Date (YYYY-MM-DD)"),
    current_athlete_id: int = Depends(get_current_athlete_id),
    repo: DailyAnalysisRepository = Depends(get_daily_analysis_repository),
) -> DailyAnalysisResult | None:
    """Get the stored LLM coaching analysis for a specific date, or null if none exists."""
    if athlete_id != current_athlete_id:
        raise HTTPException(status_code=403, detail="You can only access your own data")
    return await repo.get(athlete_id, date)


@router.get("/{athlete_id}/weekly-intensity")
async def get_weekly_intensity_route(
    athlete_id: int,
    date: str = Query(..., description="Date (YYYY-MM-DD), the end of the 7-day window"),
    current_athlete_id: int = Depends(get_current_athlete_id),
    db: AsyncDatabase = Depends(get_db),
) -> dict[str, Any] | None:
    """Deterministic weekly intensity-distribution assessment for the 7 days ending on date.

    Computed directly from synced workout analyses — available as soon as a day's
    workouts have been fetched from Strava, independent of the daily LLM coaching crew.
    Returns null if the athlete has no training philosophy set, or one this chart isn't
    defined for (see weekly_intensity_service._WEEKLY_ASSESSMENT_PHILOSOPHIES).
    """
    if athlete_id != current_athlete_id:
        raise HTTPException(status_code=403, detail="You can only access your own data")
    return await get_weekly_intensity(
        AthleteRepository(db),
        WorkoutRepository(db),
        CrewDefinitionRepository(db),
        athlete_id,
        date,
    )
