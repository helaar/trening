from datetime import date
from fastapi import APIRouter, Depends, Query, HTTPException
from pymongo.asynchronous.database import AsyncDatabase

from auth.dependencies import get_current_athlete_id, get_strava_client, get_athlete_repository
from clients.strava.client import StravaClient
from database.mongodb import get_db
from database.workout_repository import WorkoutRepository
from database.athlete_repository import AthleteRepository
from models.workout import DailySummary
from services.workout_analysis import WorkoutAnalysisService


router = APIRouter(prefix="/athlete", tags=["workouts"])


async def get_workout_repository(db: AsyncDatabase = Depends(get_db)) -> WorkoutRepository:
    """Dependency to get workout repository."""
    return WorkoutRepository(db)


async def get_workout_analysis_service(
    strava_client: StravaClient = Depends(get_strava_client),
    workout_repo: WorkoutRepository = Depends(get_workout_repository),
    athlete_repo: AthleteRepository = Depends(get_athlete_repository)
) -> WorkoutAnalysisService:
    """Dependency to get workout analysis service."""
    return WorkoutAnalysisService(strava_client, workout_repo, athlete_repo)


@router.get("/{athlete_id}/workouts", response_model=DailySummary)
async def get_athlete_workouts(
    athlete_id: int,
    date: date = Query(..., description="Date to fetch workouts for (YYYY-MM-DD)"),
    refresh: bool = Query(False, description="Force refresh from Strava API"),
    current_athlete_id: int = Depends(get_current_athlete_id),
    analysis_service: WorkoutAnalysisService = Depends(get_workout_analysis_service)
) -> DailySummary:
    """
    Get workouts for a specific athlete and date with analysis.
    
    If already downloaded from Strava and stored in database, reads from database
    unless refresh parameter is set to true.
    
    Uses athlete settings (FTP, threshold HR) from database for training load calculations.
    Returns activity summaries with calculated training load metrics (TSS, HRSS).
    """
    # Verify the authenticated user is requesting their own data
    if athlete_id != current_athlete_id:
        raise HTTPException(
            status_code=403,
            detail="You can only access your own workout data"
        )
    
    daily_summary = await analysis_service.get_workouts_for_date(
        athlete_id=athlete_id,
        activity_date=date,
        refresh=refresh
    )
    
    return daily_summary
