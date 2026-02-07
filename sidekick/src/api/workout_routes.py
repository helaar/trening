import logging

from datetime import datetime
from fastapi import APIRouter, Depends, Query, HTTPException
from pymongo.asynchronous.database import AsyncDatabase

from auth.dependencies import get_current_athlete_id, get_strava_client, get_athlete_repository
from clients.strava.client import StravaClient
from database.mongodb import get_db
from database.workout_repository import WorkoutRepository
from database.athlete_repository import AthleteRepository
from models.workout import DailySummary
from analysis.models import WorkoutAnalysis
from services.workout_analysis import WorkoutAnalysisService

logger = logging.getLogger(__name__)
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


@router.get("/{athlete_id}/workouts/detailed", response_model=list[WorkoutAnalysis])
async def get_athlete_workouts_detailed(
    athlete_id: int,
    date: datetime = Query(..., description="Date to fetch workouts for (YYYY-MM-DD)"),
    refresh: bool = Query(False, description="Force refresh from Strava API"),
    current_athlete_id: int = Depends(get_current_athlete_id),
    analysis_service: WorkoutAnalysisService = Depends(get_workout_analysis_service)
) -> list[WorkoutAnalysis]:
    """
    Get detailed analysis for all workouts on a specific date.
    
    Provides the same rich analytics as the strava-analyze script (JSON variant),
    including zone distributions, lap analysis, heart rate drift, power metrics, etc.
    
    Uses athlete settings (FTP, threshold HR, zones) from database.
    
    Expected URL: /api/v1/athlete/{athlete_id}/workouts/detailed?date=YYYY-MM-DD
    """
    
    logger.info(f"GET /athlete/{athlete_id}/workouts/detailed called with date={date}, refresh={refresh}")
    
    # Verify the authenticated user is requesting their own data
    if athlete_id != current_athlete_id:
        raise HTTPException(
            status_code=403,
            detail="You can only access your own workout data"
        )
    
    analyses = await analysis_service.get_detailed_analyses_for_date(
        athlete_id=athlete_id,
        activity_date=date,
        refresh=refresh
    )
    
    return analyses


@router.get("/{athlete_id}/workouts", response_model=DailySummary)
async def get_athlete_workouts(
    athlete_id: int,
    date: datetime = Query(..., description="Date to fetch workouts for (YYYY-MM-DD)"),
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


@router.get("/{athlete_id}/activities/{activity_id}/analysis", response_model=WorkoutAnalysis)
async def get_activity_analysis(
    athlete_id: int,
    activity_id: int,
    refresh: bool = Query(False, description="Force refresh from Strava API"),
    current_athlete_id: int = Depends(get_current_athlete_id),
    analysis_service: WorkoutAnalysisService = Depends(get_workout_analysis_service)
) -> WorkoutAnalysis:
    """
    Get detailed analysis for a specific activity.
    
    Provides the same rich analytics as the strava-analyze script (JSON variant),
    including zone distributions, lap analysis, heart rate drift, power metrics, etc.
    
    Uses athlete settings (FTP, threshold HR, zones) from database.
    """
    # Verify the authenticated user is requesting their own data
    if athlete_id != current_athlete_id:
        raise HTTPException(
            status_code=403,
            detail="You can only access your own workout data"
        )
    
    try:
        analysis = await analysis_service.get_detailed_analysis(
            athlete_id=athlete_id,
            activity_id=activity_id,
            refresh=refresh
        )
        return analysis
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
