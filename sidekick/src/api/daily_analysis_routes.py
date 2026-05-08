from fastapi import APIRouter, Depends, HTTPException, Query
from pymongo.asynchronous.database import AsyncDatabase

from auth.dependencies import get_current_athlete_id
from database.daily_analysis_repository import DailyAnalysisRepository
from database.mongodb import get_db
from models.daily_analysis import DailyAnalysisResult

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
