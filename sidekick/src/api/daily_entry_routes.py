import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from pymongo.asynchronous.database import AsyncDatabase

from auth.dependencies import get_current_athlete_id
from database.daily_entry_repository import DailyEntryRepository
from database.mongodb import get_db
from models.daily_entry import DailyEntry, DailyEntryRequest

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/athlete", tags=["daily-entry"])


async def get_daily_entry_repository(
    db: AsyncDatabase = Depends(get_db),
) -> DailyEntryRepository:
    return DailyEntryRepository(db)


@router.get("/{athlete_id}/daily-entry", response_model=DailyEntry | None)
async def get_daily_entry(
    athlete_id: int,
    date: str = Query(..., description="Date (YYYY-MM-DD)"),
    current_athlete_id: int = Depends(get_current_athlete_id),
    repo: DailyEntryRepository = Depends(get_daily_entry_repository),
) -> DailyEntry | None:
    """Get the daily entry (restitution + self-assessments) for a specific date."""
    if athlete_id != current_athlete_id:
        raise HTTPException(status_code=403, detail="You can only access your own data")
    return await repo.get(athlete_id, date)


@router.post("/{athlete_id}/daily-entry", response_model=DailyEntry)
async def upsert_daily_entry(
    athlete_id: int,
    entry: DailyEntryRequest,
    current_athlete_id: int = Depends(get_current_athlete_id),
    repo: DailyEntryRepository = Depends(get_daily_entry_repository),
) -> DailyEntry:
    """Create or update the daily entry for a specific date."""
    if athlete_id != current_athlete_id:
        raise HTTPException(status_code=403, detail="You can only access your own data")
    logger.info(f"Upserting daily entry for athlete {athlete_id} on {entry.date}")
    return await repo.upsert(athlete_id, entry)
