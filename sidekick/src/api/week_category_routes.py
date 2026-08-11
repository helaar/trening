import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from pymongo.asynchronous.database import AsyncDatabase

from auth.dependencies import get_current_athlete_id
from database.mongodb import get_db
from database.week_category_repository import WeekCategoryRepository
from models.week_category import WeekCategoryEntry, WeekCategoryEntryRequest

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/athlete", tags=["week-category"])


async def get_week_category_repository(
    db: AsyncDatabase = Depends(get_db),
) -> WeekCategoryRepository:
    return WeekCategoryRepository(db)


@router.get("/{athlete_id}/week-category", response_model=list[WeekCategoryEntry])
async def get_week_categories(
    athlete_id: int,
    start: str = Query(..., description="Start week_start (YYYY-MM-DD, Monday), inclusive"),
    end: str = Query(..., description="End week_start (YYYY-MM-DD, Monday), inclusive"),
    current_athlete_id: int = Depends(get_current_athlete_id),
    repo: WeekCategoryRepository = Depends(get_week_category_repository),
) -> list[WeekCategoryEntry]:
    """Get the declared week categories for an athlete within a range of weeks."""
    if athlete_id != current_athlete_id:
        raise HTTPException(status_code=403, detail="You can only access your own data")
    return await repo.get_range(athlete_id, start, end)


@router.post("/{athlete_id}/week-category", response_model=WeekCategoryEntry)
async def upsert_week_category(
    athlete_id: int,
    entry: WeekCategoryEntryRequest,
    current_athlete_id: int = Depends(get_current_athlete_id),
    repo: WeekCategoryRepository = Depends(get_week_category_repository),
) -> WeekCategoryEntry:
    """Create or update the declared category for a specific week."""
    if athlete_id != current_athlete_id:
        raise HTTPException(status_code=403, detail="You can only access your own data")
    logger.info(f"Upserting week category for athlete {athlete_id}, week {entry.week_start}")
    return await repo.upsert(athlete_id, entry)
