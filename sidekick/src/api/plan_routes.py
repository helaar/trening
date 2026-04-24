import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from pymongo.asynchronous.database import AsyncDatabase

from auth.dependencies import get_current_athlete_id
from database.mongodb import get_db
from database.plan_repository import PlanRepository
from models.plan import PlannedActivity, PlannedActivityRequest

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/athlete", tags=["plans"])


async def get_plan_repository(db: AsyncDatabase = Depends(get_db)) -> PlanRepository:
    return PlanRepository(db)


@router.get("/{athlete_id}/plans", response_model=list[PlannedActivity])
async def get_plans(
    athlete_id: int,
    date: str | None = Query(None, description="Single date (YYYY-MM-DD)"),
    start: str | None = Query(None, description="Range start (YYYY-MM-DD)"),
    end: str | None = Query(None, description="Range end (YYYY-MM-DD)"),
    current_athlete_id: int = Depends(get_current_athlete_id),
    repo: PlanRepository = Depends(get_plan_repository),
) -> list[PlannedActivity]:
    if athlete_id != current_athlete_id:
        raise HTTPException(status_code=403, detail="You can only access your own data")
    if date:
        return await repo.get_for_date(athlete_id, date)
    if start and end:
        return await repo.get_for_range(athlete_id, start, end)
    raise HTTPException(status_code=400, detail="Provide either 'date' or both 'start' and 'end'")


@router.post("/{athlete_id}/plans", response_model=PlannedActivity, status_code=201)
async def create_plan(
    athlete_id: int,
    request: PlannedActivityRequest,
    current_athlete_id: int = Depends(get_current_athlete_id),
    repo: PlanRepository = Depends(get_plan_repository),
) -> PlannedActivity:
    if athlete_id != current_athlete_id:
        raise HTTPException(status_code=403, detail="You can only access your own data")
    logger.info(f"Creating plan for athlete {athlete_id} on {request.date}")
    return await repo.create(athlete_id, request)


@router.put("/{athlete_id}/plans/{plan_id}", response_model=PlannedActivity)
async def update_plan(
    athlete_id: int,
    plan_id: str,
    request: PlannedActivityRequest,
    current_athlete_id: int = Depends(get_current_athlete_id),
    repo: PlanRepository = Depends(get_plan_repository),
) -> PlannedActivity:
    if athlete_id != current_athlete_id:
        raise HTTPException(status_code=403, detail="You can only access your own data")
    updated = await repo.update(athlete_id, plan_id, request)
    if not updated:
        raise HTTPException(status_code=404, detail="Plan not found")
    return updated


@router.delete("/{athlete_id}/plans/{plan_id}", status_code=204)
async def delete_plan(
    athlete_id: int,
    plan_id: str,
    current_athlete_id: int = Depends(get_current_athlete_id),
    repo: PlanRepository = Depends(get_plan_repository),
) -> None:
    if athlete_id != current_athlete_id:
        raise HTTPException(status_code=403, detail="You can only access your own data")
    deleted = await repo.delete(athlete_id, plan_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Plan not found")
