import logging
from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from pymongo.asynchronous.database import AsyncDatabase

from auth.dependencies import get_current_athlete_id
from clients.trainingpeaks.ical_client import TPPlannedWorkout, TrainingPeaksICalClient
from database.athlete_repository import AthleteRepository
from database.mongodb import get_db
from database.plan_repository import PlanRepository
from models.plan import PlannedActivity, PlannedActivityRequest

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/athlete", tags=["plans"])


class TPPlannedWorkoutResponse(BaseModel):
    uid: str
    date: date
    sport_type: str
    name: str
    description: str | None
    duration_min: int | None


async def get_plan_repository(db: AsyncDatabase = Depends(get_db)) -> PlanRepository:
    return PlanRepository(db)


async def get_athlete_repository(db: AsyncDatabase = Depends(get_db)) -> AthleteRepository:
    return AthleteRepository(db)


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


@router.get("/{athlete_id}/plans/trainingpeaks", response_model=list[TPPlannedWorkoutResponse])
async def preview_trainingpeaks_plans(
    athlete_id: int,
    start: str = Query(..., description="Range start (YYYY-MM-DD)"),
    end: str = Query(..., description="Range end (YYYY-MM-DD)"),
    current_athlete_id: int = Depends(get_current_athlete_id),
    athlete_repo: AthleteRepository = Depends(get_athlete_repository),
) -> list[TPPlannedWorkout]:
    """Fetch planned workouts from TrainingPeaks iCal feed (read-only preview, nothing is saved)."""
    if athlete_id != current_athlete_id:
        raise HTTPException(status_code=403, detail="You can only access your own data")

    settings = await athlete_repo.get_athlete_settings(athlete_id)
    if settings is None:
        raise HTTPException(status_code=404, detail="Athlete not found")
    if not settings.trainingpeaks_ical_url:
        raise HTTPException(
            status_code=400,
            detail="No TrainingPeaks calendar URL configured. Set it via PATCH /athlete/{id}/settings.",
        )

    try:
        start_date = date.fromisoformat(start)
        end_date = date.fromisoformat(end)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format; use YYYY-MM-DD")

    try:
        tp_client = TrainingPeaksICalClient()
        workouts = await tp_client.get_planned_workouts(
            settings.trainingpeaks_ical_url, start_date, end_date
        )
    except Exception as exc:
        logger.error("Failed to fetch TrainingPeaks iCal feed for athlete %d: %s", athlete_id, exc)
        raise HTTPException(
            status_code=502,
            detail="Failed to fetch TrainingPeaks calendar. Check that the URL is correct and try again.",
        )

    return workouts
