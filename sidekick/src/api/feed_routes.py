import asyncio
import logging
from collections import defaultdict
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from pymongo.asynchronous.database import AsyncDatabase

from auth.dependencies import get_current_athlete_id
from database.daily_analysis_repository import DailyAnalysisRepository
from database.daily_entry_repository import DailyEntryRepository
from database.mongodb import get_db
from database.plan_repository import PlanRepository
from database.workout_repository import WorkoutRepository
from models.daily_entry import ActivityAssessment, Restitution
from models.plan import PlannedActivity

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/athlete", tags=["feed"])


class FeedDay(BaseModel):
    date: str
    workouts: list[dict]
    plans: list[PlannedActivity]
    restitution: Restitution | None
    activity_assessments: list[ActivityAssessment]
    has_analysis: bool = False


def _date_str(start_time: str | datetime | None) -> str | None:
    if start_time is None:
        return None
    if isinstance(start_time, datetime):
        return start_time.strftime("%Y-%m-%d")
    return str(start_time)[:10]


@router.get("/{athlete_id}/feed", response_model=list[FeedDay])
async def get_feed(
    athlete_id: int,
    start: str = Query(..., description="Start date YYYY-MM-DD (inclusive)"),
    end: str = Query(..., description="End date YYYY-MM-DD (inclusive)"),
    current_athlete_id: int = Depends(get_current_athlete_id),
    db: AsyncDatabase = Depends(get_db),
) -> list[FeedDay]:
    if athlete_id != current_athlete_id:
        raise HTTPException(status_code=403, detail="You can only access your own data")

    start_dt = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    workout_repo = WorkoutRepository(db)
    plan_repo = PlanRepository(db)
    entry_repo = DailyEntryRepository(db)
    analysis_repo = DailyAnalysisRepository(db)

    workouts, plans, entries = await _fetch_all(
        workout_repo, plan_repo, entry_repo, athlete_id, start, end, start_dt, end_dt
    )

    workouts_by_date: dict[str, list[dict]] = defaultdict(list)
    for w in workouts:
        date = _date_str(w.get("session", {}).get("start_time"))
        if date:
            workouts_by_date[date].append(w)

    plans_by_date: dict[str, list[PlannedActivity]] = defaultdict(list)
    for p in plans:
        plans_by_date[p.date].append(p)

    entries_by_date = {e.date: e for e in entries}

    all_dates = set(workouts_by_date) | set(plans_by_date) | set(entries_by_date)
    dates_with_analysis = await analysis_repo.get_dates_with_analysis(
        athlete_id, list(all_dates)
    )

    result = []
    for date in sorted(all_dates, reverse=True):
        entry = entries_by_date.get(date)
        result.append(
            FeedDay(
                date=date,
                workouts=workouts_by_date[date],
                plans=plans_by_date[date],
                restitution=entry.restitution if entry else None,
                activity_assessments=entry.activity_assessments if entry else [],
                has_analysis=date in dates_with_analysis,
            )
        )

    return result


async def _fetch_all(
    workout_repo: WorkoutRepository,
    plan_repo: PlanRepository,
    entry_repo: DailyEntryRepository,
    athlete_id: int,
    start: str,
    end: str,
    start_dt: datetime,
    end_dt: datetime,
) -> tuple:
    return await asyncio.gather(
        workout_repo.get_analyses_for_range(athlete_id, start_dt, end_dt),
        plan_repo.get_for_range(athlete_id, start, end),
        entry_repo.get_range(athlete_id, start, end),
    )
