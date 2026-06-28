from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from pymongo.asynchronous.database import AsyncDatabase

from auth.dependencies import (
    get_athlete_repository,
    get_coach_repository,
    get_current_coach_id,
    verify_coach_athlete_access,
)
from api.feed_routes import FeedDay, build_feed
from database.athlete_repository import AthleteRepository
from database.coach_repository import CoachRepository
from database.daily_analysis_repository import DailyAnalysisRepository
from database.daily_entry_repository import DailyEntryRepository
from database.mongodb import get_db
from models.daily_analysis import DailyAnalysisResult

router = APIRouter(prefix="/coach", tags=["coach"])


class RosterAthlete(BaseModel):
    athlete_id: int
    name: str | None
    profile_picture: str | None


class CoachProfile(BaseModel):
    coach_id: int
    display_name: str | None
    roster: list[RosterAthlete]


class RiskCounts(BaseModel):
    low: int = 0
    moderate: int = 0
    high: int = 0


class AthleteStatus(BaseModel):
    athlete_id: int
    name: str | None
    profile_picture: str | None
    last_activity_date: str | None
    latest_readiness: float | None
    latest_hrv: int | None
    performance_risk: RiskCounts
    restitution_risk: RiskCounts
    overall_recovery_quality: str | None


def _athlete_name(athlete) -> str | None:
    if athlete is None:
        return None
    parts = [p for p in (athlete.firstname, athlete.lastname) if p]
    return " ".join(parts) if parts else athlete.username


@router.get("/me", response_model=CoachProfile)
async def get_coach_profile(
    coach_id: int = Depends(get_current_coach_id),
    coach_repo: CoachRepository = Depends(get_coach_repository),
    athlete_repo: AthleteRepository = Depends(get_athlete_repository),
) -> CoachProfile:
    coach = await coach_repo.get_coach(coach_id)
    roster = []
    for athlete_id in coach.athlete_ids if coach else []:
        athlete = await athlete_repo.get_athlete(athlete_id)
        roster.append(
            RosterAthlete(
                athlete_id=athlete_id,
                name=_athlete_name(athlete),
                profile_picture=athlete.profile_picture if athlete else None,
            )
        )
    return CoachProfile(
        coach_id=coach_id,
        display_name=coach.display_name if coach else None,
        roster=roster,
    )


@router.get("/roster", response_model=list[AthleteStatus])
async def get_roster_status(
    window_days: int = Query(14, ge=1, le=90),
    coach_id: int = Depends(get_current_coach_id),
    coach_repo: CoachRepository = Depends(get_coach_repository),
    athlete_repo: AthleteRepository = Depends(get_athlete_repository),
    db: AsyncDatabase = Depends(get_db),
) -> list[AthleteStatus]:
    roster_ids = await coach_repo.get_roster_athlete_ids(coach_id)

    end_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    start_date = (datetime.now(timezone.utc) - timedelta(days=window_days)).strftime("%Y-%m-%d")

    entry_repo = DailyEntryRepository(db)
    analysis_repo = DailyAnalysisRepository(db)

    statuses = []
    for athlete_id in roster_ids:
        athlete = await athlete_repo.get_athlete(athlete_id)
        entries = await entry_repo.get_range(athlete_id, start_date, end_date)
        risk_summaries = await analysis_repo.get_risk_summaries_for_range(
            athlete_id, start_date, end_date
        )

        last_activity_date = entries[-1].date if entries else None
        latest_readiness = None
        latest_hrv = None
        for entry in reversed(entries):
            if entry.restitution:
                if latest_readiness is None:
                    latest_readiness = entry.restitution.readiness
                if latest_hrv is None:
                    latest_hrv = entry.restitution.hrv
            if latest_readiness is not None and latest_hrv is not None:
                break

        performance_risk = RiskCounts()
        restitution_risk = RiskCounts()
        overall_recovery_quality = None
        for summary in risk_summaries:
            for flag in summary["performance_risk_flags"]:
                setattr(
                    performance_risk,
                    flag["severity"],
                    getattr(performance_risk, flag["severity"]) + 1,
                )
            for flag in summary["restitution_risk_flags"]:
                setattr(
                    restitution_risk,
                    flag["severity"],
                    getattr(restitution_risk, flag["severity"]) + 1,
                )
            if summary["overall_recovery_quality"]:
                overall_recovery_quality = summary["overall_recovery_quality"]

        statuses.append(
            AthleteStatus(
                athlete_id=athlete_id,
                name=_athlete_name(athlete),
                profile_picture=athlete.profile_picture if athlete else None,
                last_activity_date=last_activity_date,
                latest_readiness=latest_readiness,
                latest_hrv=latest_hrv,
                performance_risk=performance_risk,
                restitution_risk=restitution_risk,
                overall_recovery_quality=overall_recovery_quality,
            )
        )

    return statuses


@router.get("/athletes/{athlete_id}/feed", response_model=list[FeedDay])
async def get_athlete_feed(
    start: str = Query(..., description="Start date YYYY-MM-DD (inclusive)"),
    end: str = Query(..., description="End date YYYY-MM-DD (inclusive)"),
    athlete_id: int = Depends(verify_coach_athlete_access),
    db: AsyncDatabase = Depends(get_db),
) -> list[FeedDay]:
    return await build_feed(db, athlete_id, start, end)


@router.get("/athletes/{athlete_id}/daily-analysis", response_model=DailyAnalysisResult | None)
async def get_athlete_daily_analysis(
    date: str = Query(..., description="Date YYYY-MM-DD"),
    athlete_id: int = Depends(verify_coach_athlete_access),
    db: AsyncDatabase = Depends(get_db),
) -> DailyAnalysisResult | None:
    analysis_repo = DailyAnalysisRepository(db)
    return await analysis_repo.get(athlete_id, date)
