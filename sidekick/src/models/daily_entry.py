from datetime import datetime, timezone
from pydantic import AwareDatetime, BaseModel, Field, field_validator
from utils.datetime_utils import ensure_utc


class Restitution(BaseModel):
    sleep_hours: float | None = None
    sleep_quality: float | None = Field(None, ge=1, le=5)
    hrv: int | None = None
    resting_hr: int | None = None
    readiness: float | None = Field(None, ge=1, le=5)
    comment: str | None = None


class RestitutionSync(BaseModel):
    """One day's objective restitution fields sourced live from Intervals.icu.

    Not persisted — used to fill gaps in a manually-entered Restitution. No
    equivalent for sleep_quality/readiness/comment: those are the athlete's
    own subjective ratings, which Intervals.icu doesn't provide.
    """

    date: str  # YYYY-MM-DD
    sleep_hours: float | None = None
    hrv: int | None = None
    resting_hr: int | None = None


class ActivityAssessment(BaseModel):
    activity_id: int
    activity_name: str
    rpe: int = Field(..., ge=1, le=10, description="Rate of Perceived Exertion (1-10)")
    notes: str | None = None
    tags: list[str] = Field(default_factory=list)


class DailyEntry(BaseModel):
    athlete_id: int
    date: str  # YYYY-MM-DD
    restitution: Restitution | None = None
    activity_assessments: list[ActivityAssessment] = Field(default_factory=list)
    created_at: AwareDatetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: AwareDatetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("created_at", "updated_at", mode="before")
    @classmethod
    def _utc(cls, v):
        return ensure_utc(v)


class DailyEntryRequest(BaseModel):
    date: str  # YYYY-MM-DD
    restitution: Restitution | None = None
    activity_assessments: list[ActivityAssessment] = Field(default_factory=list)
