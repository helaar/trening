from datetime import datetime, timezone
from pydantic import BaseModel, Field


class Restitution(BaseModel):
    sleep_hours: float | None = None
    hrv: int | None = None
    resting_hr: int | None = None
    comment: str | None = None


class ActivityAssessment(BaseModel):
    activity_id: int
    activity_name: str
    rpe: int = Field(..., ge=1, le=10, description="Rate of Perceived Exertion (1-10)")
    notes: str | None = None


class DailyEntry(BaseModel):
    athlete_id: int
    date: str  # YYYY-MM-DD
    restitution: Restitution | None = None
    activity_assessments: list[ActivityAssessment] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class DailyEntryRequest(BaseModel):
    date: str  # YYYY-MM-DD
    restitution: Restitution | None = None
    activity_assessments: list[ActivityAssessment] = Field(default_factory=list)
