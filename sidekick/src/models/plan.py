from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4

from pydantic import AwareDatetime, BaseModel, Field, field_validator

from utils.datetime_utils import ensure_utc


class PlannedActivity(BaseModel):
    """A planned workout. Read-only: sourced live from the athlete's Intervals.icu
    calendar (see services/intervals_calendar.py) — sidekick has no storage or
    authoring UI of its own for these.
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    athlete_id: int
    date: str  # YYYY-MM-DD
    sport: Literal["cycling", "running", "strength", "skiing_cross", "skiing_alpine", "day_off", "other"]
    name: str
    description: str | None = None
    purpose: str | None = None
    labels: list[str] = Field(default_factory=list)
    estimated_duration_min: int | None = None
    estimated_tss: int | None = None
    external_reference: str | None = None
    race_priority: Literal["A", "B", "C"] | None = None
    matched_activity_id: int | None = Field(
        default=None,
        description=(
            "Strava activity id of the completed workout that fulfilled this "
            "plan, computed at read time from Intervals.icu's own event/activity "
            "pairing (see services/plan_matching.py). Not sourced from the "
            "Intervals.icu event payload itself, so absent whenever the caller "
            "didn't run the matcher."
        ),
    )
    created_at: AwareDatetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: AwareDatetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("created_at", "updated_at", mode="before")
    @classmethod
    def _utc(cls, v):
        return ensure_utc(v)
