from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4

from pydantic import AwareDatetime, BaseModel, Field, field_validator

from utils.datetime_utils import ensure_utc


class PlannedActivity(BaseModel):
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
    created_at: AwareDatetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: AwareDatetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("created_at", "updated_at", mode="before")
    @classmethod
    def _utc(cls, v):
        return ensure_utc(v)


class PlannedActivityRequest(BaseModel):
    date: str  # YYYY-MM-DD
    sport: Literal["cycling", "running", "strength", "skiing_cross", "skiing_alpine", "day_off", "other"]
    name: str
    description: str | None = None
    purpose: str | None = None
    labels: list[str] = Field(default_factory=list)
    estimated_duration_min: int | None = None
    estimated_tss: int | None = None
    external_reference: str | None = None
