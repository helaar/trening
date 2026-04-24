from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field


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
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class PlannedActivityRequest(BaseModel):
    date: str  # YYYY-MM-DD
    sport: Literal["cycling", "running", "strength", "skiing_cross", "skiing_alpine", "day_off", "other"]
    name: str
    description: str | None = None
    purpose: str | None = None
    labels: list[str] = Field(default_factory=list)
    estimated_duration_min: int | None = None
    estimated_tss: int | None = None
