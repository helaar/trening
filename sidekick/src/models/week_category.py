from datetime import datetime, timezone
from enum import Enum

from pydantic import AwareDatetime, BaseModel, Field, field_validator

from utils.datetime_utils import ensure_utc


class WeekCategory(str, Enum):
    BUILD = "build"
    MAINTAIN = "maintain"
    TAPER = "taper"
    RACE = "race"
    RESTITUTION = "restitution"
    RECOVERING = "recovering"


class WeekCategoryEntry(BaseModel):
    athlete_id: int
    week_start: str  # YYYY-MM-DD, Monday of the week
    category: WeekCategory
    created_at: AwareDatetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: AwareDatetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("created_at", "updated_at", mode="before")
    @classmethod
    def _utc(cls, v):
        return ensure_utc(v)


class WeekCategoryEntryRequest(BaseModel):
    week_start: str  # YYYY-MM-DD, Monday of the week
    category: WeekCategory
