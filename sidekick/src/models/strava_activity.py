from datetime import datetime, timezone
from typing import Any
from pydantic import AwareDatetime, BaseModel, Field, field_validator


def _ensure_utc(v: Any) -> Any:
    if isinstance(v, datetime) and v.tzinfo is None:
        return v.replace(tzinfo=timezone.utc)
    return v


class StravaActivityRaw(BaseModel):
    """Raw Strava activity data as stored in the database."""

    athlete_id: int = Field(..., description="Athlete ID who owns this activity")
    activity_id: int = Field(..., description="Strava activity ID")
    activity_date: datetime = Field(..., description="Date of the activity (time zeroed)")
    raw_data: dict = Field(..., description="Raw activity data from Strava API")
    streams: dict[str, Any] | None = Field(None, description="Activity streams data (power, HR, cadence, etc.)")
    laps: list[dict[str, Any]] | None = Field(None, description="Activity laps data")
    created_at: AwareDatetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: AwareDatetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("created_at", "updated_at", mode="before")
    @classmethod
    def ensure_utc(cls, v: Any) -> Any:
        return _ensure_utc(v)


class WorkoutAnalysisData(BaseModel):
    """Workout analysis results stored in the database."""

    athlete_id: int = Field(..., description="Athlete ID who owns this activity")
    activity_id: int = Field(..., description="Strava activity ID")
    settings_hash: str = Field(..., description="Hash of athlete settings used for analysis")
    analysis_data: dict[str, Any] = Field(..., description="Serialized WorkoutAnalysis data")
    created_at: AwareDatetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: AwareDatetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("created_at", "updated_at", mode="before")
    @classmethod
    def ensure_utc(cls, v: Any) -> Any:
        return _ensure_utc(v)
