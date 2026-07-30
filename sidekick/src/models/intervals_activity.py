from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import AwareDatetime, BaseModel, Field, field_validator

from utils.datetime_utils import ensure_utc


class IntervalsActivityRaw(BaseModel):
    """Raw Intervals.icu activity data, cross-referenced to a Strava activity.

    Strava's activity_id stays the canonical key across the app. An Intervals.icu
    activity may reach us via a direct Strava-id back-reference, or — since the
    athlete's devices (Garmin, Zwift) upload to Strava and Intervals.icu
    independently rather than through a single chained hop — via fuzzy matching
    on start time/duration/route. match_method records which path produced the
    pairing so a wrong fuzzy match is auditable.
    """

    athlete_id: int = Field(..., description="Athlete ID who owns this activity")
    strava_activity_id: int | None = Field(
        None, description="Cross-referenced Strava activity ID, if matched"
    )
    intervals_activity_id: str = Field(..., description="Intervals.icu activity ID")
    match_method: Literal["strava_id", "fuzzy_polyline_date_duration", "ambiguous", "manual"]
    raw_data: dict[str, Any] = Field(..., description="Raw activity data from Intervals.icu API")
    intervals: list[dict[str, Any]] | None = Field(
        None, description="Per-interval/lap breakdown from Intervals.icu, if available"
    )
    fetched_at: AwareDatetime = Field(..., description="When this record was fetched from Intervals.icu")
    created_at: AwareDatetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: AwareDatetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("fetched_at", "created_at", "updated_at", mode="before")
    @classmethod
    def _ensure_utc(cls, v: Any) -> Any:
        return ensure_utc(v)
