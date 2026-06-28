from datetime import datetime, timezone

from pydantic import AwareDatetime, BaseModel, Field, field_validator

from utils.datetime_utils import ensure_utc


class Coach(BaseModel):
    """Coach model. A coach is a Strava user (by athlete_id) who can read a roster
    of other athletes' data. Coach-ness is a DB lookup, not a JWT claim."""

    coach_id: int = Field(..., description="Strava athlete ID of the coach")
    display_name: str | None = None
    athlete_ids: list[int] = Field(
        default_factory=list, description="Roster of athlete_ids this coach may read"
    )
    created_at: AwareDatetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: AwareDatetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("created_at", "updated_at", mode="before")
    @classmethod
    def _utc(cls, v):
        return ensure_utc(v)
