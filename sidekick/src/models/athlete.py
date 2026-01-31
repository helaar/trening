from datetime import datetime, timezone
from pydantic import BaseModel, Field


class Athlete(BaseModel):
    """Athlete model representing a Strava user."""
    
    athlete_id: int = Field(..., description="Strava athlete ID")
    username: str | None = None
    firstname: str | None = None
    lastname: str | None = None
    profile_picture: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class StravaTokens(BaseModel):
    """Strava OAuth tokens model - stored separately from athlete data."""
    
    athlete_id: int = Field(..., description="Strava athlete ID")
    access_token: str
    refresh_token: str
    expires_at: int  # Unix timestamp
    token_type: str = "Bearer"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))