from datetime import datetime
from typing import Any
from pydantic import BaseModel, Field


class StravaActivityRaw(BaseModel):
    """Raw Strava activity data as stored in the database."""
    
    athlete_id: int = Field(..., description="Athlete ID who owns this activity")
    activity_id: int = Field(..., description="Strava activity ID")
    activity_date: datetime = Field(..., description="Date of the activity (time zeroed)")
    raw_data: dict = Field(..., description="Raw activity data from Strava API")
    streams: dict[str, Any] | None = Field(None, description="Activity streams data (power, HR, cadence, etc.)")
    laps: list[dict[str, Any]] | None = Field(None, description="Activity laps data")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class WorkoutAnalysisData(BaseModel):
    """Workout analysis results stored in the database."""
    
    athlete_id: int = Field(..., description="Athlete ID who owns this activity")
    activity_id: int = Field(..., description="Strava activity ID")
    settings_hash: str = Field(..., description="Hash of athlete settings used for analysis")
    analysis_data: dict[str, Any] = Field(..., description="Serialized WorkoutAnalysis data")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
