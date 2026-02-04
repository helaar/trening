from datetime import datetime, date
from pydantic import BaseModel, Field


class StravaActivityRaw(BaseModel):
    """Raw Strava activity data as stored in the database."""
    
    athlete_id: int = Field(..., description="Athlete ID who owns this activity")
    activity_id: int = Field(..., description="Strava activity ID")
    activity_date: date = Field(..., description="Date of the activity")
    raw_data: dict = Field(..., description="Raw activity data from Strava API")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            date: lambda v: v.isoformat()
        }
