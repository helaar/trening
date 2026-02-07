from datetime import datetime, timezone, timedelta
from pydantic import BaseModel, Field


class ZoneDefinition(BaseModel):
    """Single zone definition with name and range."""
    name: str
    min: int
    max: int | None = None  # None means no upper limit
    
    @classmethod
    def from_yaml_string(cls, yaml_str: str) -> "ZoneDefinition":
        """Parse zone from YAML format like 'Recovery: 0-115' or 'Neuromuscular: 391-'"""
        name, range_str = yaml_str.split(":", 1)
        name = name.strip()
        range_str = range_str.strip()
        
        if "-" in range_str:
            parts = range_str.split("-")
            min_val = int(parts[0])
            max_val = int(parts[1]) if parts[1] else None
        else:
            min_val = int(range_str)
            max_val = None
        
        return cls(name=name, min=min_val, max=max_val)


class HeartRateSettings(BaseModel):
    """Heart rate settings and zones."""
    max: int = Field(..., description="Maximum heart rate")
    lt: int = Field(..., description="Lactate threshold heart rate")
    measured_activity: str | None = Field(None, description="Activity type where measured")
    measured_date: str | None = Field(None, description="Date when measured (YYYY-MM-DD)")
    hr_zones: list[ZoneDefinition] = Field(default_factory=list, description="HR zone definitions")


class SportSettings(BaseModel):
    """Sport-specific settings (cycling or running)."""
    ftp: int = Field(..., description="Functional Threshold Power")
    measured_activity: str | None = Field(None, description="Activity type where FTP was measured")
    measured_date: str | None = Field(None, description="Date when measured (YYYY-MM-DD)")
    power_zones: list[ZoneDefinition] = Field(default_factory=list, description="Power zone definitions")


class ERGDetectionSettings(BaseModel):
    """Settings for ERG mode detection in virtual workouts."""
    threshold: float = Field(default=0.02, description="Maximum relative difference between NP and avg power for ERG detection")
    min_ratio: float = Field(default=0.6, description="Minimum ratio of ERG laps/time to classify workout as ERG")


class AthleteSettings(BaseModel):
    """Athlete-specific settings including zones and FTP values."""
    heart_rate: HeartRateSettings | None = None
    cycling: SportSettings | None = None
    running: SportSettings | None = None
    commute_routes: dict[str, str] = Field(default_factory=dict, description="Map of route name to polyline")
    erg_detection: ERGDetectionSettings = Field(default_factory=ERGDetectionSettings, description="ERG mode detection settings")
    autolap: timedelta | None = Field(default=timedelta(minutes=10), description="Autolap interval for consistent interval analysis (default: 10 minutes). Set to None to disable.")
    
    def get_sport_settings(self, category: str) -> SportSettings | None:
        """
        Get sport-specific settings for a given category.
        
        Args:
            category: Sport category ('cycling', 'running', etc.)
            
        Returns:
            SportSettings for the category or None if not configured
        """
        if category == "cycling":
            return self.cycling
        elif category == "running":
            return self.running
        return None


class Athlete(BaseModel):
    """Athlete model representing a Strava user."""
    
    athlete_id: int = Field(..., description="Strava athlete ID")
    username: str | None = None
    firstname: str | None = None
    lastname: str | None = None
    profile_picture: str | None = None
    settings: AthleteSettings = Field(default_factory=AthleteSettings, description="Athlete training settings")
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