
"""
"""


from typing import Literal
from datetime import datetime
from pydantic import BaseModel, PrivateAttr, computed_field, field_validator


class Sensor(BaseModel):
    name: str
    manufacturer: str

class Device(BaseModel):
    _product: str | None = PrivateAttr(None)
    _manufacturer: str | None = PrivateAttr(None)
    _sensors: list[Sensor] = PrivateAttr(default_factory=list)

    @computed_field
    def device(self) -> str:
        prod = self._product if self._product != self._manufacturer else ""
        sensor_str = "med "+self.sensors if self.sensors is not None else ""
        return  f"{self._manufacturer} {prod} {sensor_str}"


    @property
    def sensors(self) -> str | None:
        if not self._sensors:
            return None
        else:
            ret = []
            for s in self._sensors:
                if s.manufacturer is None or s.manufacturer == self._manufacturer:
                    ret.append(s.name)
                else:
                    ret.append(f"{s.manufacturer} {s.name}")
            return ", ".join(ret)


class Workout(BaseModel):
    name: str | None = None
    start_time: datetime | None = None
    sport: str
    sub_sport: str | None = None
    
    device: Device
    _distance: float = PrivateAttr(0.0)
    
    @field_validator('start_time', mode='before')
    @classmethod
    def parse_start_time(cls, v):
        if v is None:
            return None
        if isinstance(v, datetime):
            return v
        if isinstance(v, str):
            # Try to parse various datetime string formats
            try:
                # Handle ISO format with Z timezone
                if v.endswith('Z'):
                    return datetime.fromisoformat(v.replace('Z', '+00:00'))
                # Handle standard ISO format
                return datetime.fromisoformat(v)
            except ValueError:
                try:
                    # Try common format
                    return datetime.strptime(v, '%Y-%m-%d %H:%M:%S')
                except ValueError:
                    try:
                        # Try another common format
                        return datetime.strptime(v, '%Y-%m-%dT%H:%M:%S')
                    except ValueError:
                        # If all parsing fails, return None
                        return None
        # For any other type, return None
        return None
    

    @computed_field
    def distance(self) -> str:
        return f"{self.distance_km:.2f} km"
    
    @property
    def distance_km(self) -> float:
        return self._distance / 1000.0

    @property
    def category(self) -> Literal["cycling","running","skiing","strength","other"] :
        match self.sport:
            case "cycling" | "running":
                return self.sport
            
            case "training":
                if self.sub_sport and "strength" in self.sub_sport:
                    return "strength"
                else:
                    return "other"
            case "cross_country_skiing":
                return "skiing"
            case _:
                return "other"

class ActivitySummary(BaseModel):
    """Summary metrics for a single activity, calculated from Strava API data."""
    
    id: int
    name: str
    sport: str
    sport_type: str
    date: datetime
    start_time: datetime
    
    # Duration metrics
    moving_time: int  # seconds
    elapsed_time: int  # seconds
    
    # Distance and elevation
    distance: float  # meters
    elevation_gain: float | None = None  # meters
    
    # Power metrics (when available)
    average_watts: float | None = None
    weighted_average_watts: float | None = None  # Strava's NP equivalent
    max_watts: float | None = None
    kilojoules: float | None = None
    
    # Heart rate metrics (when available)
    average_heartrate: float | None = None
    max_heartrate: float | None = None
    
    # Speed and cadence
    average_speed: float | None = None  # m/s
    max_speed: float | None = None
    average_cadence: float | None = None
    
    # Calculated training load metrics
    tss: float | None = None
    hrss: float | None = None
    intensity_factor: float | None = None
    hr_intensity_factor: float | None = None
    
    @field_validator('date', mode='before')
    @classmethod
    def parse_date(cls, v):
        if isinstance(v, datetime):
            # If it's already a datetime, zero out the time portion
            return v.replace(hour=0, minute=0, second=0, microsecond=0)
        if isinstance(v, str):
            # Parse string and ensure time is zeroed
            dt = datetime.fromisoformat(v)
            return dt.replace(hour=0, minute=0, second=0, microsecond=0)
        return v
    
    @classmethod
    def from_strava_activity(cls, activity) -> 'ActivitySummary':
        """Create ActivitySummary from StravaActivity."""
        return cls(
            id=activity.id,
            name=activity.name,
            sport=activity.type,
            sport_type=activity.sport_type,
            date=activity.start_date_local.replace(hour=0, minute=0, second=0, microsecond=0),
            start_time=activity.start_date_local,
            moving_time=activity.moving_time,
            elapsed_time=activity.elapsed_time,
            distance=activity.distance or 0.0,
            elevation_gain=activity.total_elevation_gain,
            average_watts=activity.average_watts,
            weighted_average_watts=activity.weighted_average_watts,
            max_watts=activity.max_watts,
            kilojoules=activity.kilojoules,
            average_heartrate=activity.average_heartrate,
            max_heartrate=activity.max_heartrate,
            average_speed=activity.average_speed,
            max_speed=activity.max_speed,
            average_cadence=activity.average_cadence
        )

    def calculate_training_load(self, cycling_ftp: float | None = None, running_ftp: float | None = None, threshold_hr: float | None = None):
        """Calculate TSS and HRSS for this activity using sport-specific FTP."""
        if self.weighted_average_watts:
            # Use sport-specific FTP
            ftp = None
            if self.sport.lower() in ['ride', 'virtualride'] and cycling_ftp:
                ftp = cycling_ftp
            elif self.sport.lower() == 'run' and running_ftp:
                ftp = running_ftp
            
            if ftp:
                self.tss = calculate_tss(self.moving_time, self.weighted_average_watts, ftp)
                self.intensity_factor = calculate_intensity_factor(self.weighted_average_watts, ftp)
        
        if self.average_heartrate and threshold_hr:
            self.hrss = calculate_hrss(self.moving_time, self.average_heartrate, threshold_hr)
            self.hr_intensity_factor = calculate_hr_intensity_factor(self.average_heartrate, threshold_hr)

    @computed_field
    @property
    def distance_km(self) -> float:
        """Distance in kilometers."""
        return self.distance / 1000.0

    @computed_field  
    @property
    def moving_time_hours(self) -> float:
        """Moving time in hours."""
        return self.moving_time / 3600.0

    @computed_field
    @property
    def pace_min_per_km(self) -> float | None:
        """Pace in minutes per kilometer (for running activities)."""
        if self.distance_km > 0 and self.moving_time > 0:
            return self.moving_time / 60 / self.distance_km
        return None


class DailySummary(BaseModel):
    """Summary of all activities for a single day."""
    
    date: datetime
    activities: list[ActivitySummary] = []
    
    @computed_field
    @property
    def activity_count(self) -> int:
        """Number of activities for this day."""
        return len(self.activities)
    
    @computed_field
    @property
    def total_time(self) -> int:
        """Total moving time in seconds."""
        return sum(activity.moving_time for activity in self.activities)
    
    @computed_field
    @property
    def total_distance(self) -> float:
        """Total distance in meters."""
        return sum(activity.distance for activity in self.activities)
    
    @computed_field
    @property
    def total_elevation(self) -> float:
        """Total elevation gain in meters."""
        return sum(activity.elevation_gain or 0.0 for activity in self.activities)
    
    @computed_field
    @property
    def total_tss(self) -> float:
        """Total Training Stress Score for the day."""
        return sum(activity.tss or 0.0 for activity in self.activities)
    
    @computed_field
    @property
    def total_hrss(self) -> float:
        """Total Heart Rate Stress Score for the day."""
        return sum(activity.hrss or 0.0 for activity in self.activities)
    
    @computed_field
    @property
    def total_kilojoules(self) -> float:
        """Total energy expenditure in kilojoules."""
        return sum(activity.kilojoules or 0.0 for activity in self.activities)
    
    @computed_field
    @property
    def sports_breakdown(self) -> dict[str, int]:
        """Count of activities by sport."""
        breakdown = {}
        for activity in self.activities:
            sport = activity.sport
            breakdown[sport] = breakdown.get(sport, 0) + 1
        return breakdown

    @computed_field
    @property
    def is_training_day(self) -> bool:
        """Whether this was a training day (has activities)."""
        return len(self.activities) > 0


def calculate_tss(moving_time_sec: int, weighted_avg_watts: float, ftp: float) -> float:
    """
    Calculate Training Stress Score from summary data.
    
    Args:
        moving_time_sec: Moving time in seconds
        weighted_avg_watts: Strava's weighted average watts (approximates NP)
        ftp: Functional Threshold Power
    
    Returns:
        Training Stress Score
    """
    if not weighted_avg_watts or not ftp or ftp <= 0:
        return 0.0
    
    intensity_factor = weighted_avg_watts / ftp
    hours = moving_time_sec / 3600
    
    return intensity_factor * intensity_factor * hours * 100


def calculate_hrss(moving_time_sec: int, avg_hr: float, threshold_hr: float) -> float:
    """
    Calculate Heart Rate Stress Score from summary data.
    
    Args:
        moving_time_sec: Moving time in seconds
        avg_hr: Average heart rate
        threshold_hr: Lactate threshold heart rate
    
    Returns:
        Heart Rate Stress Score
    """
    if not avg_hr or not threshold_hr or threshold_hr <= 0:
        return 0.0
    
    intensity_factor = avg_hr / threshold_hr
    hours = moving_time_sec / 3600
    
    return intensity_factor * intensity_factor * hours * 100


def calculate_intensity_factor(weighted_avg_watts: float, ftp: float) -> float:
    """Calculate power-based intensity factor."""
    if not weighted_avg_watts or not ftp or ftp <= 0:
        return 0.0
    return weighted_avg_watts / ftp


def calculate_hr_intensity_factor(avg_hr: float, threshold_hr: float) -> float:
    """Calculate heart rate-based intensity factor."""
    if not avg_hr or not threshold_hr or threshold_hr <= 0:
        return 0.0
    return avg_hr / threshold_hr
