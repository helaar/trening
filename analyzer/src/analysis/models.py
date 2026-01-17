"""
Data models for workout analysis results.

This module defines structured data models that represent the results of workout analysis,
separating data computation from formatting. All models store raw numeric values and are
designed to support both endurance and strength training workouts.
"""
from datetime import datetime
from typing import Literal
from pydantic import BaseModel, Field


class SessionInfo(BaseModel):
    """Basic workout session metadata."""
    
    name: str | None
    sport: str
    sub_sport: str | None
    category: Literal["cycling", "running", "skiing", "strength", "other"]
    start_time: datetime | None
    distance_km: float
    duration_sec: float
    data_points: int
    sample_interval: float
    
    # Data source information
    device_name: str | None = None
    manual: bool = False
    from_accepted_tag: bool = False
    commute: Literal["yes, marked by athlete", "yes, detected", "no"] = "no"


class StatsSummary(BaseModel):
    """Statistical summary for a metric (power, heart rate, cadence, etc.)."""
    
    mean: float | None = None
    min: float | None = None
    max: float | None = None
    std: float | None = None



class WorkoutMetrics(BaseModel):
    """Core workout metrics and derived calculations."""
    
    # Power metrics
    power: StatsSummary | None = None
    normalized_power: float | None = None
    variability_index: float | None = None
    intensity_factor: float | None = None
    training_stress_score: float | None = None
    
    # Heart rate metrics
    heart_rate: StatsSummary | None = None
    
    # Cadence metrics (RPM for cycling, SPM for running)
    cadence: StatsSummary | None = None
    
    # Speed metrics
    speed: StatsSummary | None = None
    
    # Athlete reference values
    athlete_ftp: float | None = None
    athlete_max_hr: int | None = None
    athlete_lt_hr: int | None = None


class ZoneInfo(BaseModel):
    """Information for a single zone."""
    name: str
    lower: float | None
    upper: float | None
    seconds: float
    percent: float


class ZoneDistribution(BaseModel):
    """Time distribution across zones for a single metric with zone metadata."""
    
    total_seconds: float = 0.0
    sample_interval: float = 1.0
    zones: list[ZoneInfo] = Field(default_factory=list)
    
    def get_zone_time(self, zone_number: int) -> float:
        """Get time spent in specific zone (1-based index)."""
        if 1 <= zone_number <= len(self.zones):
            return self.zones[zone_number - 1].seconds
        return 0.0
    
    def set_zone_info(self, zone_number: int, zone_info: ZoneInfo) -> None:
        """Set complete zone information (1-based index)."""
        # Extend zones list if needed
        while len(self.zones) < zone_number:
            self.zones.append(ZoneInfo(name="", lower=None, upper=None, seconds=0.0, percent=0.0))
        
        if 1 <= zone_number <= len(self.zones):
            self.zones[zone_number - 1] = zone_info
    
    def get_zone_info(self, zone_number: int) -> ZoneInfo | None:
        """Get complete zone information (1-based index)."""
        if 1 <= zone_number <= len(self.zones):
            return self.zones[zone_number - 1]
        return None


class ZoneAnalysis(BaseModel):
    """Time spent in different power and heart rate zones."""
    
    power_zones: ZoneDistribution | None = None
    heart_rate_zones: ZoneDistribution | None = None


class HeartRateDrift(BaseModel):
    """Heart rate drift analysis results."""
    
    duration_sec: float
    avg_hr_p1: float
    avg_hr_p2: float
    avg_power_p1: float
    avg_power_p2: float
    hr_per_watt_p1: float
    hr_per_watt_p2: float
    drift_pct: float


class LapAnalysis(BaseModel):
    """Analysis data for a single lap with raw numeric values."""
    
    lap_number: int
    start_time_sec: float  # seconds from workout start
    duration_sec: float
    distance_km: float | None = None
    
    # Power metrics
    normalized_power: float | None = None
    avg_power: float | None = None
    max_power: float | None = None
    
    # Heart rate metrics
    avg_heart_rate: float | None = None
    max_heart_rate: float | None = None
    hr_drift_pct: float | None = None
    
    # Cadence (RPM for cycling, SPM for running)
    avg_cadence: float | None = None
    
    # Speed and elevation
    avg_speed_kph: float | None = None
    elevation_gain_m: float | None = None
    elevation_loss_m: float | None = None
    
    # Environmental
    avg_temperature_c: float | None = None
    
    # ERG mode detection
    is_erg_mode: bool = False
    
    # Lap metadata
    intensity_type: str | None = None  # e.g., "warmup", "interval", "recovery"
    label: str | None = None
    power_zone: str | None = None
    description: str | None = None


class ERGAnalysis(BaseModel):
    """ERG mode detection analysis results."""
    
    is_erg_workout: bool
    erg_laps_count: int
    total_laps_count: int
    erg_time_sec: float
    erg_ratio: float
    detection_threshold: float = 0.02
    min_ratio_threshold: float = 0.6
    
    @property
    def erg_mode(self) -> str :
        """Get confidence level for ERG detection."""
        if self.erg_ratio >= 0.8:
            return "ON, detected for most of workout"
        elif self.erg_ratio >= 0.6:
            return "ON, detected for significant part of workout"
        elif self.erg_ratio >= 0.3:
            return "ON, detected for some parts of workout"
        elif self.erg_ratio >= 0.1:
            return "Possibly ON, weak indications"
        else:
            return "OFF"


class WorkoutAnalysis(BaseModel):
    """Top-level container for complete workout analysis results."""
    
    # Required fields (no defaults) must come first
    analysis_type: Literal["endurance", "strength"]
    session: SessionInfo
    metrics: WorkoutMetrics
    
    # Optional fields with defaults
    analysis_timestamp: datetime = Field(default_factory=datetime.now)
    zones: ZoneAnalysis | None = None
    
    # Detailed analysis
    laps: list[LapAnalysis] = Field(default_factory=list)
    heart_rate_drift: HeartRateDrift | None = None
    erg_analysis: ERGAnalysis | None = None
    
    # Raw data characteristics
    has_power_data: bool = False
    has_heart_rate_data: bool = False
    has_cadence_data: bool = False
    has_speed_data: bool = False
    
    @property
    def is_virtual_activity(self) -> bool:
        """Check if this is a virtual/indoor activity."""
        device = (self.session.device_name or "").lower()
        virtual_platforms = ['zwift', 'trainerroad', 'rouvy', 'fulgaz', 'tacx', 'wahoo systm']
        return any(platform in device for platform in virtual_platforms)
    
    @property
    def workout_duration_formatted(self) -> str:
        """Get formatted duration string (HH:MM:SS)."""
        duration = int(self.session.duration_sec)
        hours = duration // 3600
        minutes = (duration % 3600) // 60
        seconds = duration % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    @property
    def total_lap_count(self) -> int:
        """Get total number of laps."""
        return len(self.laps)
    
    @property
    def erg_lap_count(self) -> int:
        """Get number of laps detected as ERG mode."""
        return sum(1 for lap in self.laps if lap.is_erg_mode)
    
    def get_lap_by_number(self, lap_number: int) -> LapAnalysis | None:
        """Get lap by lap number."""
        for lap in self.laps:
            if lap.lap_number == lap_number:
                return lap
        return None
    
    def get_power_laps(self) -> list[LapAnalysis]:
        """Get laps with power data."""
        return [lap for lap in self.laps if lap.avg_power is not None]
    
    def get_interval_laps(self) -> list[LapAnalysis]:
        """Get laps marked as intervals."""
        return [lap for lap in self.laps if lap.intensity_type == "interval"]