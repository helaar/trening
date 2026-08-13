"""
Pure analysis engine for workout data.

This module contains the core analysis logic extracted from strava_analyze.py.
All functions are pure computations that return structured data models without side effects.
No logging or output formatting is performed here - only data computation.
"""
from typing import Any, Literal
import pandas as pd

from clients.strava.client import StravaDataParser, StravaActivity
from models.athlete import AthleteSettings
from models.intervals_activity import IntervalsActivityRaw
from analysis.intervals_mapping import map_athlete_ftp, map_power_metrics, map_zone_analysis
from analysis.models import (
    WorkoutAnalysis, SessionInfo, WorkoutMetrics, StatsSummary,
    HeartRateDrift, LapAnalysis, ERGAnalysis, PowerHistogram
)

from analysis.calculations import (
    Zone, series_stats, parse_zone_definitions,
    compute_heart_rate_drift, infer_sample_interval,
    compute_segment_stats, split_into_autolaps, calculate_elevation
)

# Bump whenever engine logic changes meaningfully — forces cached WorkoutAnalysis
# documents (whose stored analysis_engine_version won't match) to recompute
# rather than serving stale locally-computed values forever, since the
# settings_hash cache-staleness field is advisory-only and never compared.
ANALYSIS_ENGINE_VERSION = 2


def _is_virtual_activity(activity: StravaActivity) -> bool:
    """
    Check if activity is from a virtual platform (Zwift, TrainerRoad, etc.).
    
    Args:
        activity: StravaActivity object
        
    Returns:
        True if activity is from a virtual platform
    """
    sport_type = str(activity.sport_type) if activity.sport_type else ''
    device = str(activity.device_name).lower() if activity.device_name else ''
    
    # Check sport type for virtual activities
    if 'Virtual' in sport_type:
        return True
    
    # Check device name for known virtual platforms
    virtual_platforms = ['zwift', 'trainerroad', 'rouvy', 'fulgaz', 'tacx', 'wahoo systm']
    return any(platform in device for platform in virtual_platforms)


def _detect_erg_lap(
    avg_power: float | None,
    np: float | None,
    power_stdev: float | None,
    threshold: float = 0.02,
    stdev_threshold: float = 0.10,
) -> bool:
    if not avg_power or not np:
        return False
    if avg_power < 50:
        return False
    if abs(np - avg_power) / avg_power > threshold:
        return False
    if power_stdev is not None and power_stdev / avg_power > stdev_threshold:
        return False
    return True


def _safe_float(value: Any) -> float | None:
    """Safely convert value to float or None."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _safe_int(value: Any) -> int | None:
    """Safely convert value to int or None."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def _create_stats_summary(stats_dict: dict[str, float | None]) -> StatsSummary:
    """Create StatsSummary from series_stats dict output."""
   
    return StatsSummary(
        mean=stats_dict.get("mean"),
        min=stats_dict.get("min"),
        max=stats_dict.get("max"),
        std=stats_dict.get("std"),
    )


class AnalysisSettings:
    """Thin wrapper providing convenient access to AthleteSettings for analysis."""
    
    def __init__(self, athlete_settings: AthleteSettings | None, workout_category: str):
        """
        Initialize analysis settings wrapper.
        
        Args:
            athlete_settings: AthleteSettings from database (holds reference, no copying)
            workout_category: Workout category (cycling, running, etc.)
        """
        self._settings = athlete_settings
        self._category = workout_category
        
        # ERG detection settings from athlete settings
        if self._settings and self._settings.erg_detection:
            self.erg_threshold = self._settings.erg_detection.threshold
            self.erg_min_ratio = self._settings.erg_detection.min_ratio
            self.erg_stdev_threshold = self._settings.erg_detection.stdev_threshold
        else:
            # Fallback to defaults if not configured
            self.erg_threshold = 0.02
            self.erg_min_ratio = 0.6
            self.erg_stdev_threshold = 0.10
        
        # Autolap setting from athlete settings (convert timedelta to seconds)
        if self._settings:
            autolap_td = self._settings.autolap_timedelta
            self.autolap_seconds = autolap_td.total_seconds() if autolap_td else None
        else:
            self.autolap_seconds = None
    
    @property
    def ftp(self) -> float | None:
        """Get FTP for the workout category."""
        if not self._settings:
            return None
        sport_settings = self._settings.get_sport_settings(self._category)
        return float(sport_settings.ftp) if sport_settings and sport_settings.ftp else None
    
    @property
    def power_zones(self) -> list[Zone] | None:
        """Get power zones for the workout category."""
        if not self._settings:
            return None
        sport_settings = self._settings.get_sport_settings(self._category)
        return parse_zone_definitions(sport_settings.power_zones) if sport_settings else None
    
    @property
    def max_hr(self) -> int | None:
        """Get maximum heart rate."""
        if not self._settings or not self._settings.heart_rate:
            return None
        return self._settings.heart_rate.max
    
    @property
    def lt_hr(self) -> int | None:
        """Get lactate threshold heart rate."""
        if not self._settings or not self._settings.heart_rate:
            return None
        return self._settings.heart_rate.lt


def _create_session_info(
    parser: StravaDataParser,
    duration_sec: float,
    data_points: int,
    sample_interval: float,
    df: pd.DataFrame | None = None,
) -> SessionInfo:
    """Create SessionInfo object from parser data."""
    elevation_gain_m = 0.0
    if df is not None and not df.empty:
        elevation_gain_m, _, _, _ = calculate_elevation(df)

    # Get commute status if available (defaults to "no")
    commute_raw = getattr(parser, 'commute_status', 'no')
    
    # Ensure the value is one of the valid literals
    commute_status: Literal["yes, marked by athlete", "yes, detected", "no"]
    if commute_raw == "yes, marked by athlete":
        commute_status = "yes, marked by athlete"
    elif commute_raw == "yes, detected":
        commute_status = "yes, detected"
    else:
        commute_status = "no"
    
    derived_tags: list[str] = []
    if commute_status != "no":
        derived_tags.append("commute")
    workout_type = getattr(parser.activity, "workout_type", None)
    if workout_type in {1, 11}:
        derived_tags.append("race")

    return SessionInfo(
        name=parser.workout.name,
        sport=parser.workout.sport,
        sub_sport=parser.workout.sub_sport,
        category=parser.workout.category,
        start_time=parser.workout.start_time,
        distance_km=parser.workout.distance_km,
        elevation_gain_m=elevation_gain_m,
        duration_sec=duration_sec,
        data_points=data_points,
        sample_interval=sample_interval,
        device_name=parser.activity.device_name,
        manual=parser.activity.manual,
        from_accepted_tag=parser.activity.from_accepted_tag,
        commute=commute_status,
        tags=derived_tags,
    )


def _compute_power_histogram(power_series: pd.Series, sample_interval: float,
                             bucket_width: float) -> PowerHistogram | None:
    """Bucket the raw power stream into fixed-width watt bins, valued in seconds.

    FTP-agnostic: a pure property of the ride. Buckets ascend from 0 W in steps of
    bucket_width; bucket i holds the time spent at [i*bucket_width, (i+1)*bucket_width).
    """
    if bucket_width <= 0:
        return None
    valid = power_series.dropna()
    valid = valid[valid >= 0]
    if valid.empty:
        return None

    bucket_index = (valid // bucket_width).astype(int)
    counts = bucket_index.value_counts()
    seconds = [0.0] * (int(bucket_index.max()) + 1)
    for idx, count in counts.items():
        seconds[int(idx)] = float(count) * sample_interval

    return PowerHistogram(
        bucket_width=float(bucket_width),
        min_watts=0.0,
        seconds=seconds,
        total_seconds=float(sum(seconds)),
    )


def _compute_heart_rate_drift_analysis(df: pd.DataFrame, drift_start: float | None,
                                      drift_duration: float | None, has_power: bool, has_hr: bool) -> HeartRateDrift | None:
    """Compute heart rate drift analysis."""
    if not has_power or not has_hr:
        return None
    
    drift_result = compute_heart_rate_drift(df, drift_start, drift_duration)
    
    if not drift_result:
        return None
    
    try:
        duration = _safe_float(drift_result.get('duration')) or 0.0
        avg_hr_p1 = _safe_float(drift_result.get('avg_hr_p1')) or 0.0
        avg_hr_p2 = _safe_float(drift_result.get('avg_hr_p2')) or 0.0
        avg_power_p1 = _safe_float(drift_result.get('avg_power_p1')) or 0.0
        avg_power_p2 = _safe_float(drift_result.get('avg_power_p2')) or 0.0
        hr_per_watt_p1 = _safe_float(drift_result.get('hr_per_watt_p1')) or 0.0
        hr_per_watt_p2 = _safe_float(drift_result.get('hr_per_watt_p2')) or 0.0
        drift_pct = _safe_float(drift_result.get('drift_pct')) or 0.0
        
        return HeartRateDrift(
            duration_sec=duration,
            avg_hr_p1=avg_hr_p1,
            avg_hr_p2=avg_hr_p2,
            avg_power_p1=avg_power_p1,
            avg_power_p2=avg_power_p2,
            hr_per_watt_p1=hr_per_watt_p1,
            hr_per_watt_p2=hr_per_watt_p2,
            drift_pct=drift_pct
        )
    except Exception:
        return None


def _analyze_laps(df: pd.DataFrame, laps: list[dict[str, Any]], settings: AnalysisSettings,
                 window: int, has_power: bool, is_virtual: bool = False) -> list[LapAnalysis]:
    """Analyze all laps and return LapAnalysis objects with raw numeric values."""
    lap_analyses = []
    
    # Map intensity names (though Strava doesn't provide this like Garmin)
    intensity_names = {
        0: "active",
        1: "rest", 
        2: "warmup",
        3: "cooldown",
        4: "recovery",
        5: "interval",
    }
    
    for idx, lap in enumerate(laps, start=1):
        start_ts = lap["start"]
        end_ts = lap["end"]
        lap_segment = df.loc[start_ts:end_ts].dropna(how="all")
        
        if lap_segment.empty:
            continue
        
        stats = compute_segment_stats(lap_segment, ftp=None, window=window)
        
        # Get lap metadata
        intensity = lap.get("intensity")
        intensity_str = intensity_names.get(intensity, str(intensity) if intensity is not None else "") if isinstance(intensity, int) else ""
        label = lap.get("label", "")
        
        # Get power zone if available
        zone = Zone.get_zone(settings.power_zones, stats["avg_power"]) if has_power and settings.power_zones and stats["avg_power"] else None
        zone_name = zone.name if zone else None
        
        # Create description from parts
        description_parts = [str(part) for part in (label, intensity_str, zone_name) if part]
        description = " / ".join(description_parts) if description_parts else None
        
        # Detect ERG mode for this lap — only valid for virtual rides
        is_erg = is_virtual and _detect_erg_lap(
            stats["avg_power"], stats["np"], stats["power_stdev"],
            threshold=settings.erg_threshold,
            stdev_threshold=settings.erg_stdev_threshold,
        )
        
        # Calculate start time in seconds from workout start
        start_time_sec = (start_ts - df.index[0]).total_seconds()
        
        lap_analysis = LapAnalysis(
            lap_number=idx,
            start_time_sec=start_time_sec,
            duration_sec=_safe_float(stats["duration_sec"]) or 0.0,
            distance_km=_safe_float(stats["distance"]),
            normalized_power=_safe_float(stats["np"]),
            avg_power=_safe_float(stats["avg_power"]),
            power_stdev=_safe_float(stats["power_stdev"]),
            max_power=_safe_float(stats["max_power"]),
            avg_heart_rate=_safe_float(stats["avg_hr"]),
            max_heart_rate=_safe_float(stats["max_hr"]),
            hr_drift_pct=_safe_float(stats["drift_pct"]),
            avg_cadence=_safe_float(stats["avg_cad"]),
            avg_speed_kph=_safe_float(stats["avg_speed"]),
            elevation_gain_m=_safe_float(stats["elev_gain"]),
            elevation_loss_m=_safe_float(stats["elev_loss"]),
            avg_temperature_c=_safe_float(stats["avg_temp"]),
            is_erg_mode=is_erg,
            intensity_type=intensity_str if intensity_str else None,
            label=label if label else None,
            power_zone=zone_name,
            description=description
        )
        
        lap_analyses.append(lap_analysis)
    
    return lap_analyses


def _compute_erg_analysis(lap_analyses: list[LapAnalysis], parser: StravaDataParser, 
                         settings: AnalysisSettings) -> ERGAnalysis | None:
    """Compute ERG mode analysis for virtual activities."""
    if not lap_analyses or not _is_virtual_activity(parser.activity):
        return None
    
    erg_laps_count = sum(1 for lap in lap_analyses if lap.is_erg_mode)
    total_laps_count = len(lap_analyses)
    erg_laps_length = sum(lap.duration_sec for lap in lap_analyses if lap.is_erg_mode)
    total_duration = sum(lap.duration_sec for lap in lap_analyses)
    
    if total_laps_count == 0:
        return None
    
    erg_ratio = max(erg_laps_count / total_laps_count, erg_laps_length / total_duration if total_duration > 0 else 0.0)
    is_erg_workout = erg_ratio >= settings.erg_min_ratio
    
    return ERGAnalysis(
        is_erg_workout=is_erg_workout,
        erg_laps_count=erg_laps_count,
        total_laps_count=total_laps_count,
        erg_time_sec=erg_laps_length,
        erg_ratio=erg_ratio,
        detection_threshold=settings.erg_threshold,
        min_ratio_threshold=settings.erg_min_ratio
    )


def analyze_endurance_workout(parser: StravaDataParser, athlete_settings: AthleteSettings | None,
                            window: int = 30, drift_start: float | None = None,
                            drift_duration: float | None = None, force_autolap: bool = False,
                            histogram_bucket_watts: float = 5.0,
                            intervals_activity: IntervalsActivityRaw | None = None) -> WorkoutAnalysis:
    """
    Analyze endurance workout data and return structured analysis results.
    
    Args:
        parser: StravaDataParser with workout data
        athlete_settings: AthleteSettings from database
        window: Window length for NP calculation (default 30 seconds)
        drift_start: Start point for heart rate drift analysis (seconds)
        drift_duration: Duration for heart rate drift analysis (seconds)
        force_autolap: Force autolap generation even if laps exist
        
    Returns:
        WorkoutAnalysis object with all computed metrics
    """
    df = parser.data_frame
    laps = parser.laps
    workout = parser.workout
    
    if df.empty:
        # No local stream data — Intervals.icu-sourced metrics/zones can still
        # be shown if a match exists, since those don't depend on the raw stream.
        session = _create_session_info(parser, 0.0, 0, 1.0)
        power_stats, np_value, vi_value, if_value, tss_value = map_power_metrics(intervals_activity)
        metrics = WorkoutMetrics(
            power=power_stats,
            normalized_power=np_value,
            variability_index=vi_value,
            intensity_factor=if_value,
            training_stress_score=tss_value,
            athlete_ftp=map_athlete_ftp(intervals_activity),
        )
        return WorkoutAnalysis(
            analysis_type="endurance",
            session=session,
            metrics=metrics,
            zones=map_zone_analysis(intervals_activity),
            has_power_data=False,
            has_heart_rate_data=False,
            has_cadence_data=False,
        )

    # Create settings wrapper
    analysis_settings = AnalysisSettings(athlete_settings, workout.category)
    
    # Calculate sample interval and duration
    sample_interval = infer_sample_interval(df.index) if isinstance(df.index, pd.DatetimeIndex) else 1.0
    if sample_interval <= 0:
        sample_interval = 1.0
    duration_sec = sample_interval * len(df)
    
    # Check available data
    has_power = "power" in df.columns and not df["power"].isna().all()
    has_hr = "heart_rate" in df.columns and not df["heart_rate"].isna().all()
    has_cadence = "cadence" in df.columns and not df["cadence"].isna().all()
    
    # Create session info
    session = _create_session_info(parser, duration_sec, len(df), sample_interval, df)

    # Whole-workout power metrics: Intervals.icu-sourced, no local fallback
    # (see analysis/intervals_mapping.py). athlete_ftp is sourced from
    # Intervals.icu too when available, so it stays consistent with the IF/TSS
    # shown alongside it rather than the locally-configured FTP.
    power_stats, np_value, vi_value, if_value, tss_value = map_power_metrics(intervals_activity)
    athlete_ftp = map_athlete_ftp(intervals_activity) or analysis_settings.ftp

    # Compute other basic stats
    hr_stats = _create_stats_summary(series_stats(df["heart_rate"], drop_nulls=True)) if has_hr else None
    cad_stats = _create_stats_summary(series_stats(df["cadence"], drop_nulls=True)) if has_cadence else None

    # Create metrics object
    metrics = WorkoutMetrics(
        power=power_stats,
        normalized_power=np_value,
        variability_index=vi_value,
        intensity_factor=if_value,
        training_stress_score=tss_value,
        heart_rate=hr_stats,
        cadence=cad_stats,
        athlete_ftp=athlete_ftp,
        athlete_max_hr=analysis_settings.max_hr,
        athlete_lt_hr=analysis_settings.lt_hr
    )

    # Zone distribution: Intervals.icu-sourced, no local fallback
    zones = map_zone_analysis(intervals_activity)

    # Fine-grained power histogram (FTP-agnostic) for sub-zone intensity analysis
    power_histogram = (
        _compute_power_histogram(df["power"], sample_interval, histogram_bucket_watts)
        if has_power else None
    )

    # Compute heart rate drift (single computation)
    hr_drift = _compute_heart_rate_drift_analysis(df, drift_start, drift_duration, has_power, has_hr)
    
    # Handle autolap generation
    if analysis_settings.autolap_seconds and (len(laps) <= 2 or force_autolap):
        autolaps = split_into_autolaps(df, analysis_settings.autolap_seconds)
        if autolaps:
            laps = autolaps
    
    # Analyze laps
    lap_analyses = _analyze_laps(df, laps, analysis_settings, window, has_power, is_virtual=_is_virtual_activity(parser.activity))
    
    # ERG mode analysis
    erg_analysis = _compute_erg_analysis(lap_analyses, parser, analysis_settings)
    
    return WorkoutAnalysis(
        analysis_type="endurance",
        session=session,
        metrics=metrics,
        zones=zones,
        power_histogram=power_histogram,
        laps=lap_analyses,
        heart_rate_drift=hr_drift,
        erg_analysis=erg_analysis,
        has_power_data=has_power,
        has_heart_rate_data=has_hr,
        has_cadence_data=has_cadence,
    )


def analyze_strength_workout(parser: StravaDataParser, athlete_settings: AthleteSettings | None,
                            intervals_activity: IntervalsActivityRaw | None = None) -> WorkoutAnalysis:
    """
    Analyze strength training workout data and return structured analysis results.

    Args:
        parser: StravaDataParser with workout data
        athlete_settings: AthleteSettings from database

    Returns:
        WorkoutAnalysis object with strength-specific metrics
    """
    df = parser.data_frame

    # Create settings wrapper (mainly heart rate for strength training)
    analysis_settings = AnalysisSettings(athlete_settings, "heart-rate")

    if df.empty:
        # Limited analysis without detailed stream data
        elapsed_time = _safe_float(parser.activity.elapsed_time) or 0.0
        session = _create_session_info(parser, elapsed_time, 0, 1.0)
        metrics = WorkoutMetrics()
        return WorkoutAnalysis(
            analysis_type="strength",
            session=session,
            metrics=metrics,
            zones=map_zone_analysis(intervals_activity),
            has_power_data=False,
            has_heart_rate_data=False,
            has_cadence_data=False,
        )

    # Calculate sample interval and duration
    sample_interval = infer_sample_interval(df.index) if isinstance(df.index, pd.DatetimeIndex) else 1.0
    if sample_interval <= 0:
        sample_interval = 1.0

    # Use elapsed_time for total duration (includes rest periods)
    duration_sec = _safe_float(parser.activity.elapsed_time) or (sample_interval * len(df))

    # Check available data
    has_hr = "heart_rate" in df.columns and not df["heart_rate"].isna().all()

    # Create session info
    session = _create_session_info(parser, duration_sec, len(df), sample_interval, df)

    # Compute heart rate stats
    hr_stats = _create_stats_summary(series_stats(df["heart_rate"])) if has_hr else None

    # Create metrics object (strength training typically doesn't have power)
    metrics = WorkoutMetrics(
        heart_rate=hr_stats,
        athlete_max_hr=analysis_settings.max_hr,
        athlete_lt_hr=analysis_settings.lt_hr
    )

    # Heart rate zone distribution: Intervals.icu-sourced, no local fallback
    zones = map_zone_analysis(intervals_activity)
    
    return WorkoutAnalysis(
        analysis_type="strength",
        session=session,
        metrics=metrics,
        zones=zones,
        has_power_data=False,
        has_heart_rate_data=has_hr,
        has_cadence_data=False,
    )


def analyze_workout(parser: StravaDataParser, athlete_settings: AthleteSettings | None,
                   window: int = 30, drift_start: float | None = None,
                   drift_duration: float | None = None, force_autolap: bool = False,
                   histogram_bucket_watts: float = 5.0,
                   intervals_activity: IntervalsActivityRaw | None = None) -> WorkoutAnalysis:
    """
    Main dispatcher function for workout analysis.

    Routes to appropriate analysis function based on workout category.

    Args:
        parser: StravaDataParser with workout data
        athlete_settings: AthleteSettings from database
        window: Window length for NP calculation (default 30 seconds)
        drift_start: Start point for heart rate drift analysis (seconds)
        drift_duration: Duration for heart rate drift analysis (seconds)
        force_autolap: Force autolap generation even if laps exist
        intervals_activity: Matched Intervals.icu activity, if any — sources
            whole-workout NP/IF/TSS/zone-distribution (no local fallback)

    Returns:
        WorkoutAnalysis object with computed metrics
    """
    match parser.workout.category:
        case "running" | "cycling" | "skiing":
            return analyze_endurance_workout(parser, athlete_settings, window, drift_start, drift_duration, force_autolap, histogram_bucket_watts, intervals_activity)
        case "strength":
            return analyze_strength_workout(parser, athlete_settings, intervals_activity)
        case _:
            # Default to endurance analysis for unknown categories
            return analyze_endurance_workout(parser, athlete_settings, window, drift_start, drift_duration, force_autolap, histogram_bucket_watts, intervals_activity)