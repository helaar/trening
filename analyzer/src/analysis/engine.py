"""
Pure analysis engine for workout data.

This module contains the core analysis logic extracted from strava_analyze.py.
All functions are pure computations that return structured data models without side effects.
No logging or output formatting is performed here - only data computation.
"""
from datetime import datetime
from typing import Any, cast
import pandas as pd

from strava.client import StravaDataParser, StravaActivity
from analysis.models import (
    ZoneInfo, WorkoutAnalysis, SessionInfo, WorkoutMetrics, StatsSummary, ZoneAnalysis, 
    ZoneDistribution, HeartRateDrift, LapAnalysis, ERGAnalysis
)

from tools.calculations import (
    Zone, normalized_power, intensity_factor, training_stress_score,
    series_stats, parse_zone_definitions, compute_zone_durations,
    compute_heart_rate_drift, infer_sample_interval,
    compute_segment_stats, parse_iso8601_duration, split_into_autolaps
)


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


def _detect_erg_lap(avg_power: float | None, np: float | None, threshold: float = 0.02) -> bool:
    """
    Detect if a lap was likely performed in ERG mode based on power consistency.
    
    ERG mode maintains constant power, so Normalized Power ≈ Average Power.
    
    Args:
        avg_power: Average power for the lap (can be None)
        np: Normalized power for the lap (can be None)
        threshold: Maximum allowed relative difference (default 2%)
        
    Returns:
        True if lap shows ERG mode characteristics
    """
    # Skip laps with no power data
    if not avg_power or not np:
        return False
    
    # Skip very low power laps (likely coasting/stopped)
    if avg_power < 50:
        return False
    
    # Calculate relative difference between NP and Avg Power
    consistency_score = abs(np - avg_power) / avg_power
    return consistency_score <= threshold


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
    count_val = stats_dict.get("count", 0)
    count = int(count_val) if count_val is not None else 0
    
    return StatsSummary(
        mean=stats_dict.get("mean"),
        min=stats_dict.get("min"),
        max=stats_dict.get("max"),
        std=stats_dict.get("std"),
        q25=stats_dict.get("q25"),
        q50=stats_dict.get("q50"),
        q75=stats_dict.get("q75"),
        count=count
    )


class AnalysisSettings:
    """Type-safe container for analysis settings."""
    
    def __init__(self, settings: dict[str, Any], workout_category: str):
        # Category-specific settings
        cat_settings = settings.get(workout_category, {})
        if isinstance(cat_settings, dict):
            self.ftp = _safe_float(cat_settings.get("ftp"))
            power_zones_raw = cat_settings.get("power-zones")
            self.power_zones = parse_zone_definitions(power_zones_raw) if power_zones_raw else None
        else:
            self.ftp = None
            self.power_zones = None
        
        # Heart rate settings
        hr_settings = settings.get("heart-rate", {})
        if isinstance(hr_settings, dict):
            self.max_hr = _safe_int(hr_settings.get("max"))
            self.lt_hr = _safe_int(hr_settings.get("lt"))
            hr_zones_raw = hr_settings.get("hr-zones")
            self.hr_zones = parse_zone_definitions(hr_zones_raw) if hr_zones_raw else None
        else:
            self.max_hr = None
            self.lt_hr = None
            self.hr_zones = None
        
        # ERG detection settings
        erg_settings = settings.get("erg-detection", {})
        if isinstance(erg_settings, dict):
            self.erg_threshold = _safe_float(erg_settings.get("threshold")) or 0.02
            self.erg_min_ratio = _safe_float(erg_settings.get("min-ratio")) or 0.6
        else:
            self.erg_threshold = 0.02
            self.erg_min_ratio = 0.6
        
        # Autolap setting
        autolap = settings.get("autolap")
        self.autolap_seconds = None
        if autolap:
            try:
                self.autolap_seconds = parse_iso8601_duration(str(autolap))
            except Exception:
                self.autolap_seconds = None


def _create_session_info(parser: StravaDataParser, duration_sec: float, data_points: int, sample_interval: float) -> SessionInfo:
    """Create SessionInfo object from parser data."""
    return SessionInfo(
        name=parser.workout.name,
        sport=parser.workout.sport,
        sub_sport=parser.workout.sub_sport,
        category=parser.workout.category,
        start_time=parser.workout.start_time,
        distance_km=parser.workout.distance_km,
        duration_sec=duration_sec,
        data_points=data_points,
        sample_interval=sample_interval,
        device_name=parser.activity.device_name,
        manual=parser.activity.manual,
        from_accepted_tag=parser.activity.from_accepted_tag
    )


def _compute_power_metrics(df: pd.DataFrame, settings: AnalysisSettings, window: int, has_power: bool) -> tuple[StatsSummary | None, float | None, float | None, float | None, float | None]:
    """
    Compute all power-related metrics in one pass.
    
    Returns:
        Tuple of (power_stats, np_value, if_value, tss_value, vi_value)
    """
    if not has_power:
        return None, None, None, None, None
    
    power_stats_dict = series_stats(df["power"], drop_nulls=True)
    power_stats = _create_stats_summary(power_stats_dict)
    
    try:
        np_value = normalized_power(df["power"], window=window)
        if_value = None
        tss_value = None
        
        if np_value and settings.ftp:
            if_value = intensity_factor(np_value, settings.ftp)
            sample_interval = infer_sample_interval(df.index) if isinstance(df.index, pd.DatetimeIndex) else 1.0
            duration_sec = len(df) * sample_interval
            tss_value = training_stress_score(duration_sec, np_value, if_value, settings.ftp)
        
        vi_value = None
        avg_power = df["power"].dropna().mean()
        if np_value and avg_power and avg_power > 0:
            vi_value = np_value / avg_power
            
        return power_stats, np_value, if_value, tss_value, vi_value
    except Exception:
        return power_stats, None, None, None, None


def _compute_zone_analysis(df: pd.DataFrame, settings: AnalysisSettings,
                          sample_interval: float, has_power: bool, has_hr: bool) -> ZoneAnalysis:
    """Compute zone distributions for power and heart rate."""   
    zones = ZoneAnalysis()
    
    if settings.power_zones and has_power:
        power_summary = compute_zone_durations(df["power"], settings.power_zones, sample_interval)
        power_dist = ZoneDistribution()
        power_dist.total_seconds = _safe_float(power_summary.get("total_seconds", 0.0)) or 0.0
        power_dist.sample_interval = _safe_float(power_summary.get("sample_interval", 1.0)) or 1.0
        
        # Extract zone data from the zones list
        zone_data = power_summary.get("zones", [])
        if isinstance(zone_data, list):
            for zone_num, zone_info in enumerate(zone_data, start=1):
                if zone_num <= 7 and isinstance(zone_info, dict):
                    zone_obj = ZoneInfo(
                        name=zone_info.get("name", f"Zone {zone_num}"),
                        lower=zone_info.get("lower"),
                        upper=zone_info.get("upper"),
                        seconds=_safe_float(zone_info.get("seconds", 0.0)) or 0.0,
                        percent=_safe_float(zone_info.get("percent", 0.0)) or 0.0
                    )
                    power_dist.set_zone_info(zone_num, zone_obj)
        zones.power_zones = power_dist
    
    if settings.hr_zones and has_hr:
        hr_summary = compute_zone_durations(df["heart_rate"], settings.hr_zones, sample_interval)
        hr_dist = ZoneDistribution()
        hr_dist.total_seconds = _safe_float(hr_summary.get("total_seconds", 0.0)) or 0.0
        hr_dist.sample_interval = _safe_float(hr_summary.get("sample_interval", 1.0)) or 1.0
        
        # Extract zone data from the zones list
        zone_data = hr_summary.get("zones", [])
        if isinstance(zone_data, list):
            for zone_num, zone_info in enumerate(zone_data, start=1):
                if zone_num <= 7 and isinstance(zone_info, dict):
                    zone_obj = ZoneInfo(
                        name=zone_info.get("name", f"Zone {zone_num}"),
                        lower=zone_info.get("lower"),
                        upper=zone_info.get("upper"),
                        seconds=_safe_float(zone_info.get("seconds", 0.0)) or 0.0,
                        percent=_safe_float(zone_info.get("percent", 0.0)) or 0.0
                    )
                    hr_dist.set_zone_info(zone_num, zone_obj)
        zones.heart_rate_zones = hr_dist
    
    return zones


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
                 window: int, has_power: bool) -> list[LapAnalysis]:
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
        
        # Detect ERG mode for this lap using configured threshold
        is_erg = _detect_erg_lap(stats["avg_power"], stats["np"], threshold=settings.erg_threshold)
        
        # Calculate start time in seconds from workout start
        start_time_sec = (start_ts - df.index[0]).total_seconds()
        
        lap_analysis = LapAnalysis(
            lap_number=idx,
            start_time_sec=start_time_sec,
            duration_sec=_safe_float(stats["duration_sec"]) or 0.0,
            distance_km=_safe_float(stats["distance"]),
            normalized_power=_safe_float(stats["np"]),
            avg_power=_safe_float(stats["avg_power"]),
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
    
    if total_laps_count == 0:
        return None
    
    erg_ratio = erg_laps_count / total_laps_count
    is_erg_workout = erg_ratio >= settings.erg_min_ratio
    
    return ERGAnalysis(
        is_erg_workout=is_erg_workout,
        erg_laps_count=erg_laps_count,
        total_laps_count=total_laps_count,
        erg_ratio=erg_ratio,
        detection_threshold=settings.erg_threshold,
        min_ratio_threshold=settings.erg_min_ratio
    )


def analyze_endurance_workout(parser: StravaDataParser, settings: dict[str, Any], 
                            window: int = 30, drift_start: float | None = None, 
                            drift_duration: float | None = None, force_autolap: bool = False) -> WorkoutAnalysis:
    """
    Analyze endurance workout data and return structured analysis results.
    
    Args:
        parser: StravaDataParser with workout data
        settings: Analysis settings from configuration
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
        # Limited analysis without detailed stream data
        session = _create_session_info(parser, 0.0, 0, 1.0)
        metrics = WorkoutMetrics()
        return WorkoutAnalysis(
            analysis_type="endurance",
            session=session,
            metrics=metrics,
            has_power_data=False,
            has_heart_rate_data=False,
            has_cadence_data=False,
            has_speed_data=False
        )
    
    # Parse settings
    analysis_settings = AnalysisSettings(settings, workout.category)
    
    # Calculate sample interval and duration
    sample_interval = infer_sample_interval(df.index) if isinstance(df.index, pd.DatetimeIndex) else 1.0
    if sample_interval <= 0:
        sample_interval = 1.0
    duration_sec = sample_interval * len(df)
    
    # Check available data
    has_power = "power" in df.columns and not df["power"].isna().all()
    has_hr = "heart_rate" in df.columns and not df["heart_rate"].isna().all()
    has_cadence = "cadence" in df.columns and not df["cadence"].isna().all()
    has_speed = "avg_speed" in df.columns and not df["avg_speed"].isna().all()  # Approximate
    
    # Create session info
    session = _create_session_info(parser, duration_sec, len(df), sample_interval)
    
    # Compute power metrics (single pass)
    power_stats, np_value, if_value, tss_value, vi_value = _compute_power_metrics(
        df, analysis_settings, window, has_power
    )
    
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
        athlete_ftp=analysis_settings.ftp,
        athlete_max_hr=analysis_settings.max_hr,
        athlete_lt_hr=analysis_settings.lt_hr
    )
    
    # Compute zone analysis (single pass)
    zones = _compute_zone_analysis(df, analysis_settings, sample_interval, has_power, has_hr)
    
    # Compute heart rate drift (single computation)
    hr_drift = _compute_heart_rate_drift_analysis(df, drift_start, drift_duration, has_power, has_hr)
    
    # Handle autolap generation
    if analysis_settings.autolap_seconds and (len(laps) <= 2 or force_autolap):
        autolaps = split_into_autolaps(df, analysis_settings.autolap_seconds)
        if autolaps:
            laps = autolaps
    
    # Analyze laps
    lap_analyses = _analyze_laps(df, laps, analysis_settings, window, has_power)
    
    # ERG mode analysis
    erg_analysis = _compute_erg_analysis(lap_analyses, parser, analysis_settings)
    
    return WorkoutAnalysis(
        analysis_type="endurance",
        session=session,
        metrics=metrics,
        zones=zones,
        laps=lap_analyses,
        heart_rate_drift=hr_drift,
        erg_analysis=erg_analysis,
        has_power_data=has_power,
        has_heart_rate_data=has_hr,
        has_cadence_data=has_cadence,
        has_speed_data=has_speed
    )


def analyze_strength_workout(parser: StravaDataParser, settings: dict[str, Any]) -> WorkoutAnalysis:
    """
    Analyze strength training workout data and return structured analysis results.
    
    Args:
        parser: StravaDataParser with workout data
        settings: Analysis settings from configuration
        
    Returns:
        WorkoutAnalysis object with strength-specific metrics
    """
    df = parser.data_frame
    workout = parser.workout
    
    # Parse settings (mainly heart rate for strength training)
    analysis_settings = AnalysisSettings(settings, "heart-rate")
    
    if df.empty:
        # Limited analysis without detailed stream data
        elapsed_time = _safe_float(parser.activity.elapsed_time) or 0.0
        session = _create_session_info(parser, elapsed_time, 0, 1.0)
        metrics = WorkoutMetrics()
        return WorkoutAnalysis(
            analysis_type="strength",
            session=session,
            metrics=metrics,
            has_power_data=False,
            has_heart_rate_data=False,
            has_cadence_data=False,
            has_speed_data=False
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
    session = _create_session_info(parser, duration_sec, len(df), sample_interval)
    
    # Compute heart rate stats
    hr_stats = _create_stats_summary(series_stats(df["heart_rate"])) if has_hr else None
    
    # Create metrics object (strength training typically doesn't have power)
    metrics = WorkoutMetrics(
        heart_rate=hr_stats,
        athlete_max_hr=analysis_settings.max_hr,
        athlete_lt_hr=analysis_settings.lt_hr
    )
    
    # Compute heart rate zones if available
    zones = None
    if has_hr and analysis_settings.hr_zones:
        hr_summary = compute_zone_durations(df["heart_rate"], analysis_settings.hr_zones, sample_interval)
        hr_dist = ZoneDistribution()
        hr_dist.total_seconds = _safe_float(hr_summary.get("total_seconds", 0.0)) or 0.0
        hr_dist.sample_interval = _safe_float(hr_summary.get("sample_interval", 1.0)) or 1.0
        
        # Extract zone data from the zones list
        zone_data = hr_summary.get("zones", [])
        if isinstance(zone_data, list):
            for zone_num, zone_info in enumerate(zone_data, start=1):
                if zone_num <= 7 and isinstance(zone_info, dict):
                    zone_obj = ZoneInfo(
                        name=zone_info.get("name", f"Zone {zone_num}"),
                        lower=zone_info.get("lower"),
                        upper=zone_info.get("upper"),
                        seconds=_safe_float(zone_info.get("seconds", 0.0)) or 0.0,
                        percent=_safe_float(zone_info.get("percent", 0.0)) or 0.0
                    )
                    hr_dist.set_zone_info(zone_num, zone_obj)
        zones = ZoneAnalysis(heart_rate_zones=hr_dist)
    
    return WorkoutAnalysis(
        analysis_type="strength",
        session=session,
        metrics=metrics,
        zones=zones,
        has_power_data=False,
        has_heart_rate_data=has_hr,
        has_cadence_data=False,
        has_speed_data=False
    )


def analyze_workout(parser: StravaDataParser, settings: dict[str, Any], 
                   window: int = 30, drift_start: float | None = None, 
                   drift_duration: float | None = None, force_autolap: bool = False) -> WorkoutAnalysis:
    """
    Main dispatcher function for workout analysis.
    
    Routes to appropriate analysis function based on workout category.
    
    Args:
        parser: StravaDataParser with workout data
        settings: Analysis settings from configuration
        window: Window length for NP calculation (default 30 seconds)
        drift_start: Start point for heart rate drift analysis (seconds)
        drift_duration: Duration for heart rate drift analysis (seconds)
        force_autolap: Force autolap generation even if laps exist
        
    Returns:
        WorkoutAnalysis object with computed metrics
    """
    match parser.workout.category:
        case "running" | "cycling" | "skiing":
            return analyze_endurance_workout(parser, settings, window, drift_start, drift_duration, force_autolap)
        case "strength":
            return analyze_strength_workout(parser, settings)
        case _:
            # Default to endurance analysis for unknown categories
            return analyze_endurance_workout(parser, settings, window, drift_start, drift_duration, force_autolap)