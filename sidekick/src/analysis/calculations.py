"""
Core analysis calculations and utilities shared between FIT and Strava analyzers.
"""
import re
from dataclasses import dataclass
from datetime import timedelta

import pandas as pd
import numpy as np

from models.athlete import ZoneDefinition


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class Zone:
    name: str
    low: float | None = None
    high: float | None = None

    def in_zone(self, value: float) -> bool:
        low = self.low or value
        high = self.high or value
        return low <= value <= high
    
    @staticmethod
    def get_zone(zones: list['Zone'], value: float) -> 'Zone | None':
        for z in zones:
            if z.in_zone(value):
                return z
        return None


# ---------------------------------------------------------------------------
# Core calculations
# ---------------------------------------------------------------------------
def normalized_power(power: pd.Series, window: int = 30) -> float | None:
    """Calculate Normalized Power (NP) from power data."""
    valid_power = power.dropna()
    if valid_power.empty:
        raise ValueError("No power data available to calculate NP from.")

    rolling_mean = valid_power.rolling(window=window, min_periods=window).mean().dropna()
    if rolling_mean.empty:
        return None

    np_value = (rolling_mean.pow(4).mean()) ** 0.25
    return float(np_value)


def intensity_factor(np_value: float, ftp: float) -> float:
    """Calculate Intensity Factor (IF)."""
    if ftp <= 0:
        raise ValueError("FTP must be > 0.")
    return np_value / ftp


def training_stress_score(duration_sec: float, np_value: float, if_value: float, ftp: float) -> float:
    """Calculate Training Stress Score (TSS)."""
    return (duration_sec * np_value * if_value) / (ftp * 3600) * 100


def series_stats(series: pd.Series, drop_nulls: bool | None = False) -> dict[str, float | None]:
    """Calculate basic statistics for a pandas Series."""
    valid = series.dropna()
    if valid.empty:
        return {"min": None, "max": None, "mean": None, "std": None}

    if drop_nulls:
        valid = valid[valid != 0.0]

    return {
        "min": float(valid.min()),
        "max": float(valid.max()),
        "mean": float(valid.mean()),
        "std": float(valid.std(ddof=0)),
    }


def infer_sample_interval(index: pd.DatetimeIndex) -> float:
    """Infer the sampling interval from a DatetimeIndex."""
    if len(index) < 2:
        return 0.0
    deltas = index.to_series().diff().dropna().dt.total_seconds()
    if deltas.empty:
        return 0.0
    return float(deltas.median())


# ---------------------------------------------------------------------------
# Zone analysis
# ---------------------------------------------------------------------------
def parse_zone_definitions(zone_defs: list[ZoneDefinition] | None) -> list[Zone]:
    """
    Convert ZoneDefinition models to Zone objects for calculations.
    
    Args:
        zone_defs: List of ZoneDefinition objects from AthleteSettings
        
    Returns:
        List of Zone objects for analysis
    """
    if not zone_defs:
        return []

    zones: list[Zone] = []
    for zd in zone_defs:
        zones.append(Zone(
            name=zd.name,
            low=float(zd.min),
            high=float(zd.max) if zd.max is not None else None
        ))
    return zones


def parse_range(range_spec: str) -> tuple[float | None, float | None]:
    """Parse a range specification like '100-150' or '150-'."""
    parts = str(range_spec).split("-", 1)
    if len(parts) != 2:
        raise ValueError(f"Could not parse range: {range_spec!r}")

    low_str, high_str = parts[0].strip(), parts[1].strip()
    low = float(low_str) if low_str else None
    high = float(high_str) if high_str else None
    return low, high


def compute_zone_durations(
    series: pd.Series,
    zones: list[Zone],
    sample_interval: float | None = None,
) -> dict[str, object]:
    """Compute time spent in each zone."""
    if not zones:
        return {"total_seconds": 0.0, "sample_interval": 0.0, "zones": []}

    if sample_interval is None or sample_interval <= 0:
        if isinstance(series.index, pd.DatetimeIndex):
            sample_interval = infer_sample_interval(series.index)
        else:
            sample_interval = 1.0
        if sample_interval <= 0:
            sample_interval = 1.0  # Fallback

    valid_mask = series.notna()
    total_samples = int(valid_mask.sum())
    total_seconds = total_samples * sample_interval

    results = []
    for i, zone in enumerate(zones):
        low = zone.low if zone.low is not None else float("-inf")
        high = zone.high if zone.high is not None else float("inf")

        if zone.high is None or i == len(zones) - 1:
            mask = (series >= low) & (series <= high)
        else:
            mask = (series >= low) & (series < high)

        mask = mask & valid_mask
        zone_seconds = float(mask.sum() * sample_interval)
        percent = (zone_seconds / total_seconds * 100.0) if total_seconds else 0.0

        results.append(
            {
                "name": zone.name,
                "lower": zone.low,
                "upper": zone.high,
                "seconds": zone_seconds,
                "percent": percent,
            }
        )

    return {
        "total_seconds": total_seconds,
        "sample_interval": sample_interval,
        "zones": results,
    }


# ---------------------------------------------------------------------------
# Heart rate drift analysis
# ---------------------------------------------------------------------------
def compute_heart_rate_drift(
    df: pd.DataFrame,
    start_offset: float | None = None,
    duration: float | None = None,
) -> dict[str, object] | None:
    """Compute heart rate drift analysis."""
    if "heart_rate" not in df or "power" not in df:
        return None

    start_ts = df.index[0]
    end_ts = df.index[-1]

    if start_offset:
        start_ts = df.index[0] + pd.to_timedelta(start_offset, unit="s")
    if duration:
        end_ts = start_ts + pd.to_timedelta(duration, unit="s")

    start_ts = max(start_ts, df.index[0])
    end_ts = min(end_ts, df.index[-1])

    if end_ts <= start_ts:
        return None

    segment = df.loc[start_ts:end_ts, ["heart_rate", "power"]].dropna()
    if len(segment) < 4:
        return None

    midpoint = len(segment) // 2
    if midpoint == 0 or midpoint == len(segment):
        return None

    p1 = segment.iloc[:midpoint]
    p2 = segment.iloc[midpoint:]

    avg_power_p1 = p1["power"].mean()
    avg_power_p2 = p2["power"].mean()

    if avg_power_p1 <= 0 or avg_power_p2 <= 0:
        return None

    avg_hr_p1 = p1["heart_rate"].mean()
    avg_hr_p2 = p2["heart_rate"].mean()

    hr_per_watt_p1 = avg_hr_p1 / avg_power_p1
    hr_per_watt_p2 = avg_hr_p2 / avg_power_p2

    drift_pct = ((hr_per_watt_p2 - hr_per_watt_p1) / hr_per_watt_p1) * 100.0

    return {
        "start_ts": segment.index[0],
        "end_ts": segment.index[-1],
        "duration": (segment.index[-1] - segment.index[0]).total_seconds(),
        "samples": len(segment),
        "avg_hr_p1": float(avg_hr_p1),
        "avg_hr_p2": float(avg_hr_p2),
        "avg_power_p1": float(avg_power_p1),
        "avg_power_p2": float(avg_power_p2),
        "hr_per_watt_p1": float(hr_per_watt_p1),
        "hr_per_watt_p2": float(hr_per_watt_p2),
        "drift_pct": float(drift_pct),
    }


# ---------------------------------------------------------------------------
# Parsing utilities
# ---------------------------------------------------------------------------
def parse_iso8601_duration(value: str) -> float:
    """Parse an ISO 8601 duration (e.g. PT10M) to seconds."""
    pattern = (
        r"^P(?:(?P<days>\d+)D)?"
        r"(?:T(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+(?:\.\d+)?)S)?)?$"
    )
    match = re.fullmatch(pattern, value)
    if not match:
        raise ValueError(f"Invalid ISO 8601 duration: {value!r}")

    days = float(match.group("days") or 0)
    hours = float(match.group("hours") or 0)
    minutes = float(match.group("minutes") or 0)
    seconds = float(match.group("seconds") or 0)
    return ((days * 24 + hours) * 60 + minutes) * 60 + seconds


def parse_hms(value: str) -> float:
    """Parse HH:MM:SS, MM:SS or SS to seconds."""
    parts = value.strip().split(":")
    if not parts or len(parts) > 3:
        raise ValueError(f"Invalid time format: {value!r}")

    parts = [int(p) for p in parts]
    if len(parts) == 3:
        h, m, s = parts
    elif len(parts) == 2:
        h, m, s = 0, parts[0], parts[1]
    else:
        h, m, s = 0, 0, parts[0]
    return float(h * 3600 + m * 60 + s)


def seconds_to_hms(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format."""
    return str(timedelta(seconds=int(round(seconds))))


# ---------------------------------------------------------------------------
# Elevation calculations
# ---------------------------------------------------------------------------
def _find_col(df: pd.DataFrame, name: str) -> str | None:
    """Find column containing name."""
    for c in df.columns.tolist():
        if name in c:
            return c
    return None


def calculate_elevation(segment: pd.DataFrame) -> tuple[float, float, float, float]:
    """Calculate elevation gain, loss, min, max from segment."""
    altcol = _find_col(segment, "altitude")
    if not altcol:
        return (0.0, 0.0, 0.0, 0.0)
    
    try:
        alt = segment[altcol].replace(0, None).dropna().astype(float).rolling(
            window=1, min_periods=1, center=True
        ).median()
        delta = alt.diff().dropna()
        
        if delta.empty:
            return (0.0, 0.0, 0.0, 0.0)
        
        # Ensure delta is numeric for comparisons
        delta = pd.to_numeric(delta, errors='coerce').dropna()
        delta_asc = delta[delta >= 0.1].sum()
        delta_desc = delta[delta <= -0.1].sum()

        return (
            float(delta_asc) if pd.notna(delta_asc) else 0.0,
            float(-delta_desc) if pd.notna(delta_desc) else 0.0,
            float(alt.min()) if pd.notna(alt.min()) else 0.0,
            float(alt.max()) if pd.notna(alt.max()) else 0.0
        )
    except Exception:
        return (0.0, 0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# Segment analysis
# ---------------------------------------------------------------------------
def compute_segment_stats(segment: pd.DataFrame, ftp: float | None, window: int = 30) -> dict[str, float | None]:
    """Compute comprehensive statistics for a segment."""
    stats: dict[str, float | None] = {}
    if len(segment) < 2:
        stats.update({
            "duration_sec": 0.0,
            "avg_power": None,
            "np": None,
            "vi": None,
            "if": None,
            "avg_hr": None,
            "avg_cad": None,
            "max_power": None,
            "max_hr": None,
            "drift_pct": None,
            "distance": None,
            "avg_speed": None,
            "elev_gain": None,
            "elev_loss": None,
            "avg_temp": None,
        })
        return stats

    duration_sec = (segment.index[-1] - segment.index[0]).total_seconds()
    stats["duration_sec"] = duration_sec

    # Power analysis
    if "power" in segment.columns:
        avg_power = segment["power"].dropna().mean()
        stats["avg_power"] = float(avg_power) if pd.notna(avg_power) else None
        np_value = normalized_power(segment["power"], window=window)
        stats["np"] = float(np_value) if np_value is not None else None
        stats["vi"] = (stats["np"] / stats["avg_power"]) if stats["np"] and stats["avg_power"] else None
        stats["if"] = (stats["np"] / ftp) if ftp and stats["np"] and ftp > 0 else None
        stats["max_power"] = float(segment["power"].dropna().max()) if not segment["power"].dropna().empty else None
        
        drift = compute_heart_rate_drift(segment)
        if drift and "drift_pct" in drift:
            drift_val = drift["drift_pct"]
            if isinstance(drift_val, (int, float)) and pd.notna(drift_val):
                stats["drift_pct"] = float(drift_val)
            else:
                stats["drift_pct"] = None
        else:
            stats["drift_pct"] = None
    else:
        stats.update({
            "avg_power": None, "np": None, "vi": None, "if": None,
            "max_power": None, "drift_pct": None
        })

    # Heart rate analysis
    if "heart_rate" in segment.columns:
        avg_hr = segment["heart_rate"].dropna().mean()
        stats["avg_hr"] = float(avg_hr) if pd.notna(avg_hr) else None
        stats["max_hr"] = float(segment["heart_rate"].dropna().max()) if not segment["heart_rate"].dropna().empty else None
    else:
        stats["avg_hr"] = None
        stats["max_hr"] = None

    # Cadence analysis
    if "cadence" in segment.columns:
        avg_cad = segment["cadence"].dropna().mean()
        stats["avg_cad"] = float(avg_cad) if avg_cad and pd.notna(avg_cad) else None
    else:
        stats["avg_cad"] = None

    # Elevation analysis
    elev_gain, elev_loss, _, _ = calculate_elevation(segment)
    stats["elev_gain"] = float(elev_gain) if pd.notna(elev_gain) else None
    stats["elev_loss"] = float(elev_loss) if pd.notna(elev_loss) else None

    # Temperature analysis
    if "temperature" in segment.columns:
        avg_temp = segment["temperature"].dropna().mean()
        stats["avg_temp"] = float(avg_temp) if pd.notna(avg_temp) else None
    else:
        stats["avg_temp"] = None

    # Distance and speed analysis
    if "distance" in segment.columns:
        distance = segment["distance"].dropna().max() - segment["distance"].dropna().min()
        stats["distance"] = float(distance / 1000.0) if pd.notna(distance) else None
        avg_speed = distance / duration_sec * 3.6 if pd.notna(distance) and pd.notna(duration_sec) else None
        stats["avg_speed"] = float(avg_speed) if pd.notna(avg_speed) else None
    else:
        stats["distance"] = None
        stats["avg_speed"] = None

    return stats


# ---------------------------------------------------------------------------
# Autolap functionality
# ---------------------------------------------------------------------------
def split_into_autolaps(df: pd.DataFrame, autolap_seconds: float) -> list[dict[str, pd.Timestamp]]:
    """Split the session into equal segments (autolaps) of autolap_seconds."""
    if len(df) < 2 or autolap_seconds <= 0:
        return []

    autolaps: list[dict[str, pd.Timestamp]] = []
    start_ts = df.index[0]
    end_ts = df.index[-1]
    current_start = start_ts

    while current_start < end_ts:
        current_end = current_start + pd.to_timedelta(autolap_seconds, unit="s")
        if current_end > end_ts:
            current_end = end_ts
        autolaps.append({"start": current_start, "end": current_end})
        current_start = current_end

    return autolaps