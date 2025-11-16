#!/usr/bin/env python3
"""
Calculates NP, IF, TSS, VI, zone distribution, heart rate drift and lap details (incl. autolap)
from a FIT file. All output is logged both to terminal and to <fitfile>-analysis.txt.
"""
import argparse
import re
from dataclasses import dataclass
from datetime import timedelta, date
from pathlib import Path

import pandas as pd
import yaml
from pydantic import BaseModel, Field

from format.garmin_fit import FitFileParser
from format.utils import ModelFormatter


# ---------------------------------------------------------------------------
# Settings models
# ---------------------------------------------------------------------------
class ApplicationSettings(BaseModel):
    output_dir: str = Field(default=".", alias="output-dir")
    
    def get_output_path(self) -> Path:
        """Get the configured output directory as a Path object."""
        output_path = Path(self.output_dir).expanduser().resolve()
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path


# ---------------------------------------------------------------------------
# Data classes and helper functions
# ---------------------------------------------------------------------------
INTENSITY_NAMES = {
    0: "active",
    1: "rest",
    2: "warmup",
    3: "cooldown",
    4: "recovery",
    5: "interval",
}

@dataclass
class Zone:
    name: str
    low: float | None = None
    high: float | None = None

    def in_zone(self,value:float) -> bool:
        low = self.low or value
        high = self.high or value

        return low <= value <= high
    
    @staticmethod
    def get_zone(zones : list['Zone'], value: float) -> 'Zone | None':
        for z in zones:
            if z.in_zone(value):
                return z
        return None



def _safe_float(value: float | None) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
    
    




def normalized_power(power: pd.Series, window: int = 30) -> float:
    valid_power = power.dropna()
    if valid_power.empty:
        raise ValueError("No power data available to calculate NP from.")

    rolling_mean = valid_power.rolling(window=window, min_periods=window).mean().dropna()
    if rolling_mean.empty:
        raise ValueError("Segment too short to calculate Normalized Power.")

    np_value = (rolling_mean.pow(4).mean()) ** 0.25
    return float(np_value)


def intensity_factor(np_value: float, ftp: float) -> float:
    if ftp <= 0:
        raise ValueError("FTP must be > 0.")
    return np_value / ftp


def training_stress_score(duration_sec: float, np_value: float, if_value: float, ftp: float) -> float:
    return (duration_sec * np_value * if_value) / (ftp * 3600) * 100


def series_stats(series: pd.Series) -> dict[str, float | None]:
    valid = series.dropna()
    if valid.empty:
        return {"min": None, "max": None, "mean": None, "std": None}

    return {
        "min": float(valid.min()),
        "max": float(valid.max()),
        "mean": float(valid.mean()),
        "std": float(valid.std(ddof=0)),
    }


def load_settings(path: str) -> dict[str, object]:
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return data


def parse_zone_definitions(raw: list[dict[str, str]] | None) -> list[Zone]:
    if not raw:
        return []

    zones: list[Zone] = []
    for entry in raw:
        if not isinstance(entry, dict) or len(entry) != 1:
            raise ValueError(f"Invalid zone definition: {entry!r}")
        name, range_spec = next(iter(entry.items()))
        low, high = parse_range(range_spec)
        zones.append(Zone(name=name, low=low, high=high))
    return zones


def parse_range(range_spec: str) -> tuple[float|None, float|None]:
    parts = str(range_spec).split("-", 1)
    if len(parts) != 2:
        raise ValueError(f"Could not parse range: {range_spec!r}")

    low_str, high_str = parts[0].strip(), parts[1].strip()
    low = float(low_str) if low_str else None
    high = float(high_str) if high_str else None
    return low, high


def parse_iso8601_duration(value: str) -> float:
    """
    Parses an ISO 8601 duration (e.g. PT10M) to seconds.
    """
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


def infer_sample_interval(index: pd.DatetimeIndex) -> float:
    if len(index) < 2:
        return 0.0
    deltas = index.to_series().diff().dropna().dt.total_seconds()
    if deltas.empty:
        return 0.0
    return float(deltas.median())


def parse_hms(value: str) -> float:
    """
    Parses HH:MM:SS, MM:SS or SS to seconds.
    """
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


def compute_zone_durations(
    series: pd.Series,
    zones: list[Zone],
    sample_interval: float | None = None,
) -> dict[str, object]:
    if not zones:
        return {"total_seconds": 0.0, "sample_interval": 0.0, "zones": []}

    if sample_interval is None or sample_interval <= 0:
        sample_interval = infer_sample_interval(series.index)
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


def compute_heart_rate_drift(
    df: pd.DataFrame,
    start_offset: float | None = None,
    duration: float | None = None,
) -> dict[str, object] | None:
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

def _find_col(df: pd.DataFrame, name:str) -> str | None:
    for c in df.columns.tolist():
        if name in c: return c
    return None    
    

def _calculate_elevation(segment: pd.DataFrame) -> tuple[float, float, float, float]:
    altcol = _find_col(segment, "altitude")
    if not altcol:
        return (0.0, 0.0, 0.0,0.0)
    
    alt = segment[altcol].replace(0,None).dropna().astype(float).rolling(window=1,min_periods=1,center=True).median()
    delta = alt.diff()
    delta_asc = delta.where(delta >= 0.1, other=0.0)
    delta_desc = delta.where(delta <= -0.1, other=0.0)

    return (delta_asc.clip(lower=0).sum(),0-delta_desc.clip(upper=0).sum(), alt.min(), alt.max())


def compute_segment_stats(segment: pd.DataFrame, ftp: float, window: int = 30) -> dict[str, float | None]:
    stats: dict[str, float] | None = {}
    if len(segment) < 2:
        stats["duration_sec"] = 0.0
        stats["avg_power"] = None
        stats["np"] = None
        stats["vi"] = None
        stats["if"] = None
        stats["avg_hr"] = None
        stats["avg_cad"] = None
        stats["max_power"] = None
        stats["max_hr"] = None
        stats["drift_pct"] = None
        stats["dist"]  = None
        stats["avg_speed"] = None
        stats["ascent"] = None
        stats["descent"] = None
        stats["avg_temp"] = None
        return stats

    duration_sec = (segment.index[-1] - segment.index[0]).total_seconds()
    stats["duration_sec"] = duration_sec

    avg_power = segment["power"].dropna().mean()
    stats["avg_power"] = float(avg_power) if pd.notna(avg_power) else None

    try:
        if segment["power"].dropna().empty:
            np_value = None
        else:
            np_value = normalized_power(segment["power"], window=window)
    except Exception:
        np_value = None

    stats["np"] = float(np_value) if np_value is not None else None

    stats["vi"] = (stats["np"] / stats["avg_power"]) if stats["np"] and stats["avg_power"] else None
    stats["if"] = (stats["np"] / ftp) if stats["np"] and ftp > 0 else None

    avg_hr = segment["heart_rate"].dropna().mean()
    stats["avg_hr"] = float(avg_hr) if pd.notna(avg_hr) else None

    avg_cad = segment["cadence"].dropna().mean()
    stats["avg_cad"] = float(avg_cad) if pd.notna(avg_cad) else None

    stats["max_power"] = float(segment["power"].dropna().max()) if not segment["power"].dropna().empty else None
    stats["max_hr"] = float(segment["heart_rate"].dropna().max()) if not segment["heart_rate"].dropna().empty else None

    drift = compute_heart_rate_drift(segment)
    stats["drift_pct"] = drift["drift_pct"] if drift else None

    ascent, descent, min, max = _calculate_elevation(segment)
    stats["ascent"] = float(ascent) if pd.notna(ascent) else None
    stats["descent"] = float(descent) if pd.notna(descent) else None

    if segment.get("temperature") is not None:
        avg_temp = segment["temperature"].dropna().mean()
        stats["avg_temp"] = float(avg_temp) if pd.notna(avg_temp) else None
    else:
        stats["avg_temp"] = None

    distance = segment["distance"].dropna().max() - segment["distance"].dropna().min()
    stats["distance"] = float(distance / 1000.0) if pd.notna(distance) else None

    avg_speed = distance / duration_sec * 3.600 if pd.notna(distance) and pd.notna(duration_sec) else None
    stats["avg_speed"] = float(avg_speed) if pd.notna(avg_speed) else None


    return stats


def split_into_autolaps(df: pd.DataFrame, autolap_seconds: float) -> list[dict[str, pd.Timestamp]]:
    """
    Splits the session into equal segments (autolaps) of autolap_seconds.
    """
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




def summarize_strength_sets(
    set_messages: list[dict[str, object]],
    records_df: pd.DataFrame,
    hr_column: str = "heart_rate",
) -> list[dict[str, object]]:
    """
    Combines a Garmin "set" list and a record dataset to calculate
    heart rate statistics per strength set.

    Parameters
    ----------
    set_messages : iterable of dict
        Result from e.g. `[msg.get_values() for msg in fitfile.get_messages("set")]`.
        Expects fields like 'timestamp', 'duration', 'message_index',
        'set_type', 'repetitions', 'weight', 'category', but includes all available fields.
    records_df : pandas.DataFrame
        Time-series (e.g. from `read_records`). Must have DatetimeIndex and contain the `hr_column` column.
    hr_column : str, default "heart_rate"
        Column name for heart rate values.

    Returns
    -------
    List[Dict[str, object]]
        One dictionary per set, sorted by message_index (or the order you provided them),
        with fields like `start_time`, `end_time`, `duration_sec`, `avg_hr`, etc.

    Raises
    ------
    ValueError
        If `records_df` is missing `hr_column` or index is not DatetimeIndex.
    """
    if hr_column not in records_df.columns:
        raise ValueError(f"records_df is missing column '{hr_column}'.")

    if not isinstance(records_df.index, pd.DatetimeIndex):
        raise ValueError("records_df must be indexed with DatetimeIndex.")

    hr_series = records_df[hr_column].dropna()
    if hr_series.empty:
        hr_available = False
    else:
        hr_available = True

    # Sort sets by message_index if available, otherwise keep original order
    def sort_key(msg: dict[str, object]):
        idx = msg.get("message_index")
        return idx if isinstance(idx, (int, float)) else float("inf")

    results: list[dict[str, object]] = []
    for ordinal, msg in enumerate(sorted(set_messages, key=sort_key), start=1):
        start_ts = msg.get("timestamp")
        duration_sec = msg.get("duration")

        if start_ts is None or duration_sec is None:
            # Missing required info -> skip
            continue

        start_ts = pd.to_datetime(start_ts)
        duration_sec = float(duration_sec)
        end_ts = start_ts + pd.to_timedelta(duration_sec, unit="s")

        # Extract heart rate data for the interval
        avg_hr = min_hr = max_hr = None
        hr_samples = 0
        if hr_available:
            window = hr_series.loc[start_ts:end_ts]
            hr_samples = int(window.size)
            if hr_samples:
                avg_hr = float(window.mean())
                min_hr = float(window.min())
                max_hr = float(window.max())

        entry = {
            "ordinal": ordinal,  # order in table
            "message_index": msg.get("message_index"),
            "start_time": start_ts,
            "end_time": end_ts,
            "duration_sec": duration_sec,
            "set_type": msg.get("set_type"),
            "repetitions": msg.get("repetitions"),
            "weight": msg.get("weight"),
            "category": msg.get("category"),  # tuple eller annet Garmin-format
            "avg_hr": avg_hr,
            "min_hr": min_hr,
            "max_hr": max_hr,
            "hr_samples": hr_samples,
        }

        results.append(entry)

    return results

def format_range(low: float | None, high: float | None) -> str:
    def fmt(value: float| None) -> str:
        if value is None:
            return ""
        return f"{int(value)}" if float(value).is_integer() else f"{value:g}"

    if low is None and high is None:
        return "—"
    if low is None:
        return f"≤{fmt(high)}"
    if high is None:
        return f"{fmt(low)}+"
    return f"{fmt(low)}–{fmt(high)}"


def seconds_to_hms(seconds: float) -> str:
    return str(timedelta(seconds=int(round(seconds))))


def _print_stats(log, stats: dict[str, float| None]) -> None:
    def fmt(value: float| None) -> str:
        return f"{value:.1f}" if value is not None else "—"

    log(f"- Min : {fmt(stats['min'])}")
    log(f"- Max: {fmt(stats['max'])}")
    log(f"- Average: {fmt(stats['mean'])}")
    log(f"- Std : {fmt(stats['std'])}")


def _print_zone_summary(log, summary: dict[str, object], label: str) -> None:
    zones = summary["zones"]
    total = summary["total_seconds"]

    if not zones or total == 0:
        log(f"No {label} data for zone calculation.")
        return

    log(f"Total time in calculation: {seconds_to_hms(total)}")
    for zone in zones:
        range_str = format_range(zone["lower"], zone["upper"])
        log(
            f"- {zone['name']:<20} {range_str:<10} "
            f"{seconds_to_hms(zone['seconds']):>8} ({zone['percent']:5.1f}%)"
        ) 


def _print_lap_table(log, tittel: str, lap_rows: list[dict[str, object]], headers: list[tuple[str, str]]) -> None:
    """
    Skriv en markdown-tabell som viser lap/sett-informasjon.
    `headers` er en liste av (kolonnetittel, nøkkel-i-row).
    """
    no_decimals = {"duration_sec", "repetitions", "lap"}
    one_decimal = {"distance", "np", "avg_power", "avg_hr", "avg_cad", "max_power", "max_hr",
                   "avg_speed", "ascent", "descent", "avg_temp", "weight"}
    two_decimals = {"drift_pct"}

    log(f"\n## {tittel.capitalize()}")
    if not lap_rows:
        log(f"No {tittel}.")
        return

    def fmt_float(value: float | None, decimals: int = 1) -> str:
        return f"{value:.{decimals}f}" if value is not None else "—"

    def format_value(row: dict, key: str) -> str:
        value = row.get(key)
        if key.endswith("_str"):
            return str(value) if value is not None else ""
        if isinstance(value, (float, int)):
            if key in one_decimal:
                return fmt_float(value, 1)
            if key in no_decimals:
                return fmt_float(value, 0)
            if key in two_decimals:
                return fmt_float(value, 2)
            return fmt_float(value, 1)
        return str(value) if value is not None else ""

    header_titles = [header for header, _ in headers]
    header_row = "| " + " | ".join(header_titles) + " |"
    separator_row = "| " + " | ".join("---" for _ in headers) + " |"

    log(header_row)
    log(separator_row)

    for row in lap_rows:
        row_values = [format_value(row, key) for _, key in headers]
        log("| " + " | ".join(row_values) + " |")

def _strength_based_summary(log, args, settings : dict[str,object], fit : FitFileParser) -> list[str]:

    df = fit.data_frame

    max_hr = settings.get("max-hr")
    hr_zones = parse_zone_definitions(settings.get("hr-zones"))
    autolap = settings.get("autolap")  # f.eks. "PT10M"

    drift_start = parse_hms(args.drift_start) if args.drift_start else None
    drift_duration = parse_hms(args.drift_duration) if args.drift_duration else None

    sample_interval = infer_sample_interval(df.index)
    if sample_interval <= 0:
        sample_interval = 1.0

    duration_sec = sample_interval * len(df)

    hr_stats = series_stats(df["heart_rate"])
#   if_value = intensity_factor(np_value, ftp)
#   tss_value = training_stress_score(duration_sec, np_value, if_value, ftp)

    if df.get("temparature"): log(f"Temperature (average): {df["temperature"].dropna().mean():.1f} ℃")
    log(f"Data points: {len(df)}")
    log(f"Estimated sampling interval: {sample_interval:.2f} s")
#   log(f"Intensity Factor (IF): {if_value:.3f}")
#   log(f"Training Stress Score (TSS): {tss_value:.1f}")

    log("\n# Heart Rate (bpm)")
    _print_stats(log, hr_stats)

    if hr_zones:
        log("\n# Time in heart rate zones")
        hr_summary = compute_zone_durations(df["heart_rate"], hr_zones, sample_interval)
        _print_zone_summary(log, hr_summary, "heart rate")

    if fit.sets:
        sets_summary = summarize_strength_sets(fit.sets, fit.data_frame)
        headers = [
            ("#", "ordinal"),
            ("Type", "set_type"),
            ("Duration (s)", "duration_sec"),
            ("Repetitions", "repetitions"),
            ("Weight (kg)", "weight"),
            ("Avg HR", "avg_hr"),
            ("Max HR", "max_hr")
        ]
        _print_lap_table(log,"strength sets", sets_summary, headers)
    

def _power_based_summary(log, args, settings : dict[str,object], fit : FitFileParser) -> list[str]:
    
    # TODO: Temporary solution. Should refactor the rest as well
    df = fit.data_frame
    laps = fit.laps

    ftp = settings.get(fit.workout.category).get("ftp")
    if ftp is None:
        raise ValueError("FTP must be specified via --ftp or in settings.yaml.")

    max_hr = settings.get("max-hr")
    power_zones = parse_zone_definitions(settings.get(fit.workout.category).get("power-zones"))
    hr_zones = parse_zone_definitions(settings.get("hr-zones"))
    autolap = settings.get("autolap")  # f.eks. "PT10M"

    drift_start = parse_hms(args.drift_start) if args.drift_start else None
    drift_duration = parse_hms(args.drift_duration) if args.drift_duration else None

    sample_interval = infer_sample_interval(df.index)
    if sample_interval <= 0:
        sample_interval = 1.0

    duration_sec = sample_interval * len(df)

    power_stats = series_stats(df["power"])
    hr_stats = series_stats(df["heart_rate"])
    cad_stats = series_stats(df["cadence"])
    # speed_stats = series_stats(df["speed"])

    np_value = normalized_power(df["power"], window=args.window)
    if_value = intensity_factor(np_value, ftp)
    tss_value = training_stress_score(duration_sec, np_value, if_value, ftp)

    avg_power = df["power"].dropna().mean()
    vi_value = (np_value / avg_power) if avg_power and avg_power > 0 else None

    asc, desc,min,max = _calculate_elevation(df)
    log(f"Total elevation gain¹: {asc:.1f} m")
    log(f"Max elevation: {max:.1f} masl")
    log(f"Min elevation: {min:.1f} masl")
    log(f"Speed (average): {fit.workout._distance/duration_sec*3.6:.1f} km/h")
    if df.get("temparature"): log(f"Temperature (average): {df["temperature"].dropna().mean():.1f} ℃")
    log(f"Data points: {len(df)}")
    log(f"Estimated sampling interval: {sample_interval:.2f} s")
    log(f"FTP: {ftp:.0f} W")
    if max_hr:
        log(f"Max HR (from settings): {max_hr} bpm")
    
    log("\n## Power (W)")
    _print_stats(log, power_stats)
    log(f"Normalized Power (NP): {np_value:.1f} W")
    if avg_power and avg_power > 0:
        log(f"Average power: {avg_power:.1f} W")
        log(f"Variability Index (VI): {vi_value:.3f}")
    else:
        log("Average power: —")
        log("Variability Index (VI): —")
    log(f"Intensity Factor (IF): {if_value:.3f}")
    log(f"Training Stress Score (TSS): {tss_value:.1f}")

    log("\n## Heart Rate (bpm)")
    _print_stats(log, hr_stats)

    log("\n## Cadence (rpm)")
    _print_stats(log, cad_stats)

    if power_zones:
        log("\n## Time in power zones")
        power_summary = compute_zone_durations(df["power"], power_zones, sample_interval)
        _print_zone_summary(log, power_summary, "power")
    if hr_zones:
        log("\n## Time in heart rate zones")
        hr_summary = compute_zone_durations(df["heart_rate"], hr_zones, sample_interval)
        _print_zone_summary(log, hr_summary, "heart rate")

    drift_result = compute_heart_rate_drift(df, drift_start, drift_duration)
    log("\n## Heart Rate Drift")
    if drift_result:
        rel_start = (drift_result["start_ts"] - df.index[0]).total_seconds()
        rel_end = (drift_result["end_ts"] - df.index[0]).total_seconds()
        log(
            f"Segment: {seconds_to_hms(rel_start)} → {seconds_to_hms(rel_end)} "
            f"({seconds_to_hms(drift_result['duration'])})"
        )
        log(f"- Avg HR (P1): {drift_result['avg_hr_p1']:.1f} bpm")
        log(f"- Avg HR (P2): {drift_result['avg_hr_p2']:.1f} bpm")
        log(f"- Avg power (P1): {drift_result['avg_power_p1']:.1f} W")
        log(f"- Avg power (P2): {drift_result['avg_power_p2']:.1f} W")
        log(f"- HR/W (P1): {drift_result['hr_per_watt_p1']:.4f}")
        log(f"- HR/W (P2): {drift_result['hr_per_watt_p2']:.4f}")
        log(f"- HR drift: {drift_result['drift_pct']:.2f} %")
    else:
        log("No valid data to calculate heart rate drift for selected segment.")

    # ------------------------------------------------------------------
    # Lap details
    # ------------------------------------------------------------------
    autolap_seconds = None
    if autolap:
        autolap_seconds = parse_iso8601_duration(str(autolap))

    # If no (or only one) lap in data file and autolap is defined:
    if autolap_seconds and (len(laps) <= 1 or args.autolap):
        autolaps = split_into_autolaps(df, autolap_seconds)
        if autolaps:
            log(
                f"Autolap enabled {autolap}({seconds_to_hms(autolap_seconds)}). "
                f"Generating {len(autolaps)} laps automatically."
            )
            laps = autolaps

    if not laps:
        log("No laps (nor autolap) found in session.")
    else:
        lap_rows = []
        for idx, lap in enumerate(laps, start=1):
            start_ts = lap["start"]
            end_ts = lap["end"]
            lap_segment = df.loc[start_ts:end_ts].dropna(how="all")
            if lap_segment.empty:
                continue

            stats = compute_segment_stats(lap_segment, ftp=ftp, window=args.window)
            intensity = lap.get("intensity")
            intensity_str = INTENSITY_NAMES.get(intensity, str(intensity) if intensity is not None else "")
            label = lap.get("label")
            zone = Zone.get_zone(power_zones, stats["avg_power"])
            zone_name = zone.name if zone else None
            description = " / ".join([part for part in (label, intensity_str, zone_name) if part])
            lap_rows.append(
                {
                    "lap": idx,
                    "start_str": seconds_to_hms((start_ts - df.index[0]).total_seconds()),
                    "duration_str": seconds_to_hms(stats["duration_sec"]),
                    "np": stats["np"],
                    "avg_power": stats["avg_power"],
                    "avg_hr": stats["avg_hr"],
                    "avg_cad": stats["avg_cad"],
                    "drift_pct": stats["drift_pct"],
                    "max_power": stats["max_power"],
                    "max_hr": stats["max_hr"],
                    "avg_speed" : stats["avg_speed"],
                    "ascent" : stats["ascent"],
                    "descent" : stats["descent"],
                    "avg_temp" : stats["avg_temp"],
                    "distance" : stats["distance"],
                    "description": description or "-"
                }
            )

        if lap_rows:
            headers = [
                ("Lap", "lap"),
                ("Start", "start_str"),
                ("Duration", "duration_str"),
                ("Distance km", "distance"),
                ("NP", "np"),
                ("Avg W", "avg_power"),
                ("Avg HR", "avg_hr"),
                ("Avg Cad", "avg_cad"),
                ("HR Drift %", "drift_pct"),
                ("Max W", "max_power"),
                ("Max HR", "max_hr"),
                ("Avg km/h", "avg_speed"),
                ("Elevation¹ ↑", "ascent"),
                ("Elevation¹ ↓", "descent"),
                ("Avg ℃", "avg_temp"),
                ("Description", "description")
                ]
            _print_lap_table(log, "laps", lap_rows, headers)
        else:
            log("Found laps, but no valid data in these segments.")

        log("--------")
        log(" ¹ - Elevation is approximately calculated. Does not match 100% with TP and Strava")
   

# ---------------------------------------------------------------------------
# Main program
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Calculate NP, IF, TSS, VI, zone distribution, heart rate drift and lap details from a FIT file."
    )
    parser.add_argument("--fitfile", required=True, help="Path to FIT file.")
    parser.add_argument("--settings", help="Path to settings.yaml.")
    parser.add_argument("--ftp", type=float, help="Override FTP (watts).")
    parser.add_argument("--window", type=int, default=30, help="Window length (s) for NP. Default 30.")
    parser.add_argument("--drift-start", help="Start point for heart rate drift (HH:MM:SS, MM:SS or SS).")
    parser.add_argument("--drift-duration", help="Duration for heart rate drift (HH:MM:SS, MM:SS or SS).")
    parser.add_argument("--autolap", type=bool,required=False, help="Autolap for entire session")
    args = parser.parse_args()

    fit_path = Path(args.fitfile).expanduser().resolve()
    log_lines: list[str] = []

    def log(msg: str = "") -> None:
        log_lines.append(msg)
        print(msg)

    try:
        settings: dict[str, object] = {}
        if args.settings:
            settings = load_settings(args.settings)
        fit = FitFileParser(fit_path)

        # Parse application settings
        app_config_data = settings.get("application", {})
        app_settings = ApplicationSettings.model_validate(app_config_data)

        log("## Session Information")
        formatter = ModelFormatter()
        formatter.format(log,fit.workout)
        log("") # newline

        
        match fit.workout.category:
            case "running"|"cycling":
                _power_based_summary( log, args, settings, fit)
            case "strength":
                _strength_based_summary(log, args, settings, fit)
            case _:
                print(f"Uncategorized sport {fit.workout.sport}/{fit.workout.sub_sport}")

        # ------------------------------------------------------------------
        # Write to file
        # ------------------------------------------------------------------
        output_path = app_settings.get_output_path()
        
        # Create filename with date prefix from workout or today's date
        if fit.workout.start_time:
            date_prefix = fit.workout.start_time.strftime("%Y-%m-%d")
        else:
            date_prefix = date.today().strftime("%Y-%m-%d")
        
        filename = f"{date_prefix}_{fit_path.stem}-analysis.md"
        
        analysis_text = "\n".join(log_lines)
        analysis_path = output_path / filename
        analysis_path.write_text(analysis_text, encoding="utf-8")

        print(f"\nAnalysis saved to: {analysis_path}")

    except Exception as exc:
        error_msg = f"\n[ERROR] {exc}"
        log(error_msg)

        if log_lines:
            # Use fallback for error case
            try:
                app_config_data = settings.get("application", {}) if 'settings' in locals() else {}
                app_settings = ApplicationSettings.model_validate(app_config_data)
                output_path = app_settings.get_output_path()
            except Exception:
                # Final fallback if settings parsing fails
                output_path = Path(".").resolve()
                output_path.mkdir(parents=True, exist_ok=True)
            
            # Create filename with date prefix - use today's date in error cases
            date_prefix = date.today().strftime("%Y-%m-%d")
            filename = f"{date_prefix}_{fit_path.stem}-analysis.md"
            
            analysis_text = "\n".join(log_lines)
            analysis_path = output_path / filename
            analysis_path.write_text(analysis_text, encoding="utf-8")
            print(f"Preliminary log saved in: {analysis_path}")

        raise


if __name__ == "__main__":
    main()