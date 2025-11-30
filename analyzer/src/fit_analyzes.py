#!/usr/bin/env python3
"""
Calculates NP, IF, TSS, VI, zone distribution, heart rate drift and lap details (incl. autolap)
from a FIT file. All output is logged both to terminal and to <fitfile>-analysis.txt.
"""
import argparse
from datetime import date
from pathlib import Path

import pandas as pd

from format.garmin_fit import FitFileParser
from format.utils import ModelFormatter
from tools.settings import load_settings, ApplicationSettings
from tools.calculations import (
    Zone, normalized_power, intensity_factor, training_stress_score,
    series_stats, parse_zone_definitions, compute_zone_durations,
    compute_heart_rate_drift, infer_sample_interval, parse_hms,
    parse_iso8601_duration, seconds_to_hms, format_range,
    compute_segment_stats, split_into_autolaps, calculate_elevation,
    _print_stats, _print_zone_summary
)


# ---------------------------------------------------------------------------
# Local constants and helper functions
# ---------------------------------------------------------------------------
INTENSITY_NAMES = {
    0: "active",
    1: "rest",
    2: "warmup",
    3: "cooldown",
    4: "recovery",
    5: "interval",
}


def _safe_float(value: float | None) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def estimate_hr_tss(df : pd.DataFrame, lthr: float, resting_hr: float | None = None) -> tuple[float | None,float | None]:
    """
    Estimate hrTSS using multiple formulas for debugging against TrainingPeaks.
    If resting_hr is provided, also computes the "resting HR subtracted" variant.
    """
    hr_series = df["heart_rate"]

    # Soneintervaller (TrainingPeaks-standard ift LTHR)
    zones = [
        (0.00, 0.81, 0.5),
        (0.81, 0.90, 0.7),
        (0.90, 0.94, 0.9),
        (0.94, 1.00, 1.0),
        (1.00, float("inf"), 1.1),
    ]

    series = hr_series.dropna()
    if series.empty:
        print("[hrTSS debug] No HR data available.")
        return (None,None)

    # Total elapsed time (from first to last timestamp)
    if not df.empty and isinstance(df.index, pd.DatetimeIndex):
        elapsed_time = (df.index[-1] - df.index[0]).total_seconds()
    else:
        elapsed_time = None

    # Moving/active time: count only samples where HR > 0
    moving_mask = (df["heart_rate"] > 0) if "heart_rate" in df else None
    moving_time = moving_mask.sum() if moving_mask is not None else None

    min_hr = series.min()
    max_hr = series.max()

    diffs = series.index.to_series().diff().dt.total_seconds().dropna()
    sample_interval = float(diffs.median()) if not diffs.empty else 1.0
    duration_secs = len(series) * sample_interval
    avg_hr = series.mean()

    total_weighted_time = 0.0
    for i, (low, high, weight) in enumerate(zones):
        threshold_low = low * lthr
        threshold_high = high * lthr

        mask = (series >= threshold_low) & (series < threshold_high)
        zone_seconds = mask.sum() * sample_interval

        total_weighted_time += zone_seconds * weight

    if duration_secs == 0:
        return (None,None)

    hr_if = total_weighted_time / duration_secs
    hr_tss_tp = duration_secs * (avg_hr / lthr) / 36.0
 
    # Resting HR subtracted variant
    if resting_hr is not None and lthr > resting_hr:
        avg_hr_adj = avg_hr - resting_hr
        lthr_adj = lthr - resting_hr
        if lthr_adj > 0:
            hr_tss_resting = duration_secs * (avg_hr_adj / lthr_adj) / 36.0
        else:
            hr_tss_resting = None
    else:
        hr_tss_resting = None

    # Variant: divisor 50 (sometimes used in other platforms)
    hr_tss_div50 = duration_secs * (avg_hr / lthr) / 50.0
 
    # Variant: non-linear scaling for low HR (experimental)
    if avg_hr < 0.8 * lthr:
        hr_tss_nl = duration_secs * ((avg_hr / lthr) ** 1.2) / 36.0
    else:
        hr_tss_nl = None

    return (hr_tss_tp, hr_if)


def _find_col(df: pd.DataFrame, name: str) -> str | None:
    for c in df.columns.tolist():
        if name in c: 
            return c
    return None    
     

def _calculate_elevation(segment: pd.DataFrame) -> tuple[float, float, float, float]:
    altcol = _find_col(segment, "altitude")
    if not altcol:
        return (0.0, 0.0, 0.0, 0.0)
    
    alt = segment[altcol].replace(0, None).dropna().astype(float).rolling(window=1, min_periods=1, center=True).median()
    delta = alt.diff().astype(float)
    delta_asc = delta.where(delta >= 0.1, other=0.0)
    delta_desc = delta.where(delta <= -0.1, other=0.0)

    try:
        gain = float(delta_asc.clip(lower=0).sum())
        loss = float(abs(delta_desc.clip(upper=0).sum()))
        min_alt = float(alt.min())
        max_alt = float(alt.max())
        return (gain, loss, min_alt, max_alt)
    except (TypeError, ValueError):
        return (0.0, 0.0, 0.0, 0.0)


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

        try:
            # Handle various timestamp formats
            if hasattr(start_ts, 'timestamp'):
                start_ts = pd.Timestamp(start_ts)
            else:
                start_ts = pd.to_datetime(str(start_ts), errors='coerce')
            
            # Handle duration conversion
            if hasattr(duration_sec, '__float__'):
                duration_sec = float(duration_sec)
            else:
                duration_sec = float(str(duration_sec)) if duration_sec is not None else 0.0
        except (TypeError, ValueError, AttributeError):
            continue
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


def _print_lap_table(log, tittel: str, lap_rows: list[dict[str, object]], headers: list[tuple[str, str]]) -> None:
    """
    Skriv en markdown-tabell som viser lap/sett-informasjon.
    `headers` er en liste av (kolonnetittel, nøkkel-i-row).
    """
    no_decimals = {"duration_sec", "repetitions", "lap"}
    one_decimal = {"distance", "np", "avg_power", "avg_hr", "avg_cad", "max_power", "max_hr",
                   "avg_speed", "elev_gain", "elev_loss", "avg_temp", "weight"}
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


def _strength_based_summary(log, args, settings : dict[str,object], fit : FitFileParser) -> None:
    df = fit.data_frame
    hr_settings = settings.get("heart-rate", {})
    if isinstance(hr_settings, dict):
        max_hr = hr_settings.get("max") 
        lt_hr = hr_settings.get("lt")
        hr_zones = parse_zone_definitions(hr_settings.get("hr-zones"))
    else:
        max_hr = None
        lt_hr = None
        hr_zones = []
    
    autolap = settings.get("autolap")  # f.eks. "PT10M"

    hr_tss, hr_if = (None, None) # TODO: estimate_hr_tss(df=df, lthr=lt_hr)
    drift_start = parse_hms(args.drift_start) if args.drift_start else None
    drift_duration = parse_hms(args.drift_duration) if args.drift_duration else None

    if isinstance(df.index, pd.DatetimeIndex):
        sample_interval = infer_sample_interval(df.index)
    else:
        sample_interval = 1.0
    if sample_interval <= 0:
        sample_interval = 1.0

    duration_sec = sample_interval * len(df)
    hr_stats = series_stats(df["heart_rate"])

    if "temperature" in df.columns: 
        log(f"Temperature (average): {df['temperature'].dropna().mean():.1f} ℃")
    log(f"Data points: {len(df)}")
    log(f"Estimated sampling interval: {sample_interval:.2f} s")
    if hr_if: 
        log(f"Intensity Factor (hrIF): {hr_if:.3f}")
    if hr_tss: 
        log(f"Training Stress Score (hrTSS): {hr_tss:.1f}")

    log("\n## Heart Rate (bpm)")
    _print_stats(log, hr_stats)

    if hr_zones:
        log("\n## Time in heart rate zones")
        hr_summary = compute_zone_durations(df["heart_rate"], hr_zones, sample_interval)
        _print_zone_summary(log, hr_summary, "heart rate")

    if fit.sets:
        # Cast fit.sets to the expected type
        sets_data = [dict(s) if hasattr(s, 'items') else s for s in fit.sets]
        sets_summary = summarize_strength_sets(sets_data, fit.data_frame)
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
    

def _endurance_based_summary(log, args, settings : dict[str,object], fit : FitFileParser) -> None:
    # TODO: Temporary solution. Should refactor the rest as well
    df = fit.data_frame
    laps = fit.laps

    cat_settings = settings.get(fit.workout.category, {})
    if isinstance(cat_settings, dict):
        ftp = cat_settings.get("ftp")
        power_zones = parse_zone_definitions(cat_settings.get("power-zones"))
    else:
        ftp = None
        power_zones = []

    hr_settings = settings.get("heart-rate", {})
    if isinstance(hr_settings, dict):
        max_hr = hr_settings.get("max")
        lt_hr = hr_settings.get("lt")
        hr_zones = parse_zone_definitions(hr_settings.get("hr-zones"))
    else:
        max_hr = None
        lt_hr = None
        hr_zones = []
        
    autolap = settings.get("autolap")  # f.eks. "PT10M"

    drift_start = parse_hms(args.drift_start) if args.drift_start else None
    drift_duration = parse_hms(args.drift_duration) if args.drift_duration else None

    if isinstance(df.index, pd.DatetimeIndex):
        sample_interval = infer_sample_interval(df.index)
    else:
        sample_interval = 1.0
    if sample_interval <= 0:
        sample_interval = 1.0

    duration_sec = sample_interval * len(df)

    has_power = "power" in df.columns
    has_hr = "heart_rate" in df.columns
    has_cadence = "cadence" in df.columns
    drop_nulls = True 
    power_stats = series_stats(df["power"], drop_nulls=drop_nulls) if has_power else None
    hr_stats = series_stats(df["heart_rate"], drop_nulls=drop_nulls) if has_hr else None
    cad_stats = series_stats(df["cadence"], drop_nulls=drop_nulls) if has_cadence else None

    np_value = normalized_power(df["power"], window=args.window) if has_power else None
    if_value = None
    tss_value = None
    
    if has_power and np_value and ftp:
        if_value = intensity_factor(np_value, ftp)
        tss_value = training_stress_score(duration_sec, np_value, if_value, ftp)

    hr_tss, hr_if = (None, None) # TODO: estimate_hr_tss(df=df,lthr=lt_hr) if not has_power and has_hr else (None, None)

    avg_power = df["power"].dropna().mean() if has_power else None
    vi_value = (np_value / avg_power) if has_power and np_value and avg_power and avg_power > 0 else None

    elev_gain, elev_loss, min_elev, max_elev = _calculate_elevation(df)
    log(f"Total elevation gain¹: {elev_gain:.1f} m")
    log(f"Max elevation: {max_elev:.1f} masl")
    log(f"Min elevation: {min_elev:.1f} masl")
    log(f"Speed (average): {fit.workout._distance/duration_sec*3.6:.1f} km/h")
    if "temperature" in df.columns:
        log(f"Temperature (average): {df['temperature'].dropna().mean():.1f} ℃")
    log(f"Data points: {len(df)}")
    log(f"Estimated sampling interval: {sample_interval:.2f} s")
    if ftp:
        log(f"Athlete FTP: {ftp:.0f} W")
    
    if max_hr:
        log(f"Athlete Max HR: {max_hr} bpm")
    if lt_hr:
        log(f"Athlete Lactate Threshold {lt_hr} bpm")
    
    if has_power and power_stats:
        log("\n## Power (W)²")
        _print_stats(log, power_stats)
        if np_value:
            log(f"Normalized Power (NP): {np_value:.1f} W")
        if avg_power and avg_power > 0:
            log(f"Average power: {avg_power:.1f} W")
            if vi_value:
                log(f"Variability Index (VI): {vi_value:.3f}")
        else:
            log("Average power: —")
            log("Variability Index (VI): —")
        if if_value:
            log(f"Intensity Factor (IF): {if_value:.3f}")
        if tss_value:
            log(f"Training Stress Score (TSS): {tss_value:.1f}")
    else:
        if hr_if: 
            log(f"Intensity Factor(hrIF) {hr_if:.3f}")
        if hr_tss: 
            log(f"Training Stress Score (hrTSS) {hr_tss:.1f}")

    if has_hr and hr_stats:
        log("\n## Heart Rate (bpm)²")
        _print_stats(log, hr_stats)

    if has_cadence and cad_stats:
        log("\n## Cadence (rpm)²")
        _print_stats(log, cad_stats)

    if power_zones and has_power:
        log("\n## Time in power zones")
        power_summary = compute_zone_durations(df["power"], power_zones, sample_interval)
        _print_zone_summary(log, power_summary, "power")
        
    if hr_zones and has_hr:
        log("\n## Time in heart rate zones")
        hr_summary = compute_zone_durations(df["heart_rate"], hr_zones, sample_interval)
        _print_zone_summary(log, hr_summary, "heart rate")

    drift_result = compute_heart_rate_drift(df, drift_start, drift_duration) if has_power and has_hr else None
  
    if drift_result:
        log("\n## Heart Rate Drift")
        try:
            # Handle drift result timestamps
            start_ts_obj = drift_result["start_ts"]
            end_ts_obj = drift_result["end_ts"]
            
            if hasattr(start_ts_obj, 'timestamp'):
                start_ts = pd.Timestamp(start_ts_obj)
            else:
                start_ts = pd.to_datetime(str(start_ts_obj), errors='coerce')
                
            if hasattr(end_ts_obj, 'timestamp'):
                end_ts = pd.Timestamp(end_ts_obj)
            else:
                end_ts = pd.to_datetime(str(end_ts_obj), errors='coerce')
            
            rel_start = float((start_ts - df.index[0]).total_seconds())
            rel_end = float((end_ts - df.index[0]).total_seconds())
            duration = float(drift_result['duration']) if isinstance(drift_result['duration'], (int, float)) else 0.0
            log(
                f"Segment: {seconds_to_hms(rel_start)} → {seconds_to_hms(rel_end)} "
                f"({seconds_to_hms(duration)})"
            )
        except (TypeError, ValueError, KeyError):
            log("Heart rate drift calculation failed")
            return
        log(f"- Avg HR (P1): {drift_result['avg_hr_p1']:.1f} bpm")
        log(f"- Avg HR (P2): {drift_result['avg_hr_p2']:.1f} bpm")
        log(f"- Avg power (P1): {drift_result['avg_power_p1']:.1f} W")
        log(f"- Avg power (P2): {drift_result['avg_power_p2']:.1f} W")
        log(f"- HR/W (P1): {drift_result['hr_per_watt_p1']:.4f}")
        log(f"- HR/W (P2): {drift_result['hr_per_watt_p2']:.4f}")
        log(f"- HR drift: {drift_result['drift_pct']:.2f} %")

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
            intensity_str = INTENSITY_NAMES.get(intensity, str(intensity) if intensity is not None else "") if isinstance(intensity, int) else ""
            label = lap.get("label")
            zone = Zone.get_zone(power_zones, stats["avg_power"]) if has_power and power_zones and stats["avg_power"] else None
            zone_name = zone.name if zone else None
            description = " / ".join([str(part) for part in (label, intensity_str, zone_name) if part])
            lap_rows.append(
                {
                    "lap": idx,
                    "start_str": seconds_to_hms((start_ts - df.index[0]).total_seconds()),
                    "duration_str": seconds_to_hms(stats["duration_sec"]) if stats["duration_sec"] else "—",
                    "np": stats["np"],
                    "avg_power": stats["avg_power"],
                    "avg_hr": stats["avg_hr"],
                    "avg_cad": stats["avg_cad"],
                    "drift_pct": stats["drift_pct"],
                    "max_power": stats["max_power"],
                    "max_hr": stats["max_hr"],
                    "avg_speed" : stats["avg_speed"],
                    "elev_gain" : stats["elev_gain"],
                    "elev_loss" : stats["elev_loss"],
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
                ("Elevation gain¹", "elev_gain"),
                ("Elevation loss¹", "elev_loss"),
                ("Avg ℃", "avg_temp"),
                ("Description", "description")
                ]
            _print_lap_table(log, "laps", lap_rows, headers)
        else:
            log("Found laps, but no valid data in these segments.")

        log("--------")
        log(" ¹ - Elevation gain/loss may be inaccurate as it does not match 100% with TP and Strava on outdoor activities. Zwift delivers a more reliable elevation profile and can be trusted.")
        log(" ² - Zeroes in heart rate, cadence and power are removed on overall statistics, but not on laps")
        log("") # newline
        log("Generated by our own fit-analyzer script (beta)")


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
            case "running"|"cycling"|"skiing":
                _endurance_based_summary( log, args, settings, fit)
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
                # Check if settings exists and has content
                if locals().get('settings') and isinstance(locals().get('settings'), dict):
                    app_config_data = locals()['settings'].get("application", {})
                else:
                    app_config_data = {}
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