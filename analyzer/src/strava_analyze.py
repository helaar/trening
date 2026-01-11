#!/usr/bin/env python3
"""
Strava workout analyzer that downloads workouts from Strava and performs
similar analysis to FIT file analysis.
"""
import argparse
import json
import os
from datetime import date, datetime
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd

from strava.client import download_strava_workouts, StravaDataParser
from strava_auth import StravaTokenManager
from tools.settings import load_settings, ApplicationSettings
from format.utils import ModelFormatter
from tools.calculations import (
    Zone, normalized_power, intensity_factor, training_stress_score,
    series_stats, parse_zone_definitions, compute_zone_durations,
    compute_heart_rate_drift, infer_sample_interval, parse_hms, seconds_to_hms,
    format_range, compute_segment_stats, _print_stats, _print_zone_summary,
    parse_iso8601_duration, split_into_autolaps
)


def print_source_info(log, parser: StravaDataParser) -> None:
    """
    Print information about the data source for a Strava activity.
    
    Args:
        log: Logging function
        parser: StravaDataParser with activity data
    """
    activity = parser.activity
    
    log("\n## Data Source Information")
    device_name = activity.device_name or "Unknown"
    log(f"Device: {device_name}")
    log(f"Manual entry: {'Yes' if activity.manual else 'No'}")
    log(f"From accepted tag: {'Yes' if activity.from_accepted_tag else 'No'}")


def _print_strava_lap_table(log, title: str, lap_rows: list[dict[str, object]], workout_category: str = "") -> None:
    """
    Print a markdown table showing lap information for Strava workouts.
    Adapted from fit_analyzes.py _print_lap_table function.
    """
    no_decimals = {"duration_sec", "repetitions", "lap"}
    one_decimal = {"distance", "np", "avg_power", "avg_hr", "avg_cad", "max_power", "max_hr",
                   "avg_speed", "elev_gain", "elev_loss", "avg_temp", "weight"}
    two_decimals = {"drift_pct"}

    log(f"\n## {title.capitalize()}")
    if not lap_rows:
        log(f"No {title}.")
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

    # Define cadence label based on workout category
    cadence_label = "Avg spm" if workout_category == "running" else "Avg Cad"

    # Define headers for Strava lap table
    headers = [
        ("Lap", "lap"),
        ("Start", "start_str"),
        ("Duration", "duration_str"),
        ("Distance km", "distance"),
        ("NP", "np"),
        ("Avg W", "avg_power"),
        ("Avg HR", "avg_hr"),
        (cadence_label, "avg_cad"),
        ("HR Drift %", "drift_pct"),
        ("Max W", "max_power"),
        ("Max HR", "max_hr"),
        ("Avg km/h", "avg_speed"),
        ("Elevation gain", "elev_gain"),
        ("Elevation loss", "elev_loss"),
        ("Avg ℃", "avg_temp"),
        ("Description", "description")
    ]

    header_titles = [header for header, _ in headers]
    header_row = "| " + " | ".join(header_titles) + " |"
    separator_row = "| " + " | ".join("---" for _ in headers) + " |"

    log(header_row)
    log(separator_row)

    for row in lap_rows:
        row_values = [format_value(row, key) for _, key in headers]
        log("| " + " | ".join(row_values) + " |")


def analyze_strava_workout(parser: StravaDataParser, settings: dict[str, object], args) -> tuple[list[str], dict]:
    """
    Analyze a single Strava workout using adapted analysis logic for Strava data.
    
    Args:
        parser: StravaDataParser with workout data
        settings: Analysis settings from YAML
        args: Command line arguments
        
    Returns:
        Tuple of (list of analysis output lines, structured data dict)
    """
    log_lines: list[str] = []
    json_data: dict = {}

    def log(msg: str = "") -> None:
        log_lines.append(msg)
        print(msg)

    log("## Session Information (Strava)")
    formatter = ModelFormatter()
    formatter.format(log, parser.workout)
    log("")  # newline
    
    # Print data source information
    print_source_info(log, parser)
    
    # Initialize basic session info in JSON
    json_data["session_info"] = {
        "name": parser.workout.name,
        "sport": parser.workout.sport,
        "sub_sport": parser.workout.sub_sport,
        "category": parser.workout.category,
        "start_time": parser.workout.start_time.isoformat() if parser.workout.start_time else None,
        "distance_km": parser.workout.distance_km,
    }
    
    json_data["data_source"] = {
        "device_name": parser.activity.device_name,
        "manual": parser.activity.manual,
        "from_accepted_tag": parser.activity.from_accepted_tag,
    }
    
    # Determine analysis type based on workout category
    match parser.workout.category:
        case "running" | "cycling" | "skiing":
            _strava_endurance_analysis(log, args, settings, parser, json_data)
        case "strength":
            _strava_strength_analysis(log, args, settings, parser, json_data)
        case _:
            log(f"Uncategorized sport {parser.workout.sport}/{parser.workout.sub_sport}")
            # Default to endurance analysis
            _strava_endurance_analysis(log, args, settings, parser, json_data)
    
    return log_lines, json_data


def _strava_endurance_analysis(log, args, settings: dict[str, object], parser: StravaDataParser, json_data: dict | None = None) -> None:
    """Endurance-focused analysis for Strava data."""
    df = parser.data_frame
    laps = parser.laps
    workout = parser.workout
    
    if df.empty:
        log("No detailed stream data available from Strava for this activity.")
        log("Analysis limited to activity summary data.")
        return
    
    # Get settings for sport category
    cat_settings = settings.get(workout.category, {})
    if isinstance(cat_settings, dict):
        ftp = cat_settings.get("ftp")
        power_zones = parse_zone_definitions(cat_settings.get("power-zones"))
    else:
        ftp = None
        power_zones = None
    
    hr_settings = settings.get("heart-rate", {})
    if isinstance(hr_settings, dict):
        max_hr = hr_settings.get("max")
        lt_hr = hr_settings.get("lt")
        hr_zones = parse_zone_definitions(hr_settings.get("hr-zones"))
    else:
        max_hr = None
        lt_hr = None
        hr_zones = None
    
    # Ensure DatetimeIndex for sample interval calculation
    if isinstance(df.index, pd.DatetimeIndex):
        sample_interval = infer_sample_interval(df.index)
    else:
        sample_interval = 1.0
    if sample_interval <= 0:
        sample_interval = 1.0
    
    duration_sec = sample_interval * len(df)
    
    # Check available data
    has_power = "power" in df.columns and not df["power"].isna().all()
    has_hr = "heart_rate" in df.columns and not df["heart_rate"].isna().all()
    has_cadence = "cadence" in df.columns and not df["cadence"].isna().all()
    
    # Calculate basic stats
    power_stats = series_stats(df["power"], drop_nulls=True) if has_power else None
    hr_stats = series_stats(df["heart_rate"], drop_nulls=True) if has_hr else None
    cad_stats = series_stats(df["cadence"], drop_nulls=True) if has_cadence else None
    
    # Power analysis
    np_value = None
    if_value = None
    tss_value = None
    vi_value = None
    
    if has_power:
        try:
            np_value = normalized_power(df["power"], window=args.window)
            if np_value and ftp:
                if_value = intensity_factor(np_value, ftp)
                tss_value = training_stress_score(duration_sec, np_value, if_value, ftp)
            
            avg_power = df["power"].dropna().mean()
            if np_value and avg_power and avg_power > 0:
                vi_value = np_value / avg_power
        except Exception as e:
            log(f"Warning: Power analysis failed: {e}")
    
    # Basic activity info
    log(f"Total distance: {workout.distance}")
    log(f"Moving time: {seconds_to_hms(duration_sec)}")
    log(f"Data points: {len(df)}")
    log(f"Estimated sampling interval: {sample_interval:.2f} s")
    
    if ftp:
        log(f"Athlete FTP: {ftp:.0f} W")
    if max_hr:
        log(f"Athlete Max HR: {max_hr} bpm")
    if lt_hr:
        log(f"Athlete Lactate Threshold: {lt_hr} bpm")
    
    # Power analysis output
    if has_power and power_stats:
        log("\n## Power (W)")
        _print_stats(log, power_stats)
        if np_value:
            log(f"Normalized Power (NP): {np_value:.1f} W")
        avg_power = power_stats.get("mean")
        if avg_power:
            log(f"Average power: {avg_power:.1f} W")
        if vi_value:
            log(f"Variability Index (VI): {vi_value:.3f}")
        if if_value:
            log(f"Intensity Factor (IF): {if_value:.3f}")
        if tss_value:
            log(f"Training Stress Score (TSS): {tss_value:.1f}")
    
    # Heart rate analysis
    if has_hr and hr_stats:
        log("\n## Heart Rate (bpm)")
        _print_stats(log, hr_stats)
    
    # Cadence analysis
    if has_cadence and cad_stats:
        cadence_title = "Steps (spm)" if workout.category == "running" else "Cadence (rpm)"
        log(f"\n## {cadence_title}")
        _print_stats(log, cad_stats)
    
    # Zone analysis
    if power_zones and has_power:
        log("\n## Time in power zones")
        power_summary = compute_zone_durations(df["power"], power_zones, sample_interval)
        _print_zone_summary(log, power_summary, "power")
    
    if hr_zones and has_hr:
        log("\n## Time in heart rate zones")
        hr_summary = compute_zone_durations(df["heart_rate"], hr_zones, sample_interval)
        _print_zone_summary(log, hr_summary, "heart rate")
    
    # Heart rate drift
    if has_power and has_hr:
        drift_start = parse_hms(args.drift_start) if args.drift_start else None
        drift_duration = parse_hms(args.drift_duration) if args.drift_duration else None
        drift_result = compute_heart_rate_drift(df, drift_start, drift_duration)
        
        if drift_result:
            log("\n## Heart Rate Drift")
            try:
                # Handle potential type issues with dict values
                duration = drift_result.get('duration', 0)
                if duration and isinstance(duration, (int, float)):
                    log(f"Analysis duration: {seconds_to_hms(float(duration))}")
                
                # Print drift metrics with safe conversion
                for key, label in [
                    ('avg_hr_p1', '- Avg HR (P1)'),
                    ('avg_hr_p2', '- Avg HR (P2)'),
                    ('avg_power_p1', '- Avg power (P1)'),
                    ('avg_power_p2', '- Avg power (P2)'),
                    ('hr_per_watt_p1', '- HR/W (P1)'),
                    ('hr_per_watt_p2', '- HR/W (P2)'),
                    ('drift_pct', '- HR drift')
                ]:
                    value = drift_result.get(key)
                    if value is not None:
                        if key == 'drift_pct':
                            log(f"{label}: {value:.2f} %")
                        elif 'hr_per_watt' in key:
                            log(f"{label}: {value:.4f}")
                        else:
                            unit = " bpm" if "HR" in label else " W"
                            log(f"{label}: {value:.1f}{unit}")
            except Exception as e:
                log(f"Warning: Could not format heart rate drift results: {e}")
    
    # ------------------------------------------------------------------
    # Lap Analysis
    # ------------------------------------------------------------------
    autolap_seconds = None
    autolap = settings.get("autolap")  # e.g. "PT10M"
    if autolap:
        autolap_seconds = parse_iso8601_duration(str(autolap))

    # If no (or only one) lap in data file and autolap is defined, or if user specified autolap:
    if autolap_seconds and (len(laps) <= 2 or args.autolap):
        autolaps = split_into_autolaps(df, autolap_seconds)
        if autolaps:
            log(
                f"Autolap enabled {autolap}({seconds_to_hms(autolap_seconds)}). "
                f"Generating {len(autolaps)} laps automatically."
            )
            laps = autolaps

    lap_rows = []
    if not laps:
        log("No laps (nor autolap) found in session.")
    else:
        
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

            stats = compute_segment_stats(lap_segment, ftp=ftp, window=args.window)
            intensity = lap.get("intensity")
            intensity_str = intensity_names.get(intensity, str(intensity) if intensity is not None else "") if isinstance(intensity, int) else ""
            label = lap.get("label", "")
            
            # Get power zone if available
            zone = Zone.get_zone(power_zones, stats["avg_power"]) if has_power and power_zones and stats["avg_power"] else None
            zone_name = zone.name if zone else None
            description = " / ".join([str(part) for part in (label, intensity_str, zone_name) if part])
            
            lap_rows.append({
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
                "avg_speed": stats["avg_speed"],
                "elev_gain": stats["elev_gain"],
                "elev_loss": stats["elev_loss"],
                "avg_temp": stats["avg_temp"],
                "distance": stats["distance"],
                "description": description or "-"
            })

        if lap_rows:
            _print_strava_lap_table(log, "laps", lap_rows, workout.category)
        else:
            log("Found laps, but no valid data in these segments.")
    
    # Populate JSON data if provided
    if json_data is not None:
        json_data["analysis_type"] = "endurance"
        json_data["duration_sec"] = duration_sec
        json_data["data_points"] = len(df)
        json_data["sample_interval"] = sample_interval
        
        if ftp:
            json_data["athlete_ftp"] = ftp
        if max_hr:
            json_data["athlete_max_hr"] = max_hr
        if lt_hr:
            json_data["athlete_lt_hr"] = lt_hr
        
        # Power metrics
        if has_power and power_stats:
            json_data["power"] = {
                "stats": power_stats,
                "normalized_power": np_value,
                "variability_index": vi_value,
                "intensity_factor": if_value,
                "training_stress_score": tss_value,
            }
        
        # Heart rate metrics
        if has_hr and hr_stats:
            json_data["heart_rate"] = {"stats": hr_stats}
        
        # Cadence metrics
        if has_cadence and cad_stats:
            json_data["cadence"] = {"stats": cad_stats}
        
        # Zone data
        if power_zones and has_power:
            power_summary = compute_zone_durations(df["power"], power_zones, sample_interval)
            json_data["power_zones"] = power_summary
        
        if hr_zones and has_hr:
            hr_summary = compute_zone_durations(df["heart_rate"], hr_zones, sample_interval)
            json_data["hr_zones"] = hr_summary
        
        # Heart rate drift
        if has_power and has_hr:
            drift_start = parse_hms(args.drift_start) if args.drift_start else None
            drift_duration = parse_hms(args.drift_duration) if args.drift_duration else None
            drift_result = compute_heart_rate_drift(df, drift_start, drift_duration)
            if drift_result:
                json_data["hr_drift"] = drift_result
        
        # Laps
        if lap_rows:
            json_data["laps"] = lap_rows
    
    log("\nGenerated by Strava analyzer (adapted from fit-analyzer)")


def _strava_strength_analysis(log, args, settings: dict[str, object], parser: StravaDataParser, json_data: dict | None = None) -> None:
    """Strength training analysis for Strava data."""
    df = parser.data_frame
    workout = parser.workout
    
    hr_settings = settings.get("heart-rate", {})
    if isinstance(hr_settings, dict):
        max_hr = hr_settings.get("max")
        lt_hr = hr_settings.get("lt")
        hr_zones = parse_zone_definitions(hr_settings.get("hr-zones"))
    else:
        max_hr = None
        lt_hr = None
        hr_zones = None
    
    if df.empty:
        log("No detailed stream data available for strength training analysis.")
        return
    
    # Ensure DatetimeIndex for sample interval calculation
    if isinstance(df.index, pd.DatetimeIndex):
        sample_interval = infer_sample_interval(df.index)
    else:
        sample_interval = 1.0
    if sample_interval <= 0:
        sample_interval = 1.0
    
    # Use elapsed_time from Strava activity for total duration (includes rest periods)
    # This is more accurate for strength training than calculating from data points
    duration_sec = float(parser.activity.elapsed_time) if parser.activity.elapsed_time else sample_interval * len(df)
    has_hr = "heart_rate" in df.columns and not df["heart_rate"].isna().all()
    hr_stats = None
    
    log(f"Data points: {len(df)}")
    log(f"Estimated sampling interval: {sample_interval:.2f} s")
    log(f"Total duration: {seconds_to_hms(duration_sec)}")
    
    if has_hr:
        hr_stats = series_stats(df["heart_rate"])
        log("\n## Heart Rate (bpm)")
        _print_stats(log, hr_stats)
        
        if hr_zones:
            log("\n## Time in heart rate zones")
            hr_summary = compute_zone_durations(df["heart_rate"], hr_zones, sample_interval)
            _print_zone_summary(log, hr_summary, "heart rate")
    
    # Populate JSON data if provided
    if json_data is not None:
        json_data["analysis_type"] = "strength"
        json_data["duration_sec"] = duration_sec
        json_data["data_points"] = len(df)
        json_data["sample_interval"] = sample_interval
        
        if has_hr and hr_stats:
            json_data["heart_rate"] = {"stats": hr_stats}
            
            if hr_zones:
                hr_summary = compute_zone_durations(df["heart_rate"], hr_zones, sample_interval)
                json_data["hr_zones"] = hr_summary
    
    log("\nNote: Detailed strength set analysis not available from Strava stream data.")
    log("Generated by Strava analyzer (adapted from fit-analyzer)")


def main():
    env_file = Path(__file__).parent.parent / '.env'
    print(f"Loading environment variables from: {env_file}")
    load_dotenv(dotenv_path=env_file)
    
    # Ensure valid Strava token
    if not StravaTokenManager.ensure_valid_token(env_file):
        print("ERROR: Could not obtain valid Strava access token")
        return 1
    
    # Reload environment if token was refreshed
    load_dotenv(dotenv_path=env_file, override=True)
    
    parser = argparse.ArgumentParser(
        description="Download and analyze Strava workouts for a given date."
    )
    parser.add_argument(
        "--date","-d",
        required=True,
        help="Date to download workouts for (YYYY-MM-DD format)"
    )
    parser.add_argument(
        "--athlete","-a",
        required=True,
        default="helge",
        help="Athlete identifier"
    )
    parser.add_argument("--app-settings", "-s", default="../app-settings.yaml", help="Path to app-settings.yaml.")
    parser.add_argument("--athlete-settings", "-as", default="../athlete-settings.yaml", help="Path to athlete-settings.yaml.")
    parser.add_argument("--ftp", type=float, help="Override FTP (watts).")
    parser.add_argument("--window", type=int, default=30, help="Window length (s) for NP. Default 30.")
    parser.add_argument("--drift-start", help="Start point for heart rate drift (HH:MM:SS, MM:SS or SS).")
    parser.add_argument("--drift-duration", help="Duration for heart rate drift (HH:MM:SS, MM:SS or SS).")
    parser.add_argument("--autolap", type=bool, required=False, help="Autolap for entire session")
    
    args = parser.parse_args()
    
    # Parse target date
    try:
        target_date = datetime.strptime(args.date, "%Y-%m-%d").date()
    except ValueError:
        print(f"Invalid date format: {args.date}. Use YYYY-MM-DD format.")
        return 1
    
    # Load settings
    app_settings_dict = load_settings(args.app_settings) if Path(args.app_settings).exists() else {}
    athlete_settings_dict = load_settings(args.athlete_settings) if Path(args.athlete_settings).exists() else {}
    
    # Get athlete-specific settings (ensure lowercase)
    athlete_id = args.athlete.lower()
    athletes_data = athlete_settings_dict.get("athletes", {})
    if isinstance(athletes_data, dict):
        athlete_data = athletes_data.get(athlete_id, {})
    else:
        athlete_data = {}
    
    # Merge settings with athlete-specific data taking precedence
    settings = {**app_settings_dict, **athlete_data}
    
    # Parse application settings
    app_config_data = app_settings_dict.get("application", {})
    if isinstance(app_config_data, dict):
        app_settings = ApplicationSettings.model_validate(app_config_data)
    else:
        app_settings = ApplicationSettings.model_validate({})
    
    try:
        print(f"Downloading Strava workouts for {target_date}...")
        
        # Check for access token
        if not os.getenv('STRAVA_ACCESS_TOKEN'):
            print("ERROR: STRAVA_ACCESS_TOKEN environment variable not found.")
            print("Please create a .env file with your Strava API credentials:")
            print("STRAVA_ACCESS_TOKEN=your_access_token_here")
            print("\nTo get an access token:")
            print("1. Go to https://www.strava.com/settings/api")
            print("2. Create an application")
            print("3. Use the API to get an access token")
            return 1
        
        # Download workouts
        workout_parsers = download_strava_workouts(target_date)
        
        if not workout_parsers:
            print(f"No workouts found for {target_date}")
            return 0
        
        print(f"Found {len(workout_parsers)} workout(s) for {target_date}")
        
        # Analyze each workout
        if isinstance(app_config_data, dict):
            output_dir_str = app_config_data.get("output-dir", "./ENV/exchange/athletes")
        else:
            output_dir_str = "./ENV/exchange/athletes"
        
        # Resolve output path relative to the app-settings.yaml file location
        settings_file_dir = Path(args.app_settings).parent
        base_output_dir = (settings_file_dir / output_dir_str).resolve()
        athlete_analyses_dir = base_output_dir / athlete_id / "analyses"
        athlete_analyses_dir.mkdir(parents=True, exist_ok=True)
        
        for i, workout_parser in enumerate(workout_parsers, 1):
            print(f"\nAnalyzing workout {i}/{len(workout_parsers)}: {workout_parser.workout.name}")
            
            # Analyze workout
            analysis_lines, analysis_data = analyze_strava_workout(workout_parser, settings, args)
            
            # Create filename
            workout_name = workout_parser.workout.name or f"workout_{i}"
            # Sanitize filename
            safe_name = "".join(c for c in workout_name if c.isalnum() or c in (' ', '-', '_')).strip()
            safe_name = safe_name.replace(' ', '_')
            
            date_prefix = target_date.strftime("%Y-%m-%d")
            base_filename = f"{date_prefix}_strava_{safe_name}-analysis"
            
            # Write markdown analysis to file
            analysis_text = "\n".join(analysis_lines)
            md_path = athlete_analyses_dir / f"{base_filename}.md"
            md_path.write_text(analysis_text, encoding="utf-8")
            print(f"Markdown analysis saved to: {md_path}")
            
            # Write JSON analysis to file
            json_path = athlete_analyses_dir / f"{base_filename}.json"
            json_path.write_text(json.dumps(analysis_data, indent=2, default=str), encoding="utf-8")
            print(f"JSON analysis saved to: {json_path}")
    
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())