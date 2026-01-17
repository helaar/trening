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
from tools.settings import (
    load_settings, ApplicationSettings, save_settings,
    get_commute_routes, save_commute_route
)
from tools.calculations import parse_hms, seconds_to_hms
from tools.polyline_analyzer import (
    normalize_polyline, find_matching_commute
)
# New pipeline imports
from analysis.engine import analyze_workout
from analysis.formatters import MarkdownFormatter, JSONFormatter
from analysis.models import WorkoutAnalysis




def analyze_strava_workout(parser: StravaDataParser, settings: dict[str, object], args) -> tuple[list[str], WorkoutAnalysis]:
    """
    Analyze a single Strava workout using the new pipeline architecture.
    
    Args:
        parser: StravaDataParser with workout data
        settings: Analysis settings from YAML
        args: Command line arguments
        
    Returns:
        Tuple of (list of analysis output lines, WorkoutAnalysis object)
    """
    # Convert command line arguments to analysis parameters
    drift_start = parse_hms(args.drift_start) if args.drift_start else None
    drift_duration = parse_hms(args.drift_duration) if args.drift_duration else None
    force_autolap = bool(args.autolap) if hasattr(args, 'autolap') and args.autolap else False
    
    # Run analysis through new pipeline
    analysis = analyze_workout(
        parser=parser,
        settings=settings,
        window=args.window,
        drift_start=drift_start,
        drift_duration=drift_duration,
        force_autolap=force_autolap
    )
    
    # Generate formatted outputs
    markdown_formatter = MarkdownFormatter()
    
    # Get markdown output and convert to lines with print side effects
    markdown_output = markdown_formatter.format(analysis)
    log_lines = markdown_output.split('\n')
    
    # Print each line (to maintain compatibility with current behavior)
    for line in log_lines:
        print(line)
    return log_lines, analysis


def process_commute_detection(
    workout_parser: StravaDataParser,
    athlete_settings_path: str,
    athlete_settings_dict: dict[str, object],
    athlete_id: str
) -> str:
    """
    Process commute detection for a workout.
    
    If the ride is tagged as a commute, normalize and store the polyline.
    If not tagged as commute, check if it matches a known commute route.
    
    Args:
        workout_parser: Parser with workout data
        athlete_settings_path: Path to athlete settings file
        athlete_settings_dict: Current athlete settings
        athlete_id: Athlete identifier
        
    Returns:
        Commute status: "yes, marked by athlete" | "yes, detected" | "no"
    """
    # Skip if no polyline data available
    if not workout_parser.summary_polyline:
        return "no"
    
    # Get known commute routes
    known_commutes = get_commute_routes(athlete_settings_dict, athlete_id)
    
    # Case 1: User tagged this as a commute
    if workout_parser.commute:
        # Normalize the polyline
        normalized = normalize_polyline(workout_parser.summary_polyline)
        
        # Check if this commute route already exists
        matched_commute = find_matching_commute(
            normalized,
            known_commutes,
            threshold_meters=100.0
        )
        
        if matched_commute:
            print(f"  ✓ Commute matches known route: {matched_commute}")
        else:
            # Generate a name for the new commute route
            workout_name = workout_parser.workout.name or "Commute"
            # Use start time to create unique name
            start_time = workout_parser.workout.start_time
            if start_time:
                route_name = f"{workout_name}_{start_time.strftime('%Y%m%d_%H%M')}"
            else:
                # Fallback if no start time
                route_name = f"{workout_name}_{len(known_commutes) + 1}"
            
            # Save the new commute route
            save_commute_route(
                athlete_settings_path,
                athlete_id,
                route_name,
                normalized
            )
            print(f"  ✓ New commute route saved: {route_name}")
        
        return "yes, marked by athlete"
    
    # Case 2: User did not tag as commute, check if it matches a known commute
    else:
        if known_commutes:
            matched_commute = find_matching_commute(
                workout_parser.summary_polyline,
                known_commutes,
                threshold_meters=100.0
            )
            
            if matched_commute:
                print(f"  ℹ️  Route matches commute: {matched_commute} (not tagged as commute)")
                return "yes, detected"
    
    return "no"







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
            
            # Process commute detection and set status on parser
            commute_status = process_commute_detection(
                workout_parser,
                args.athlete_settings,
                athlete_settings_dict,
                athlete_id
            )
            
            # Set commute status on the parser so it's picked up in analysis
            workout_parser.commute_status = commute_status
            
            # Analyze workout
            analysis_lines, analysis_obj = analyze_strava_workout(workout_parser, settings, args)
            
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
            
            # Write JSON analysis to file using JSONFormatter for proper precision
            json_formatter = JSONFormatter()
            clean_json_data = json_formatter.format(analysis_obj)
            
            json_path = athlete_analyses_dir / f"{base_filename}.json"
            json_path.write_text(json.dumps(clean_json_data, indent=2, default=str), encoding="utf-8")
            print(f"JSON analysis saved to: {json_path}")
    
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())