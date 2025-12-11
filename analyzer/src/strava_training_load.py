#!/usr/bin/env python3
"""
Strava Training Load Analyzer - 28-day rolling window analysis
Provides efficient training load overview with minimal API calls.
"""
import argparse
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv

from strava.client import StravaClient
from strava.training_load import TrainingLoadAnalysis
from strava_auth import StravaTokenManager
from tools.settings import load_settings, ApplicationSettings


def format_time(seconds: int) -> str:
    """Format seconds to HH:MM format."""
    if seconds <= 0:
        return "0:00"
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    if hours > 0:
        return f"{hours}:{minutes:02d}"
    return f"{minutes}m"


def format_distance(meters: float) -> str:
    """Format meters to km with appropriate precision."""
    km = meters / 1000
    if km >= 100:
        return f"{km:.0f}km"
    elif km >= 10:
        return f"{km:.1f}km"
    else:
        return f"{km:.2f}km"


def generate_training_load_report(analysis: TrainingLoadAnalysis, settings: dict) -> list[str]:
    """Generate a comprehensive markdown training load report."""
    lines = []
    
    def add_line(text: str = ""):
        lines.append(text)
    
    # Header
    add_line(f"# Training Load Analysis")
    add_line(f"**Period**: {analysis.period_start} to {analysis.period_end}")
    add_line(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    add_line()
    
    # Methodology note
    add_line("## Methodology")
    add_line("**TSS**: `(Strava Weighted Avg Watts / Sport FTP)² × Hours × 100` - Uses sport-specific FTP (cycling/running)")
    add_line("**HRSS**: `(Avg HR / Threshold HR)² × Hours × 100` - Heart rate based stress score")
    add_line("*Note: Values may differ slightly from detailed stream analysis which uses second-by-second data*")
    add_line()
    
    # Overall Summary
    add_line("## Overall Summary")
    add_line(f"- **Total Activities**: {analysis.total_activities}")
    add_line(f"- **Total Training Time**: {format_time(analysis.total_time)}")
    add_line(f"- **Total TSS**: {analysis.total_tss:.1f}")
    add_line(f"- **Total HRSS**: {analysis.total_hrss:.1f}")
    add_line(f"- **Average Weekly TSS**: {analysis.avg_weekly_tss:.1f}")
    add_line(f"- **Average Weekly HRSS**: {analysis.avg_weekly_hrss:.1f}")
    add_line(f"- **Average Weekly Time**: {format_time(int(analysis.avg_weekly_time * 3600))}")
    add_line()
    
    # Weekly Summary Table
    add_line("## Weekly Summary")
    add_line("| Week | Dates | Activities | Time | TSS | HRSS | Training Days | Rest Days |")
    add_line("|------|-------|------------|------|-----|------|---------------|-----------|")
    
    for i, week in enumerate(analysis.weekly_summaries, 1):
        week_label = f"W{i}"
        date_range = f"{week.start_date} - {week.end_date}"
        add_line(f"| {week_label} | {date_range} | {week.total_activities} | "
                f"{format_time(week.total_time)} | {week.total_tss:.1f} | "
                f"{week.total_hrss:.1f} | {week.training_days} | {week.rest_days} |")
    
    add_line()
    
    # Sport Breakdown (overall period)
    add_line("## Sport Distribution")
    all_sports = {}
    for day in analysis.daily_summaries:
        for activity in day.activities:
            sport = activity.sport
            if sport not in all_sports:
                all_sports[sport] = {'count': 0, 'time': 0, 'tss': 0, 'hrss': 0}
            all_sports[sport]['count'] += 1
            all_sports[sport]['time'] += activity.moving_time
            all_sports[sport]['tss'] += activity.tss or 0
            all_sports[sport]['hrss'] += activity.hrss or 0
    
    if all_sports:
        add_line("| Sport | Activities | Time | TSS | HRSS |")
        add_line("|-------|------------|------|-----|------|")
        for sport, data in sorted(all_sports.items()):
            add_line(f"| {sport} | {data['count']} | {format_time(data['time'])} | "
                    f"{data['tss']:.1f} | {data['hrss']:.1f} |")
        add_line()
    
    # TSS/HRSS Trends (if enough data)
    if analysis.tss_trend and len(analysis.tss_trend) >= 7:
        add_line("## 7-Day Rolling Trends")
        add_line("*Rolling 7-day totals for training load metrics*")
        add_line()
        add_line("| Period End | 7-Day TSS | 7-Day HRSS |")
        add_line("|------------|-----------|-------------|")
        
        trend_start = 6  # First rolling window ends on day 7
        for i, (tss, hrss) in enumerate(zip(analysis.tss_trend, analysis.hrss_trend)):
            trend_date = analysis.period_start + timedelta(days=trend_start + i)
            add_line(f"| {trend_date} | {tss:.1f} | {hrss:.1f} |")
        
        add_line()
    
    # Daily Details - Single condensed table
    add_line("## Daily Activity Details")
    add_line("*All activities for the 28-day period*")
    add_line()
    
    add_line("| Date | Activity | Sport | Time | Distance | TSS | HRSS | IF | Avg HR | Avg Power |")
    add_line("|------|----------|-------|------|----------|-----|------|----|----|-----------|")
    
    for day in analysis.daily_summaries:
        date_str = day.date.strftime("%m/%d")
        
        if not day.activities:
            add_line(f"| {date_str} | *Rest Day* | — | — | — | — | — | — | — | — |")
            continue
        
        for activity in day.activities:
            tss_str = f"{activity.tss:.1f}" if activity.tss else "—"
            hrss_str = f"{activity.hrss:.1f}" if activity.hrss else "—"
            if_str = f"{activity.intensity_factor:.2f}" if activity.intensity_factor else "—"
            hr_str = f"{activity.average_heartrate:.0f}" if activity.average_heartrate else "—"
            power_str = f"{activity.average_watts:.0f}W" if activity.average_watts else "—"
            distance_str = format_distance(activity.distance) if activity.distance > 0 else "—"
            
            # Truncate activity name for table readability
            activity_name = activity.name[:25] + "..." if len(activity.name) > 28 else activity.name
            
            add_line(f"| {date_str} | {activity_name} | {activity.sport} | {format_time(activity.moving_time)} | "
                    f"{distance_str} | {tss_str} | {hrss_str} | {if_str} | {hr_str} | {power_str} |")
    
    add_line()
    
    add_line(f"---")
    add_line(f"*Generated by Strava Training Load Analyzer*")
    
    return lines


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
    
    # Set up directories
    project_root = Path(__file__).parent.parent
    exchange_dir = project_root.parent / "ENV" / "exchange"
    load_dir = exchange_dir / "load"
    cache_dir = load_dir / "cache"
    
    parser = argparse.ArgumentParser(
        description="Analyze Strava training load for the last 28 days with minimal API calls."
    )
    parser.add_argument(
        "--end-date", 
        help="End date for analysis period (YYYY-MM-DD format). Default: today"
    )
    parser.add_argument("--days", type=int, default=28, help="Number of days to analyze (default: 28)")
    parser.add_argument("--settings", help="Path to settings.yaml")
    parser.add_argument("--ftp", type=float, help="Override FTP for TSS calculations (watts)")
    parser.add_argument("--threshold-hr", type=float, help="Override threshold HR for HRSS calculations (bpm)")
    
    args = parser.parse_args()
    
    # Parse end date
    if args.end_date:
        try:
            end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date()
        except ValueError:
            print(f"Invalid date format: {args.end_date}. Use YYYY-MM-DD format.")
            return 1
    else:
        end_date = date.today()
    
    # Load settings
    settings: dict = {}
    cycling_ftp = args.ftp  # CLI FTP overrides cycling FTP
    running_ftp = None
    threshold_hr = args.threshold_hr
    
    if args.settings:
        settings = load_settings(args.settings)
        
        # Extract sport-specific FTPs and threshold HR from settings
        if not cycling_ftp:
            cycling_settings = settings.get("cycling", {})
            cycling_ftp = cycling_settings.get("ftp") if isinstance(cycling_settings, dict) else None
        
        # Get running FTP from settings
        running_settings = settings.get("running", {})
        if isinstance(running_settings, dict):
            running_ftp = running_settings.get("ftp")
        
        if not threshold_hr:
            hr_settings = settings.get("heart-rate", {})
            threshold_hr = hr_settings.get("lt") if isinstance(hr_settings, dict) else None
    
    # Ensure directories exist
    load_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Check for access token
        if not os.getenv('STRAVA_ACCESS_TOKEN'):
            print("ERROR: STRAVA_ACCESS_TOKEN environment variable not found.")
            print("Please create a .env file with your Strava API credentials.")
            return 1
        
        print(f"Analyzing training load for {args.days} days ending {end_date}")
        if cycling_ftp:
            print(f"Using cycling FTP: {cycling_ftp}W for TSS calculations")
        if running_ftp:
            print(f"Using running FTP: {running_ftp}W for TSS calculations")
        if threshold_hr:
            print(f"Using threshold HR: {threshold_hr} bpm for HRSS calculations")
        print()
        
        # Create client and get analysis with caching
        client = StravaClient()
        analysis = client.get_training_load_analysis(
            end_date=end_date,
            days=args.days,
            cycling_ftp=cycling_ftp,
            running_ftp=running_ftp,
            threshold_hr=threshold_hr,
            cache_dir=cache_dir
        )
        
        # Generate report
        report_lines = generate_training_load_report(analysis, settings)
        
        # Create filename and save to load directory
        start_date = analysis.period_start
        filename = f"{start_date}_to_{end_date}_training_load_analysis.md"
        
        analysis_path = load_dir / filename
        
        analysis_text = "\n".join(report_lines)
        analysis_path.write_text(analysis_text, encoding="utf-8")
        
        print(f"\nTraining load analysis saved to: {analysis_path}")
        
        # Print quick summary
        print(f"\nQuick Summary:")
        print(f"- {analysis.total_activities} activities in {args.days} days")
        print(f"- {format_time(analysis.total_time)} total training time")
        print(f"- TSS: {analysis.total_tss:.1f} (avg {analysis.avg_weekly_tss:.1f}/week)")
        print(f"- HRSS: {analysis.total_hrss:.1f} (avg {analysis.avg_weekly_hrss:.1f}/week)")
        
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())