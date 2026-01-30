#!/usr/bin/env python3
"""
Strava API client for downloading workout data and converting to analysis format.
Uses direct API calls to avoid dependency conflicts.
"""
import os
import json
import pandas as pd
import requests
from datetime import datetime, date, timedelta
from typing import Any
from pathlib import Path

from models.workout import (
    Workout, 
    Device,
    ActivitySummary,
    DailySummary,
    WeeklySummary,
    TrainingLoadAnalysis
)


class StravaActivity:
    """Simple Strava activity representation from API response."""
    
    def __init__(self, data: dict):
        self.data = data
        self.id = data['id']
        self.name = data['name']
        self.type = data['type']
        self.sport_type = data.get('sport_type', data['type'])
        self.start_date = datetime.fromisoformat(data['start_date'].replace('Z', '+00:00'))
        self.start_date_local = datetime.fromisoformat(data['start_date_local'].replace('Z', '+00:00'))
        self.distance = data.get('distance', 0.0)
        self.moving_time = data.get('moving_time', 0)
        self.elapsed_time = data.get('elapsed_time', 0)
        self.total_elevation_gain = data.get('total_elevation_gain')
        self.average_speed = data.get('average_speed')
        self.max_speed = data.get('max_speed')
        self.average_heartrate = data.get('average_heartrate')
        self.max_heartrate = data.get('max_heartrate')
        self.average_watts = data.get('average_watts')
        self.weighted_average_watts = data.get('weighted_average_watts')
        self.kilojoules = data.get('kilojoules')
        self.max_watts = data.get('max_watts')
        self.average_cadence = data.get('average_cadence')
        self.device_name = data.get('device_name')
        self.manual = data.get('manual', False)
        self.from_accepted_tag = data.get('from_accepted_tag', False)
        self.commute = data.get('commute', False)
        # Extract map data including summary_polyline
        map_data = data.get('map', {})
        self.summary_polyline = map_data.get('summary_polyline', '') if isinstance(map_data, dict) else ''


class StravaStream:
    """Simple Strava stream representation."""
    
    def __init__(self, stream_type: str, data: list):
        self.type = stream_type
        self.data = data


class StravaClient:
    """Strava API client with authentication and data download capabilities."""
    
    BASE_URL = "https://www.strava.com/api/v3"
    
    def __init__(self):
        
        self.access_token = os.getenv('STRAVA_ACCESS_TOKEN')
        
        if not self.access_token:
            raise ValueError(
                "STRAVA_ACCESS_TOKEN environment variable not found. "
                "Please set up your Strava API credentials in a .env file."
            )
        
        self.headers = {
            'Authorization': f'Bearer {self.access_token}',
            'Content-Type': 'application/json'
        }
    
    def get_activities_for_date(self, target_date: date) -> list[StravaActivity]:
        """
        Get all activities for a specific date.
        
        Args:
            target_date: Date to fetch activities for
            
        Returns:
            List of StravaActivity objects
        """
        # Set time range for the entire day
        start_time = datetime.combine(target_date, datetime.min.time())
        end_time = start_time + timedelta(days=1)
        
        url = f"{self.BASE_URL}/athlete/activities"
        params = {
            'after': int(start_time.timestamp()),
            'before': int(end_time.timestamp()),
            'per_page': 200  # Strava max
        }
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            activities_data = response.json()
            
            return [StravaActivity(activity_data) for activity_data in activities_data]
        
        except requests.exceptions.RequestException as e:
            print(f"Error fetching activities: {e}")
            return []
    
    def get_activity_streams(self, activity_id: int) -> dict[str, StravaStream]:
        """
        Get detailed stream data for an activity.
        
        Args:
            activity_id: Strava activity ID
            
        Returns:
            Dictionary of stream type to StravaStream objects
        """
        stream_types = [
            'time', 'distance', 'altitude', 'velocity_smooth', 
            'heartrate', 'cadence', 'watts', 'temp', 'moving', 'grade_smooth'
        ]
        
        url = f"{self.BASE_URL}/activities/{activity_id}/streams"
        params = {
            'keys': ','.join(stream_types),
            'key_by_type': 'true'
        }
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            streams_data = response.json()
            
            streams = {}
            for stream_type, stream_info in streams_data.items():
                if 'data' in stream_info:
                    streams[stream_type] = StravaStream(stream_type, stream_info['data'])
            
            return streams
        
        except requests.exceptions.RequestException as e:
            print(f"Warning: Could not fetch streams for activity {activity_id}: {e}")
            return {}
    
    def get_activity_laps(self, activity_id: int) -> list[dict[str, Any]]:
        """
        Get lap data for an activity.
        
        Args:
            activity_id: Strava activity ID
            
        Returns:
            List of lap dictionaries
        """
        url = f"{self.BASE_URL}/activities/{activity_id}/laps"
        
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            laps_data = response.json()
            
            return laps_data if isinstance(laps_data, list) else []
        
        except requests.exceptions.RequestException as e:
            print(f"Warning: Could not fetch laps for activity {activity_id}: {e}")
            return []

    def _get_cache_path(self, cache_dir: Path, target_date: date) -> Path:
        """Get cache file path for a specific date."""
        return cache_dir / f"{target_date.isoformat()}_activities.json"

    def _load_cached_activities(self, cache_path: Path) -> list[StravaActivity] | None:
        """Load activities from cache if available and recent."""
        if not cache_path.exists():
            return None
            
        try:
            # Check if cache is less than 1 day old
            cache_age = datetime.now().timestamp() - cache_path.stat().st_mtime
            if cache_age > 24 * 3600:  # 24 hours
                return None
                
            with open(cache_path, 'r') as f:
                data = json.load(f)
                return [StravaActivity(activity_data) for activity_data in data]
        except (json.JSONDecodeError, KeyError, OSError):
            return None

    def _save_activities_cache(self, cache_path: Path, activities: list[StravaActivity]):
        """Save activities to cache."""
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, 'w') as f:
                json.dump([activity.data for activity in activities], f)
        except OSError:
            pass  # Silently ignore cache write failures

    def get_activities_for_date_cached(self, target_date: date, cache_dir: Path | None = None) -> list[StravaActivity]:
        """
        Get activities for a date with caching support.
        
        Args:
            target_date: Date to fetch activities for
            cache_dir: Directory for cache files (optional)
            
        Returns:
            List of StravaActivity objects
        """
        # Skip caching if no cache directory provided
        if cache_dir is None:
            return self.get_activities_for_date(target_date)
            
        cache_path = self._get_cache_path(cache_dir, target_date)
        
        # Try to load from cache first
        cached_activities = self._load_cached_activities(cache_path)
        if cached_activities is not None:
            print(f"  {target_date}: Using cached data ({len(cached_activities)} activities)")
            return cached_activities
        
        # Fetch from API if not cached
        activities = self.get_activities_for_date(target_date)
        
        # Save to cache
        self._save_activities_cache(cache_path, activities)
        
        return activities

    def get_training_load_analysis(
        self,
        end_date: date,
        days: int = 28,
        cycling_ftp: float | None = None,
        running_ftp: float | None = None,
        threshold_hr: float | None = None,
        cache_dir: Path | None = None
    ) -> TrainingLoadAnalysis:
        """
        Get comprehensive training load analysis for a specified period.
        Uses only activity summary data for efficient API usage.
        
        Args:
            end_date: End date for analysis period
            days: Number of days to analyze (default 28)
            ftp: Functional Threshold Power for TSS calculations
            threshold_hr: Lactate threshold heart rate for HRSS calculations
            
        Returns:
            TrainingLoadAnalysis with complete 28-day overview
        """
        start_date = end_date - timedelta(days=days-1)
        
        print(f"Fetching training load data from {start_date} to {end_date}")
        print(f"This will require approximately {days} API calls...")
        if cycling_ftp:
            print(f"Using cycling FTP: {cycling_ftp}W")
        if running_ftp:
            print(f"Using running FTP: {running_ftp}W")
        
        # Fetch activities for each day (efficient: 1 API call per day)
        daily_summaries = []
        
        for i in range(days):
            current_date = start_date + timedelta(days=i)
            activities = self.get_activities_for_date_cached(current_date, cache_dir)
            
            # Convert to activity summaries and calculate training load
            activity_summaries = []
            for activity in activities:
                summary = ActivitySummary.from_strava_activity(activity)
                summary.calculate_training_load(cycling_ftp=cycling_ftp, running_ftp=running_ftp, threshold_hr=threshold_hr)
                activity_summaries.append(summary)
            
            daily_summary = DailySummary(
                date=current_date,
                activities=activity_summaries
            )
            daily_summaries.append(daily_summary)
            
            if activity_summaries:
                print(f"  {current_date}: {len(activity_summaries)} activities")
        
        # Create weekly summaries (rolling 7-day windows)
        weekly_summaries = []
        weeks = days // 7
        
        for week in range(weeks):
            week_start = week * 7
            week_end = min(week_start + 7, len(daily_summaries))
            week_days = daily_summaries[week_start:week_end]
            
            if week_days:
                weekly_summary = WeeklySummary(
                    start_date=week_days[0].date,
                    end_date=week_days[-1].date,
                    days=week_days
                )
                weekly_summaries.append(weekly_summary)
        
        # Create complete analysis
        analysis = TrainingLoadAnalysis(
            period_start=start_date,
            period_end=end_date,
            daily_summaries=daily_summaries,
            weekly_summaries=weekly_summaries
        )
        
        print(f"Analysis complete: {analysis.total_activities} activities, "
              f"{analysis.total_time/3600:.1f} hours, "
              f"TSS: {analysis.total_tss:.1f}")
        
        return analysis

    def get_activities_for_period(self, start_date: date, end_date: date) -> list[StravaActivity]:
        """
        Get all activities for a date range using efficient batch API calls.
        
        Args:
            start_date: Start date for period
            end_date: End date for period
            
        Returns:
            List of all StravaActivity objects in the period
        """
        # Use single API call for entire period if <= 30 days
        if (end_date - start_date).days <= 30:
            start_time = datetime.combine(start_date, datetime.min.time())
            end_time = datetime.combine(end_date + timedelta(days=1), datetime.min.time())
            
            url = f"{self.BASE_URL}/athlete/activities"
            params = {
                'after': int(start_time.timestamp()),
                'before': int(end_time.timestamp()),
                'per_page': 200
            }
            
            try:
                response = requests.get(url, headers=self.headers, params=params)
                response.raise_for_status()
                activities_data = response.json()
                
                return [StravaActivity(activity_data) for activity_data in activities_data]
                
            except requests.exceptions.RequestException as e:
                print(f"Error fetching activities for period: {e}")
                return []
        else:
            # For longer periods, fetch day by day to respect API limits
            all_activities = []
            current_date = start_date
            
            while current_date <= end_date:
                daily_activities = self.get_activities_for_date(current_date)
                all_activities.extend(daily_activities)
                current_date += timedelta(days=1)
            
            return all_activities


class StravaDataParser:
    """Parser to convert Strava data to format compatible with existing analysis."""
    
    def __init__(
        self,
        activity: StravaActivity,
        streams: dict[str, StravaStream] | None = None,
        laps: list[dict[str, Any]] | None = None,
        commute_status: str = "no"
    ):
        self.activity = activity
        self.streams = streams or {}
        self.strava_laps = laps or []
        self.data_frame = self._create_dataframe()
        self.workout = self._create_workout()
        self.laps = self._create_laps()
        self.sets = []  # Strava doesn't typically have strength set data in streams
        # Expose commute and polyline data
        self.commute = activity.commute
        self.summary_polyline = activity.summary_polyline
        self.commute_status = commute_status
    
    def _create_dataframe(self) -> pd.DataFrame:
        """Create a pandas DataFrame from Strava stream data."""
        if not self.streams:
            # If no streams available, create minimal dataframe with activity summary
            return pd.DataFrame()
        
        # Get time stream as the base
        time_stream = self.streams.get('time')
        if not time_stream:
            return pd.DataFrame()
        
        # Create datetime index
        start_time = self.activity.start_date_local or self.activity.start_date
        timestamps = [start_time + timedelta(seconds=int(t)) for t in time_stream.data]
        
        # Build dataframe with all available streams
        data = {}
        
        # Map stream types to column names used in existing analysis
        stream_mapping = {
            'heartrate': 'heart_rate',
            'watts': 'power',
            'velocity_smooth': 'speed',
            'grade_smooth': 'grade',
            'temp': 'temperature',
            'cadence': 'cadence',
            'altitude': 'altitude',
            'distance': 'distance',
            'moving': 'moving'
        }
        
        for stream_type, column_name in stream_mapping.items():
            stream = self.streams.get(stream_type)
            if stream and len(stream.data) == len(timestamps):
                data[column_name] = stream.data
        
        df = pd.DataFrame(data, index=pd.DatetimeIndex(timestamps))
        
        # Convert units to match FIT file format
        if 'speed' in df.columns:
            # Convert m/s to km/h if needed by analysis (check existing code)
            pass  # Keep as m/s for now
            
        return df.sort_index()
    
    def _create_workout(self) -> Workout:
        """Convert Strava Activity to Workout model."""
        # Create device representation
        device = Device()
        device._manufacturer = "strava"
        device._product = self.activity.device_name or "Strava"
        device._sensors = []
        
        # Map Strava sport types to internal sport categories
        sport_mapping = {
            'Ride': 'cycling',
            'Run': 'running',
            'VirtualRide': 'cycling', 
            'Workout': 'training',
            'WeightTraining': 'training',
            'CrossCountrySkiing': 'cross_country_skiing',
            'NordicSki': 'cross_country_skiing',
        }
        
        sport_type = str(self.activity.sport_type) if self.activity.sport_type else 'unknown'
        sport = sport_mapping.get(sport_type, sport_type.lower())
        
        # Determine sub_sport
        sub_sport = None
        if sport_type in ['Workout', 'WeightTraining']:
            sub_sport = 'strength_training'
        
        workout = Workout(
            name=self.activity.name,
            start_time=self.activity.start_date_local or self.activity.start_date,
            sport=sport,
            sub_sport=sub_sport,
            device=device
        )
        
        # Set distance
        distance_m = float(self.activity.distance) if self.activity.distance else 0.0
        workout._distance = distance_m
        
        return workout
    
    def _create_laps(self) -> list[dict[str, Any]]:
        """Create lap data from Strava laps API or fallback to single lap for entire activity."""
        if not self.strava_laps:
            # Fallback to single lap for entire activity
            if not self.data_frame.empty:
                start_time = self.data_frame.index[0]
                end_time = self.data_frame.index[-1]
                
                return [{
                    'start': start_time,
                    'end': end_time,
                    'label': 'Full Activity',
                    'intensity': None
                }]
            else:
                return []
        
        # Convert Strava lap data to format expected by analysis
        laps = []
        base_start = self.activity.start_date_local or self.activity.start_date
        
        for i, strava_lap in enumerate(self.strava_laps):
            # Extract lap timing info
            elapsed_time = strava_lap.get('elapsed_time', 0)
            start_index = strava_lap.get('start_index', i * 1000)  # Fallback estimation
            
            # Calculate lap timestamps
            if i == 0:
                lap_start = base_start
            else:
                # Use elapsed time to estimate start relative to previous laps
                prev_elapsed = sum(lap.get('elapsed_time', 0) for lap in self.strava_laps[:i])
                lap_start = base_start + timedelta(seconds=prev_elapsed)
            
            lap_end = lap_start + timedelta(seconds=elapsed_time)
            
            # Create lap entry
            lap_entry = {
                'start': pd.Timestamp(lap_start),
                'end': pd.Timestamp(lap_end),
                'label': f"Lap {i + 1}",
                'intensity': None,  # Strava doesn't provide intensity in same format as Garmin
                'strava_data': strava_lap  # Keep original Strava data for additional metrics
            }
            
            laps.append(lap_entry)
        
        return laps


def download_strava_workouts(target_date: date) -> list[StravaDataParser]:
    """
    Download all Strava workouts for a given date.
    
    Args:
        target_date: Date to download workouts for
        
    Returns:
        List of StravaDataParser objects ready for analysis
    """
    client = StravaClient()
    activities = client.get_activities_for_date(target_date)
    
    parsers = []
    for activity in activities:
        print(f"Downloading data for activity: {activity.name} ({activity.id})")
        
        # Download streams and laps data
        streams = client.get_activity_streams(activity.id)
        laps = client.get_activity_laps(activity.id)
        
        if laps:
            print(f"  Found {len(laps)} lap(s)")
        else:
            print(f"  No laps found, using entire activity as single lap")
        
        parser = StravaDataParser(activity, streams, laps)
        parsers.append(parser)
    
    return parsers