#!/usr/bin/env python3
"""
Strava API client for downloading workout data and converting to analysis format.
Uses direct API calls to avoid dependency conflicts.
"""
import os
import pandas as pd
import requests
from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Optional
from pathlib import Path

from format.models import Workout, Device, Sensor


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
    
    def get_activities_for_date(self, target_date: date) -> List[StravaActivity]:
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
    
    def get_activity_streams(self, activity_id: int) -> Dict[str, StravaStream]:
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


class StravaDataParser:
    """Parser to convert Strava data to format compatible with existing analysis."""
    
    def __init__(self, activity: StravaActivity, streams: Dict[str, StravaStream] | None = None):
        self.activity = activity
        self.streams = streams or {}
        self.data_frame = self._create_dataframe()
        self.workout = self._create_workout()
        self.laps = self._create_laps()
        self.sets = []  # Strava doesn't typically have strength set data in streams
    
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
    
    def _create_laps(self) -> List[Dict[str, Any]]:
        """Create lap data. Strava doesn't always have detailed laps, so we'll create a single lap for the entire activity."""
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


def download_strava_workouts(target_date: date) -> List[StravaDataParser]:
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
        print(f"Downloading streams for activity: {activity.name} ({activity.id})")
        streams = client.get_activity_streams(activity.id)
        parser = StravaDataParser(activity, streams)
        parsers.append(parser)
    
    return parsers