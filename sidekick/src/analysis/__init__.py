"""
Workout analysis engine.

Provides comprehensive analysis for endurance and strength workouts including:
- Lap analysis with power/HR metrics
- Zone distributions (power and HR)
- Heart rate drift analysis
- ERG mode detection
- Normalized power and training metrics
"""

from analysis.engine import analyze_workout, analyze_endurance_workout, analyze_strength_workout
from analysis.models import (
    WorkoutAnalysis, SessionInfo, WorkoutMetrics, StatsSummary,
    ZoneAnalysis, ZoneDistribution, ZoneInfo,
    HeartRateDrift, LapAnalysis, ERGAnalysis
)
from analysis.calculations import (
    Zone, normalized_power, intensity_factor, training_stress_score,
    series_stats, compute_zone_durations, compute_heart_rate_drift
)

__all__ = [
    # Main analysis functions
    'analyze_workout',
    'analyze_endurance_workout', 
    'analyze_strength_workout',
    
    # Result models
    'WorkoutAnalysis',
    'SessionInfo',
    'WorkoutMetrics',
    'StatsSummary',
    'ZoneAnalysis',
    'ZoneDistribution',
    'ZoneInfo',
    'HeartRateDrift',
    'LapAnalysis',
    'ERGAnalysis',
    
    # Calculation functions
    'Zone',
    'normalized_power',
    'intensity_factor',
    'training_stress_score',
    'series_stats',
    'compute_zone_durations',
    'compute_heart_rate_drift',
]
