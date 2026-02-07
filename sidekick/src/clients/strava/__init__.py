"""Strava client and utilities."""

from clients.strava.client import StravaClient, StravaActivity, StravaDataParser
from clients.strava.polyline import (
    Point,
    decode_polyline,
    encode_polyline,
    haversine_distance,
    normalize_polyline,
    calculate_route_similarity,
    is_route_match,
    find_matching_commute
)

__all__ = [
    "StravaClient",
    "StravaActivity",
    "StravaDataParser",
    "Point",
    "decode_polyline",
    "encode_polyline",
    "haversine_distance",
    "normalize_polyline",
    "calculate_route_similarity",
    "is_route_match",
    "find_matching_commute"
]
