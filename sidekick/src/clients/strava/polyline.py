"""
Polyline analysis utilities for matching routes.
Handles encoded polylines from Strava and provides route similarity matching.
"""
import math
from typing import NamedTuple


class Point(NamedTuple):
    """A geographic point with latitude and longitude."""
    lat: float
    lon: float


def decode_polyline(encoded: str) -> list[Point]:
    """
    Decode a Google-encoded polyline string into a list of coordinates.
    
    This uses the same encoding algorithm as Google Maps and Strava.
    See: https://developers.google.com/maps/documentation/utilities/polylinealgorithm
    
    Args:
        encoded: Encoded polyline string
        
    Returns:
        List of Point objects representing the route
    """
    if not encoded:
        return []
    
    points = []
    index = 0
    lat = 0
    lon = 0
    
    while index < len(encoded):
        # Decode latitude
        result = 0
        shift = 0
        while True:
            b = ord(encoded[index]) - 63
            index += 1
            result |= (b & 0x1f) << shift
            shift += 5
            if b < 0x20:
                break
        
        dlat = ~(result >> 1) if (result & 1) else (result >> 1)
        lat += dlat
        
        # Decode longitude
        result = 0
        shift = 0
        while True:
            b = ord(encoded[index]) - 63
            index += 1
            result |= (b & 0x1f) << shift
            shift += 5
            if b < 0x20:
                break
        
        dlon = ~(result >> 1) if (result & 1) else (result >> 1)
        lon += dlon
        
        points.append(Point(lat=lat / 1e5, lon=lon / 1e5))
    
    return points


def encode_polyline(points: list[Point]) -> str:
    """
    Encode a list of coordinates into a Google-encoded polyline string.
    
    Args:
        points: List of Point objects
        
    Returns:
        Encoded polyline string
    """
    if not points:
        return ""
    
    def encode_value(value: int) -> str:
        """Encode a single coordinate value."""
        value = ~(value << 1) if value < 0 else (value << 1)
        encoded = ""
        while value >= 0x20:
            encoded += chr((0x20 | (value & 0x1f)) + 63)
            value >>= 5
        encoded += chr(value + 63)
        return encoded
    
    encoded = ""
    prev_lat = 0
    prev_lon = 0
    
    for point in points:
        lat = int(round(point.lat * 1e5))
        lon = int(round(point.lon * 1e5))
        
        encoded += encode_value(lat - prev_lat)
        encoded += encode_value(lon - prev_lon)
        
        prev_lat = lat
        prev_lon = lon
    
    return encoded


def haversine_distance(p1: Point, p2: Point) -> float:
    """
    Calculate the great circle distance between two points in meters.
    
    Args:
        p1: First point
        p2: Second point
        
    Returns:
        Distance in meters
    """
    # Earth radius in meters
    R = 6371000
    
    lat1, lon1 = math.radians(p1.lat), math.radians(p1.lon)
    lat2, lon2 = math.radians(p2.lat), math.radians(p2.lon)
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    return R * c


def normalize_polyline(encoded: str, target_points: int = 50) -> str:
    """
    Normalize a polyline to have a consistent number of points.
    This makes route comparison more reliable by standardizing the resolution.
    
    Args:
        encoded: Encoded polyline string
        target_points: Target number of points for normalized polyline
        
    Returns:
        Normalized encoded polyline string
    """
    points = decode_polyline(encoded)
    
    if not points or len(points) < 2:
        return encoded
    
    # Calculate total route distance
    total_distance = 0.0
    for i in range(len(points) - 1):
        total_distance += haversine_distance(points[i], points[i + 1])
    
    if total_distance == 0:
        return encoded
    
    # Calculate cumulative distances
    cumulative_distances = [0.0]
    for i in range(len(points) - 1):
        cumulative_distances.append(
            cumulative_distances[-1] + haversine_distance(points[i], points[i + 1])
        )
    
    # Sample points at regular distance intervals
    normalized_points = [points[0]]  # Always include start
    interval = total_distance / (target_points - 1)
    
    for i in range(1, target_points - 1):
        target_distance = i * interval
        
        # Find the segment containing this distance
        for j in range(len(cumulative_distances) - 1):
            if cumulative_distances[j] <= target_distance <= cumulative_distances[j + 1]:
                # Interpolate between points[j] and points[j + 1]
                segment_distance = cumulative_distances[j + 1] - cumulative_distances[j]
                if segment_distance > 0:
                    ratio = (target_distance - cumulative_distances[j]) / segment_distance
                    lat = points[j].lat + ratio * (points[j + 1].lat - points[j].lat)
                    lon = points[j].lon + ratio * (points[j + 1].lon - points[j].lon)
                    normalized_points.append(Point(lat=lat, lon=lon))
                else:
                    normalized_points.append(points[j])
                break
    
    normalized_points.append(points[-1])  # Always include end
    
    return encode_polyline(normalized_points)


def calculate_route_similarity(polyline1: str, polyline2: str, normalize: bool = True) -> float:
    """
    Calculate the similarity between two routes based on their polylines.
    
    The similarity score is based on the average distance between corresponding
    points after normalization. Lower scores indicate more similar routes.
    
    Args:
        polyline1: First encoded polyline
        polyline2: Second encoded polyline
        normalize: Whether to normalize polylines before comparison
        
    Returns:
        Similarity score (average distance in meters between corresponding points).
        Returns 0.0 if routes are identical, higher values for less similar routes.
    """
    if not polyline1 or not polyline2:
        return float('inf')
    
    # Normalize polylines to have same number of points
    if normalize:
        polyline1 = normalize_polyline(polyline1)
        polyline2 = normalize_polyline(polyline2)
    
    points1 = decode_polyline(polyline1)
    points2 = decode_polyline(polyline2)
    
    if not points1 or not points2:
        return float('inf')
    
    if len(points1) != len(points2):
        # If normalization didn't work or wasn't used, fall back to comparing fewer points
        min_len = min(len(points1), len(points2))
        points1 = points1[:min_len]
        points2 = points2[:min_len]
    
    # Calculate average distance between corresponding points
    total_distance = 0.0
    for p1, p2 in zip(points1, points2):
        total_distance += haversine_distance(p1, p2)
    
    return total_distance / len(points1)


def is_route_match(polyline1: str, polyline2: str, threshold_meters: float = 100.0) -> bool:
    """
    Determine if two routes match within a given threshold.
    
    Args:
        polyline1: First encoded polyline
        polyline2: Second encoded polyline  
        threshold_meters: Maximum average distance (in meters) between points
                          to consider routes as matching (default: 100m)
        
    Returns:
        True if routes match within threshold, False otherwise
    """
    if not polyline1 or not polyline2:
        return False
    
    similarity = calculate_route_similarity(polyline1, polyline2)
    return similarity <= threshold_meters


def find_matching_commute(
    ride_polyline: str,
    known_commutes: dict[str, str],
    threshold_meters: float = 100.0
) -> str | None:
    """
    Find a matching commute route from a list of known commutes.
    
    Args:
        ride_polyline: Encoded polyline of the ride to check
        known_commutes: Dictionary mapping commute names to normalized polylines
        threshold_meters: Matching threshold in meters
        
    Returns:
        Name of matching commute, or None if no match found
    """
    if not ride_polyline:
        return None
    
    best_match = None
    best_score = float('inf')
    
    for commute_name, commute_polyline in known_commutes.items():
        if not commute_polyline:
            continue
        
        score = calculate_route_similarity(ride_polyline, commute_polyline)
        
        if score < best_score:
            best_score = score
            best_match = commute_name
    
    if best_score <= threshold_meters:
        return best_match
    
    return None
