"""
Commute detection service for identifying and managing commute routes.

This service detects whether an activity matches known commute routes and can
store new commute routes for future detection.
"""
from clients.strava.client import StravaActivity
from clients.strava.polyline import (
    normalize_polyline,
    find_matching_commute,
    calculate_route_similarity
)
from database.athlete_repository import AthleteRepository
from models.athlete import AthleteSettings


class CommuteDetectionService:
    """Service for detecting and managing commute routes."""
    
    def __init__(self, athlete_repo: AthleteRepository):
        """
        Initialize commute detection service.
        
        Args:
            athlete_repo: Repository for accessing athlete settings
        """
        self.athlete_repo = athlete_repo
    
    async def detect_commute(
        self,
        athlete_id: int,
        activity: StravaActivity,
        threshold_meters: float = 100.0
    ) -> tuple[bool, str | None]:
        """
        Detect if an activity matches a known commute route.
        
        Args:
            athlete_id: Athlete ID
            activity: Strava activity to check
            threshold_meters: Maximum average distance between points to consider a match
            
        Returns:
            Tuple of (is_commute, commute_name). If is_commute is True, commute_name
            contains the name of the matching route. Otherwise commute_name is None.
        """
        # Check if activity was manually marked as commute by athlete
        if activity.commute:
            return True, "marked by athlete"
        
        # Get activity polyline
        polyline = activity.summary_polyline
        if not polyline:
            return False, None
        
        # Get athlete settings with commute routes
        settings = await self.athlete_repo.get_athlete_settings(athlete_id)
        if not settings or not settings.commute_routes:
            return False, None
        
        # Find matching commute route
        matching_route = find_matching_commute(
            polyline,
            settings.commute_routes,
            threshold_meters
        )
        
        if matching_route:
            return True, matching_route
        
        return False, None
    
    async def add_commute_route(
        self,
        athlete_id: int,
        route_name: str,
        polyline: str,
        normalize: bool = True
    ) -> None:
        """
        Add a new commute route to athlete's settings.
        
        Args:
            athlete_id: Athlete ID
            route_name: Name for the commute route
            polyline: Encoded polyline string
            normalize: Whether to normalize the polyline before storing
        """
        # Normalize polyline for consistent matching
        if normalize:
            polyline = normalize_polyline(polyline)
        
        # Get current settings
        settings = await self.athlete_repo.get_athlete_settings(athlete_id)
        if not settings:
            settings = AthleteSettings()
        
        # Add route to commute_routes
        settings.commute_routes[route_name] = polyline
        
        # Update athlete settings
        await self.athlete_repo.update_athlete_settings(athlete_id, settings)
    
    async def remove_commute_route(
        self,
        athlete_id: int,
        route_name: str
    ) -> bool:
        """
        Remove a commute route from athlete's settings.
        
        Args:
            athlete_id: Athlete ID
            route_name: Name of the route to remove
            
        Returns:
            True if route was removed, False if route didn't exist
        """
        settings = await self.athlete_repo.get_athlete_settings(athlete_id)
        if not settings or route_name not in settings.commute_routes:
            return False
        
        # Remove route
        del settings.commute_routes[route_name]
        
        # Update athlete settings
        await self.athlete_repo.update_athlete_settings(athlete_id, settings)
        return True
    
    async def list_commute_routes(
        self,
        athlete_id: int
    ) -> dict[str, str]:
        """
        List all commute routes for an athlete.
        
        Args:
            athlete_id: Athlete ID
            
        Returns:
            Dictionary mapping route names to polylines
        """
        settings = await self.athlete_repo.get_athlete_settings(athlete_id)
        if not settings:
            return {}
        return settings.commute_routes.copy()
    
    async def get_route_similarity(
        self,
        polyline1: str,
        polyline2: str
    ) -> float:
        """
        Calculate similarity between two routes.
        
        Args:
            polyline1: First encoded polyline
            polyline2: Second encoded polyline
            
        Returns:
            Similarity score (average distance in meters between corresponding points)
        """
        return calculate_route_similarity(polyline1, polyline2)
