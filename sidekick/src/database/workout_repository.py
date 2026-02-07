import hashlib
import json
from datetime import datetime
from typing import Any

from pymongo.asynchronous.database import AsyncDatabase

from models.athlete import AthleteSettings
from models.strava_activity import StravaActivityRaw, WorkoutAnalysisData


class WorkoutRepository:
    """Repository for workout/activity data operations."""
    
    def __init__(self, db: AsyncDatabase):
        self.db = db
        self.activities_collection = db["strava_activities"]
        self.analyses_collection = db["workout_analyses"]
    
    async def store_activity(self, activity: StravaActivityRaw) -> bool:
        """
        Store or update a Strava activity in the database.
        
        Args:
            activity: StravaActivityRaw object to store
            
        Returns:
            True if stored successfully
        """
        activity_dict = activity.model_dump()
        
        # Use upsert to handle both insert and update
        result = await self.activities_collection.update_one(
            {
                "athlete_id": activity.athlete_id,
                "activity_id": activity.activity_id
            },
            {
                "$set": {
                    **activity_dict,
                    "updated_at": datetime.utcnow()
                }
            },
            upsert=True
        )
        
        return result.acknowledged
    
    async def get_activities_for_date(
        self,
        athlete_id: int,
        activity_date: datetime
    ) -> list[StravaActivityRaw]:
        """
        Get all activities for a specific athlete and date.
        
        Args:
            athlete_id: Athlete ID
            activity_date: Date to fetch activities for
            
        Returns:
            List of StravaActivityRaw objects
        """
        cursor = self.activities_collection.find({
            "athlete_id": athlete_id,
            "activity_date": activity_date
        })
        
        activities = []
        async for doc in cursor:
            # Remove MongoDB _id field before creating model
            doc.pop("_id", None)
            activities.append(StravaActivityRaw(**doc))
        
        return activities
    
    async def get_activity(
        self,
        athlete_id: int,
        activity_id: int
    ) -> StravaActivityRaw | None:
        """
        Get a specific activity by ID.
        
        Args:
            athlete_id: Athlete ID
            activity_id: Strava activity ID
            
        Returns:
            StravaActivityRaw object or None if not found
        """
        doc = await self.activities_collection.find_one({
            "athlete_id": athlete_id,
            "activity_id": activity_id
        })
        
        if doc:
            # Remove MongoDB _id field before creating model
            doc.pop("_id", None)
            return StravaActivityRaw(**doc)
        
        return None
    
    async def activity_exists(
        self,
        athlete_id: int,
        activity_id: int
    ) -> bool:
        """
        Check if an activity exists in the database.
        
        Args:
            athlete_id: Athlete ID
            activity_id: Strava activity ID
            
        Returns:
            True if activity exists
        """
        count = await self.activities_collection.count_documents({
            "athlete_id": athlete_id,
            "activity_id": activity_id
        })
        
        return count > 0
    
    async def delete_activity(
        self,
        athlete_id: int,
        activity_id: int
    ) -> bool:
        """
        Delete an activity from the database.
        
        Args:
            athlete_id: Athlete ID
            activity_id: Strava activity ID
            
        Returns:
            True if deleted successfully
        """
        result = await self.activities_collection.delete_one({
            "athlete_id": athlete_id,
            "activity_id": activity_id
        })
        
        return result.deleted_count > 0
    
    async def get_analysis(
        self,
        athlete_id: int,
        activity_id: int
    ) -> tuple[dict[str, Any], str] | None:
        """
        Get cached analysis for an activity.
        
        Returns the cached analysis regardless of whether settings have changed.
        The caller should use the settings_hash to determine if user should be
        notified about stale data, but never automatically invalidate the cache.
        
        Args:
            athlete_id: Athlete ID
            activity_id: Strava activity ID
            
        Returns:
            Tuple of (analysis_data dict, settings_hash) or None if not found
        """
        # Find the analysis
        doc = await self.analyses_collection.find_one({
            "athlete_id": athlete_id,
            "activity_id": activity_id
        })
        
        if not doc:
            return None
        
        return (doc.get("analysis_data"), doc.get("settings_hash", ""))
    
    async def store_analysis(
        self,
        athlete_id: int,
        activity_id: int,
        analysis_data: dict[str, Any],
        settings: AthleteSettings | None = None
    ) -> bool:
        """
        Store analysis results for an activity.
        
        Args:
            athlete_id: Athlete ID
            activity_id: Strava activity ID
            analysis_data: Serialized WorkoutAnalysis data
            settings: Athlete settings used for analysis
            
        Returns:
            True if stored successfully
        """
        settings_hash = self._hash_settings(settings) if settings else ""
        
        analysis_obj = WorkoutAnalysisData(
            athlete_id=athlete_id,
            activity_id=activity_id,
            settings_hash=settings_hash,
            analysis_data=analysis_data
        )
        
        analysis_dict = analysis_obj.model_dump()
        
        # Use upsert to handle both insert and update
        result = await self.analyses_collection.update_one(
            {
                "athlete_id": athlete_id,
                "activity_id": activity_id
            },
            {
                "$set": {
                    **analysis_dict,
                    "updated_at": datetime.utcnow()
                }
            },
            upsert=True
        )
        
        return result.acknowledged
    
    async def delete_analysis(
        self,
        athlete_id: int,
        activity_id: int
    ) -> bool:
        """
        Delete cached analysis for an activity.
        
        Args:
            athlete_id: Athlete ID
            activity_id: Strava activity ID
            
        Returns:
            True if deleted successfully
        """
        result = await self.analyses_collection.delete_one({
            "athlete_id": athlete_id,
            "activity_id": activity_id
        })
        
        return result.deleted_count > 0
    
    def _hash_settings(self, settings: AthleteSettings | None) -> str:
        """
        Create a hash of athlete settings to detect changes.
        
        This hash is used for cache invalidation - when athlete settings change
        (e.g., FTP is updated), the hash will differ and cached analyses will be
        considered stale, forcing recomputation with the new settings.
        
        Args:
            settings: Athlete settings object
            
        Returns:
            SHA-256 hash string of settings that affect analysis results
        """
        if not settings:
            return ""
        
        # Extract only the fields that directly affect workout analysis results.
        # This includes FTP values (used for TSS, IF, NP calculations) and zone
        # definitions (used for time-in-zone calculations). Other fields like
        # measured_date or commute_routes don't affect analysis and are excluded.
        relevant_data = {
            "cycling_ftp": settings.cycling.ftp if settings.cycling else None,
            "cycling_power_zones": [z.model_dump() for z in settings.cycling.power_zones] if settings.cycling else None,
            "running_ftp": settings.running.ftp if settings.running else None,
            "running_power_zones": [z.model_dump() for z in settings.running.power_zones] if settings.running else None,
            "hr_lt": settings.heart_rate.lt if settings.heart_rate else None,
            "hr_zones": [z.model_dump() for z in settings.heart_rate.hr_zones] if settings.heart_rate else None,
        }
        
        # Create consistent JSON string (sorted keys ensure same hash for same data)
        # and compute SHA-256 hash for efficient comparison
        json_str = json.dumps(relevant_data, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()
