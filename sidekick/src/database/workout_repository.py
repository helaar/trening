from datetime import date, datetime
from pymongo.asynchronous.database import AsyncDatabase

from models.strava_activity import StravaActivityRaw


class WorkoutRepository:
    """Repository for workout/activity data operations."""
    
    def __init__(self, db: AsyncDatabase):
        self.db = db
        self.activities_collection = db["strava_activities"]
    
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
        activity_date: date
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
            "activity_date": activity_date.isoformat()
        })
        
        activities = []
        async for doc in cursor:
            # Remove MongoDB _id field before creating model
            doc.pop("_id", None)
            activities.append(StravaActivityRaw(**doc))
        
        return activities
    
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
