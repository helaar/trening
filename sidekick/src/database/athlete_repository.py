from datetime import datetime, timezone
from pymongo.asynchronous.database import AsyncDatabase

from models.athlete import Athlete, StravaTokens


class AthleteRepository:
    """Repository for athlete and token management in MongoDB."""
    
    def __init__(self, db: AsyncDatabase):
        self.db = db
        self.athletes_collection = db["athletes"]
        self.tokens_collection = db["strava_tokens"]
    
    async def get_athlete(self, athlete_id: int) -> Athlete | None:
        """Get athlete by ID."""
        doc = await self.athletes_collection.find_one({"athlete_id": athlete_id})
        if doc:
            doc.pop("_id", None)
            return Athlete(**doc)
        return None
    
    async def create_or_update_athlete(self, athlete: Athlete) -> Athlete:
        """Create or update athlete information."""
        athlete.updated_at = datetime.now(timezone.utc)
        
        await self.athletes_collection.update_one(
            {"athlete_id": athlete.athlete_id},
            {"$set": athlete.model_dump()},
            upsert=True
        )
        return athlete
    
    async def get_tokens(self, athlete_id: int) -> StravaTokens | None:
        """Get Strava tokens for an athlete."""
        doc = await self.tokens_collection.find_one({"athlete_id": athlete_id})
        if doc:
            doc.pop("_id", None)
            return StravaTokens(**doc)
        return None
    
    async def save_tokens(self, tokens: StravaTokens) -> StravaTokens:
        """Save or update Strava tokens for an athlete."""
        tokens.updated_at = datetime.now(timezone.utc)
        
        await self.tokens_collection.update_one(
            {"athlete_id": tokens.athlete_id},
            {"$set": tokens.model_dump()},
            upsert=True
        )
        return tokens
    
    async def delete_tokens(self, athlete_id: int) -> bool:
        """Delete tokens for an athlete (disconnect)."""
        result = await self.tokens_collection.delete_one({"athlete_id": athlete_id})
        return result.deleted_count > 0
    
    async def delete_athlete(self, athlete_id: int) -> bool:
        """Delete athlete and their tokens."""
        await self.delete_tokens(athlete_id)
        result = await self.athletes_collection.delete_one({"athlete_id": athlete_id})
        return result.deleted_count > 0
