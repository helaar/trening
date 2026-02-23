from datetime import datetime, timezone
from pymongo.asynchronous.database import AsyncDatabase

from models.athlete import Athlete, StravaTokens, AthleteSettings, SportSettings, HeartRateSettings


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
        """
        Create or update athlete information.
        
        DEPRECATED: Use ensure_athlete_exists() and update_athlete_profile() instead
        to avoid overwriting settings.
        """
        athlete.updated_at = datetime.now(timezone.utc)
        
        await self.athletes_collection.update_one(
            {"athlete_id": athlete.athlete_id},
            {"$set": athlete.model_dump()},
            upsert=True
        )
        return athlete
    
    async def ensure_athlete_exists(
        self,
        athlete_id: int,
        username: str | None = None,
        firstname: str | None = None,
        lastname: str | None = None,
        profile_picture: str | None = None
    ) -> Athlete:
        """
        Ensure athlete exists, creating with defaults if needed.
        
        This method only creates the athlete if they don't exist.
        Use update_athlete_profile() to update profile fields.
        """
        existing = await self.get_athlete(athlete_id)
        if existing:
            return existing
        
        # Create new athlete with default settings
        athlete = Athlete(
            athlete_id=athlete_id,
            username=username,
            firstname=firstname,
            lastname=lastname,
            profile_picture=profile_picture
        )
        
        await self.athletes_collection.insert_one(athlete.model_dump())
        return athlete
    
    async def update_athlete_profile(
        self,
        athlete_id: int,
        username: str | None = None,
        firstname: str | None = None,
        lastname: str | None = None,
        profile_picture: str | None = None
    ) -> Athlete | None:
        """
        Update only profile fields from Strava OAuth.
        
        This preserves all settings and only updates profile information.
        """
        update_fields: dict[str, str | datetime] = {"updated_at": datetime.now(timezone.utc)}
        
        if username is not None:
            update_fields["username"] = username
        if firstname is not None:
            update_fields["firstname"] = firstname
        if lastname is not None:
            update_fields["lastname"] = lastname
        if profile_picture is not None:
            update_fields["profile_picture"] = profile_picture
        
        result = await self.athletes_collection.update_one(
            {"athlete_id": athlete_id},
            {"$set": update_fields}
        )
        
        if result.matched_count > 0:
            return await self.get_athlete(athlete_id)
        return None
    
    async def update_commute_routes(
        self,
        athlete_id: int,
        routes: dict[str, str]
    ) -> Athlete | None:
        """
        Update only commute routes, preserving other settings.
        
        Args:
            athlete_id: Athlete ID
            routes: Dictionary mapping route names to polylines
        """
        result = await self.athletes_collection.update_one(
            {"athlete_id": athlete_id},
            {
                "$set": {
                    "settings.commute_routes": routes,
                    "updated_at": datetime.now(timezone.utc)
                }
            }
        )
        
        if result.matched_count > 0:
            return await self.get_athlete(athlete_id)
        return None
    
    async def update_sport_settings(
        self,
        athlete_id: int,
        sport: str,
        settings: SportSettings
    ) -> Athlete | None:
        """
        Update cycling or running settings only.
        
        Args:
            athlete_id: Athlete ID
            sport: 'cycling' or 'running'
            settings: SportSettings object with FTP and zones
        """
        if sport not in ["cycling", "running"]:
            raise ValueError(f"Invalid sport: {sport}. Must be 'cycling' or 'running'")
        
        result = await self.athletes_collection.update_one(
            {"athlete_id": athlete_id},
            {
                "$set": {
                    f"settings.{sport}": settings.model_dump(),
                    "updated_at": datetime.now(timezone.utc)
                }
            }
        )
        
        if result.matched_count > 0:
            return await self.get_athlete(athlete_id)
        return None
    
    async def update_heart_rate_settings(
        self,
        athlete_id: int,
        hr_settings: HeartRateSettings
    ) -> Athlete | None:
        """
        Update heart rate settings only.
        
        Args:
            athlete_id: Athlete ID
            hr_settings: HeartRateSettings object with HR zones
        """
        result = await self.athletes_collection.update_one(
            {"athlete_id": athlete_id},
            {
                "$set": {
                    "settings.heart_rate": hr_settings.model_dump(),
                    "updated_at": datetime.now(timezone.utc)
                }
            }
        )
        
        if result.matched_count > 0:
            return await self.get_athlete(athlete_id)
        return None
    
    async def update_athlete_settings(self, athlete_id: int, settings: AthleteSettings) -> Athlete | None:
        """Update athlete settings."""
        result = await self.athletes_collection.update_one(
            {"athlete_id": athlete_id},
            {
                "$set": {
                    "settings": settings.model_dump(),
                    "updated_at": datetime.now(timezone.utc)
                }
            }
        )
        
        if result.matched_count > 0:
            return await self.get_athlete(athlete_id)
        return None
    
    async def get_athlete_settings(self, athlete_id: int) -> AthleteSettings | None:
        """Get athlete settings."""
        athlete = await self.get_athlete(athlete_id)
        return athlete.settings if athlete else None
    
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
