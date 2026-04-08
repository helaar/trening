from datetime import datetime, timezone

from pymongo.asynchronous.database import AsyncDatabase

from models.daily_entry import DailyEntry, DailyEntryRequest


class DailyEntryRepository:
    """Repository for daily entry (restitution + self-assessment) data."""

    def __init__(self, db: AsyncDatabase):
        self.collection = db["daily_entries"]

    async def get(self, athlete_id: int, date: str) -> DailyEntry | None:
        """Get daily entry for an athlete on a specific date."""
        doc = await self.collection.find_one({"athlete_id": athlete_id, "date": date})
        if doc:
            doc.pop("_id", None)
            return DailyEntry(**doc)
        return None

    async def upsert(self, athlete_id: int, request: DailyEntryRequest) -> DailyEntry:
        """Create or update daily entry, preserving original created_at on update."""
        now = datetime.now(timezone.utc)
        existing = await self.get(athlete_id, request.date)
        created_at = existing.created_at if existing else now

        entry = DailyEntry(
            athlete_id=athlete_id,
            date=request.date,
            restitution=request.restitution,
            activity_assessments=request.activity_assessments,
            created_at=created_at,
            updated_at=now,
        )

        await self.collection.update_one(
            {"athlete_id": athlete_id, "date": request.date},
            {"$set": entry.model_dump(mode="json")},
            upsert=True,
        )
        return entry
