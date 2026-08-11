from datetime import datetime, timezone

from pymongo.asynchronous.database import AsyncDatabase

from models.week_category import WeekCategoryEntry, WeekCategoryEntryRequest


class WeekCategoryRepository:
    """Repository for athlete/coach-declared weekly training intensity category."""

    def __init__(self, db: AsyncDatabase):
        self.collection = db["week_categories"]

    async def ensure_indexes(self) -> None:
        await self.collection.create_index([("athlete_id", 1), ("week_start", 1)], unique=True)

    async def get(self, athlete_id: int, week_start: str) -> WeekCategoryEntry | None:
        """Get the week category for an athlete's given week (week_start = Monday, YYYY-MM-DD)."""
        doc = await self.collection.find_one({"athlete_id": athlete_id, "week_start": week_start})
        if doc:
            doc.pop("_id", None)
            return WeekCategoryEntry(**doc)
        return None

    async def get_range(
        self, athlete_id: int, start_week: str, end_week: str
    ) -> list[WeekCategoryEntry]:
        """Get all week categories for an athlete within an inclusive week_start range."""
        cursor = self.collection.find(
            {
                "athlete_id": athlete_id,
                "week_start": {"$gte": start_week, "$lte": end_week},
            }
        ).sort("week_start", 1)
        entries = []
        async for doc in cursor:
            doc.pop("_id", None)
            entries.append(WeekCategoryEntry(**doc))
        return entries

    async def upsert(self, athlete_id: int, request: WeekCategoryEntryRequest) -> WeekCategoryEntry:
        """Create or update the week category, preserving original created_at on update."""
        now = datetime.now(timezone.utc)
        existing = await self.get(athlete_id, request.week_start)
        created_at = existing.created_at if existing else now

        entry = WeekCategoryEntry(
            athlete_id=athlete_id,
            week_start=request.week_start,
            category=request.category,
            created_at=created_at,
            updated_at=now,
        )

        await self.collection.update_one(
            {"athlete_id": athlete_id, "week_start": request.week_start},
            {"$set": entry.model_dump()},
            upsert=True,
        )
        return entry
