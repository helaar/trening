from datetime import datetime, timezone

from pymongo.asynchronous.database import AsyncDatabase

from models.intervals_activity import IntervalsActivityRaw


class IntervalsActivityRepository:
    """Repository for Intervals.icu activity cross-reference data.

    Mirrors WorkoutRepository's store/get shape for strava_activities. Keyed on
    (athlete_id, strava_activity_id) when a Strava cross-reference exists;
    (athlete_id, intervals_activity_id) is the fallback key for records that
    only ever got a fuzzy or ambiguous match.
    """

    def __init__(self, db: AsyncDatabase):
        self.db = db
        self.collection = db["intervals_activities"]

    async def store_activity(self, activity: IntervalsActivityRaw) -> bool:
        """Store or update an Intervals.icu activity record."""
        key: dict = {
            "athlete_id": activity.athlete_id,
            "intervals_activity_id": activity.intervals_activity_id,
        }
        activity_dict = activity.model_dump()

        result = await self.collection.update_one(
            key,
            {"$set": {**activity_dict, "updated_at": datetime.now(timezone.utc)}},
            upsert=True,
        )
        return result.acknowledged

    async def get_by_strava_activity_id(
        self, athlete_id: int, strava_activity_id: int
    ) -> IntervalsActivityRaw | None:
        """Get the Intervals.icu record cross-referenced to a given Strava activity."""
        doc = await self.collection.find_one(
            {"athlete_id": athlete_id, "strava_activity_id": strava_activity_id}
        )
        if doc:
            doc.pop("_id", None)
            return IntervalsActivityRaw(**doc)
        return None

    async def get_by_intervals_activity_id(
        self, athlete_id: int, intervals_activity_id: str
    ) -> IntervalsActivityRaw | None:
        """Get an Intervals.icu record by its own id (used for fuzzy/ambiguous matches)."""
        doc = await self.collection.find_one(
            {"athlete_id": athlete_id, "intervals_activity_id": intervals_activity_id}
        )
        if doc:
            doc.pop("_id", None)
            return IntervalsActivityRaw(**doc)
        return None

    async def exists_for_strava_activity(self, athlete_id: int, strava_activity_id: int) -> bool:
        count = await self.collection.count_documents(
            {"athlete_id": athlete_id, "strava_activity_id": strava_activity_id}
        )
        return count > 0
