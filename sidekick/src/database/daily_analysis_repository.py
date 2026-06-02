from pymongo.asynchronous.database import AsyncDatabase

from models.daily_analysis import DailyAnalysisResult


class DailyAnalysisRepository:
    def __init__(self, db: AsyncDatabase):
        self.collection = db["daily_analyses"]

    async def get(self, athlete_id: int, date: str) -> DailyAnalysisResult | None:
        doc = await self.collection.find_one({"athlete_id": athlete_id, "date": date})
        if doc:
            doc.pop("_id", None)
            return DailyAnalysisResult(**doc)
        return None

    async def get_summaries_for_range(self, athlete_id: int, start_date: str, end_date: str) -> list[dict]:
        """Return lightweight per-day summaries without deserializing nested LLM output fields.

        Reads only the fields needed for memory consolidation context, tolerating old
        documents where nested fields may be raw strings rather than dicts.
        """
        projection = {
            "_id": 0,
            "date": 1,
            "coaching_feedback.key_takeaway": 1,
            "coaching_feedback.coach_notes": 1,
            "restitution_analysis.overall_recovery_quality": 1,
        }
        cursor = self.collection.find(
            {"athlete_id": athlete_id, "date": {"$gte": start_date, "$lte": end_date}},
            projection,
        )
        summaries = []
        async for doc in cursor:
            entry: dict = {"date": doc.get("date")}
            cf = doc.get("coaching_feedback")
            if isinstance(cf, dict):
                entry["key_takeaway"] = cf.get("key_takeaway")
                entry["coach_notes"] = cf.get("coach_notes")
            ra = doc.get("restitution_analysis")
            if isinstance(ra, dict):
                entry["recovery_quality"] = ra.get("overall_recovery_quality")
            summaries.append(entry)
        return summaries

    async def upsert(self, result: DailyAnalysisResult) -> None:
        await self.collection.update_one(
            {"athlete_id": result.athlete_id, "date": result.date},
            {"$set": result.model_dump(mode="json")},
            upsert=True,
        )
