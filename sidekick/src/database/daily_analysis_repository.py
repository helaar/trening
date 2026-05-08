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

    async def upsert(self, result: DailyAnalysisResult) -> None:
        await self.collection.update_one(
            {"athlete_id": result.athlete_id, "date": result.date},
            {"$set": result.model_dump(mode="json")},
            upsert=True,
        )
