from pymongo.asynchronous.database import AsyncDatabase

from models.coach import Coach


class CoachRepository:
    """Repository for coach roster lookups."""

    def __init__(self, db: AsyncDatabase):
        self.collection = db["coaches"]

    async def get_coach(self, coach_id: int) -> Coach | None:
        doc = await self.collection.find_one({"coach_id": coach_id})
        if doc:
            doc.pop("_id", None)
            return Coach(**doc)
        return None

    async def is_coach(self, coach_id: int) -> bool:
        return await self.get_coach(coach_id) is not None

    async def get_roster_athlete_ids(self, coach_id: int) -> list[int]:
        coach = await self.get_coach(coach_id)
        return coach.athlete_ids if coach else []

    async def upsert(self, coach: Coach) -> None:
        await self.collection.update_one(
            {"coach_id": coach.coach_id},
            {"$set": coach.model_dump()},
            upsert=True,
        )
