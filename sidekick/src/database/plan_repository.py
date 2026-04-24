from datetime import datetime, timezone

from pymongo.asynchronous.database import AsyncDatabase

from models.plan import PlannedActivity, PlannedActivityRequest


class PlanRepository:
    def __init__(self, db: AsyncDatabase):
        self.collection = db["training_plans"]

    def _to_model(self, doc: dict) -> PlannedActivity:
        doc.pop("_id", None)
        return PlannedActivity(**doc)

    async def get_for_date(self, athlete_id: int, date: str) -> list[PlannedActivity]:
        cursor = self.collection.find({"athlete_id": athlete_id, "date": date})
        return [self._to_model(doc) async for doc in cursor]

    async def get_for_range(self, athlete_id: int, start: str, end: str) -> list[PlannedActivity]:
        cursor = self.collection.find(
            {"athlete_id": athlete_id, "date": {"$gte": start, "$lte": end}}
        ).sort("date", 1)
        return [self._to_model(doc) async for doc in cursor]

    async def create(self, athlete_id: int, request: PlannedActivityRequest) -> PlannedActivity:
        now = datetime.now(timezone.utc)
        activity = PlannedActivity(
            athlete_id=athlete_id,
            created_at=now,
            updated_at=now,
            **request.model_dump(),
        )
        await self.collection.insert_one(activity.model_dump(mode="json"))
        return activity

    async def update(
        self, athlete_id: int, plan_id: str, request: PlannedActivityRequest
    ) -> PlannedActivity | None:
        existing = await self.collection.find_one({"id": plan_id, "athlete_id": athlete_id})
        if not existing:
            return None
        now = datetime.now(timezone.utc)
        updated = PlannedActivity(
            id=plan_id,
            athlete_id=athlete_id,
            created_at=existing["created_at"],
            updated_at=now,
            **request.model_dump(),
        )
        await self.collection.update_one(
            {"id": plan_id, "athlete_id": athlete_id},
            {"$set": updated.model_dump(mode="json")},
        )
        return updated

    async def delete(self, athlete_id: int, plan_id: str) -> bool:
        result = await self.collection.delete_one({"id": plan_id, "athlete_id": athlete_id})
        return result.deleted_count > 0
