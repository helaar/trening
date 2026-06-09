from datetime import datetime, timezone

from pymongo.asynchronous.database import AsyncDatabase

from models.memory import Memory, MemoryScope


class MemoryRepository:
    def __init__(self, db: AsyncDatabase):
        self.collection = db["coach_memories"]

    async def get_active(self, athlete_id: int, scope: MemoryScope | None = None) -> list[Memory]:
        query: dict = {"athlete_id": athlete_id, "active": True, "expires_at": {"$gt": datetime.now(timezone.utc).replace(tzinfo=None)}}
        if scope is not None:
            query["scope"] = scope.value
        cursor = self.collection.find(query)
        memories = []
        async for doc in cursor:
            doc.pop("_id", None)
            memories.append(Memory(**doc))
        return memories

    async def upsert(self, memory: Memory) -> None:
        await self.collection.update_one(
            {"memory_id": memory.memory_id},
            {"$set": memory.model_dump(mode="python")},
            upsert=True,
        )

    async def deactivate(self, memory_id: str) -> None:
        await self.collection.update_one(
            {"memory_id": memory_id},
            {"$set": {"active": False}},
        )

    async def get_expiring_before(self, cutoff: datetime) -> list[Memory]:
        naive_cutoff = cutoff.replace(tzinfo=None) if cutoff.tzinfo is not None else cutoff
        cursor = self.collection.find({"active": True, "expires_at": {"$lt": naive_cutoff}})
        memories = []
        async for doc in cursor:
            doc.pop("_id", None)
            memories.append(Memory(**doc))
        return memories
