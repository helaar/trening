from datetime import datetime, timezone

from pymongo.asynchronous.database import AsyncDatabase

from models.memory import Memory, MemoryScope


class MemoryRepository:
    def __init__(self, db: AsyncDatabase):
        self.collection = db["coach_memories"]

    async def get_active(self, athlete_id: int, scope: MemoryScope | None = None) -> list[Memory]:
        query: dict = {"athlete_id": athlete_id, "active": True, "expires_at": {"$gt": datetime.now(timezone.utc)}}
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

    async def get_by_id(self, memory_id: str) -> Memory | None:
        doc = await self.collection.find_one({"memory_id": memory_id})
        if doc is None:
            return None
        doc.pop("_id", None)
        return Memory(**doc)

    async def get_inactive(self, athlete_id: int, scope: MemoryScope | None = None) -> list[Memory]:
        query: dict = {"athlete_id": athlete_id, "active": False}
        if scope is not None:
            query["scope"] = scope.value
        cursor = self.collection.find(query)
        memories = []
        async for doc in cursor:
            doc.pop("_id", None)
            memories.append(Memory(**doc))
        return memories

    async def reactivate(self, memory_id: str) -> None:
        memory = await self.get_by_id(memory_id)
        if memory is None:
            return
        now = datetime.now(timezone.utc)
        refreshed = memory.model_copy(update={"active": True, "updated_at": now}).refresh_expiry()
        await self.collection.update_one(
            {"memory_id": memory_id},
            {"$set": {"active": True, "updated_at": now, "expires_at": refreshed.expires_at}},
        )

    async def get_expiring_before(self, cutoff: datetime) -> list[Memory]:
        cursor = self.collection.find({"active": True, "expires_at": {"$lt": cutoff}})
        memories = []
        async for doc in cursor:
            doc.pop("_id", None)
            memories.append(Memory(**doc))
        return memories
