from pymongo.asynchronous.database import AsyncDatabase

from models.prompt_log import PromptLogEntry

_RETENTION_SECONDS = 30 * 24 * 60 * 60


class PromptLogRepository:
    def __init__(self, db: AsyncDatabase):
        self.col = db["prompt_logs"]

    async def ensure_indexes(self) -> None:
        await self.col.create_index("created_at", expireAfterSeconds=_RETENTION_SECONDS)
        await self.col.create_index("run_id")
        await self.col.create_index("athlete_id")

    async def insert_many(self, entries: list[PromptLogEntry]) -> None:
        if not entries:
            return
        await self.col.insert_many([e.model_dump() for e in entries])

    async def list_runs(self, athlete_id: int | None = None, limit: int = 50) -> list[dict]:
        """Return one summary per run_id, newest first."""
        match: dict = {}
        if athlete_id is not None:
            match["athlete_id"] = athlete_id
        pipeline: list[dict] = []
        if match:
            pipeline.append({"$match": match})
        pipeline += [
            {"$sort": {"created_at": 1}},
            {
                "$group": {
                    "_id": "$run_id",
                    "run_id": {"$first": "$run_id"},
                    "athlete_id": {"$first": "$athlete_id"},
                    "crew_name": {"$first": "$crew_name"},
                    "agent_roles": {"$addToSet": "$agent_role"},
                    "call_count": {"$sum": 1},
                    "started_at": {"$first": "$created_at"},
                }
            },
            {"$sort": {"started_at": -1}},
            {"$limit": limit},
            {"$project": {"_id": 0}},
        ]
        cursor = await self.col.aggregate(pipeline)
        return await cursor.to_list(length=None)

    async def get_run(self, run_id: str) -> list[PromptLogEntry]:
        """Return all calls for a run, in chronological order."""
        cursor = self.col.find({"run_id": run_id}).sort("created_at", 1)
        docs = await cursor.to_list(length=None)
        for doc in docs:
            doc.pop("_id", None)
        return [PromptLogEntry(**doc) for doc in docs]
