from datetime import datetime, timezone

from pymongo.asynchronous.database import AsyncDatabase

from models.prompt_config import PromptConfig


class PromptRepository:
    def __init__(self, db: AsyncDatabase):
        self.col = db["prompt_configs"]

    async def get_all(self) -> list[PromptConfig]:
        docs = await self.col.find({}).to_list(length=None)
        for doc in docs:
            doc.pop("_id", None)
        return [PromptConfig(**doc) for doc in docs]

    async def delete_many(self, keys: list[str]) -> None:
        if keys:
            await self.col.delete_many({"key": {"$in": keys}})

    async def upsert_many(self, items: list[dict]) -> list[PromptConfig]:
        now = datetime.now(timezone.utc)
        results = []
        for item in items:
            key = item["key"]
            value = item["value"]
            await self.col.update_one(
                {"key": key},
                {"$set": {"key": key, "value": value, "updated_at": now}},
                upsert=True,
            )
            results.append(PromptConfig(key=key, value=value, updated_at=now))
        return results
