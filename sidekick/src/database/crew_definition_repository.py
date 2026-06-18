"""Repository for the unified `crew_definitions` collection (agents, tasks, philosophies)."""
from datetime import datetime, timezone

from pymongo.asynchronous.database import AsyncDatabase

from models.crew_definition import (
    AgentDoc,
    CrewDefinition,
    PhilosophyDoc,
    TaskDoc,
    crew_definition_adapter,
)


class CrewDefinitionRepository:
    def __init__(self, db: AsyncDatabase):
        self.col = db["crew_definitions"]

    async def ensure_indexes(self) -> None:
        await self.col.create_index([("type", 1), ("name", 1)], unique=True)

    @staticmethod
    def _parse(doc: dict) -> CrewDefinition:
        doc.pop("_id", None)
        return crew_definition_adapter.validate_python(doc)

    async def get_all(self) -> list[CrewDefinition]:
        docs = await self.col.find({}).to_list(length=None)
        return [self._parse(d) for d in docs]

    async def get_by_type(self, type_: str) -> list[CrewDefinition]:
        docs = await self.col.find({"type": type_}).to_list(length=None)
        return [self._parse(d) for d in docs]

    async def _get_one(self, type_: str, name: str) -> CrewDefinition | None:
        doc = await self.col.find_one({"type": type_, "name": name})
        return self._parse(doc) if doc else None

    async def get_agent(self, name: str) -> AgentDoc | None:
        doc = await self._get_one("agent", name)
        return doc if isinstance(doc, AgentDoc) else None

    async def get_task(self, name: str) -> TaskDoc | None:
        doc = await self._get_one("task", name)
        return doc if isinstance(doc, TaskDoc) else None

    async def get_philosophy(self, slug: str) -> PhilosophyDoc | None:
        doc = await self._get_one("philosophy", slug)
        return doc if isinstance(doc, PhilosophyDoc) else None

    async def upsert(self, doc: CrewDefinition) -> None:
        data = doc.model_dump()
        data["updated_at"] = datetime.now(timezone.utc)
        await self.col.update_one(
            {"type": data["type"], "name": data["name"]},
            {"$set": data},
            upsert=True,
        )

    async def insert_if_absent(self, doc: CrewDefinition) -> bool:
        """Insert only when no doc with this (type, name) exists. Returns True if inserted.

        Used by the seed script so re-running never clobbers admin-edited values.
        """
        existing = await self.col.find_one({"type": doc.type, "name": doc.name})
        if existing:
            return False
        await self.col.insert_one(doc.model_dump())
        return True

    async def delete(self, type_: str, name: str) -> None:
        await self.col.delete_one({"type": type_, "name": name})
