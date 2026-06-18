"""Admin API for managing crew definitions (agents, tasks, philosophies) in MongoDB.

NOTE: These endpoints are intentionally unauthenticated during the alpha phase.
Add authorization before any production rollout.
"""
import logging

from fastapi import APIRouter, Depends
from pymongo.asynchronous.database import AsyncDatabase

from database.crew_definition_repository import CrewDefinitionRepository
from database.mongodb import get_db
from models.crew_definition import CrewDefinition

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/admin/prompts", tags=["admin"])


async def get_crew_def_repo(db: AsyncDatabase = Depends(get_db)) -> CrewDefinitionRepository:
    return CrewDefinitionRepository(db)


@router.get("", response_model=list[CrewDefinition])
async def list_prompts(repo: CrewDefinitionRepository = Depends(get_crew_def_repo)) -> list[CrewDefinition]:
    """Return all crew definitions (the single source of truth — no YAML merging)."""
    defs = await repo.get_all()
    return sorted(defs, key=lambda d: (d.type, d.name))


@router.put("", response_model=list[CrewDefinition])
async def save_prompts(
    updates: list[CrewDefinition],
    repo: CrewDefinitionRepository = Depends(get_crew_def_repo),
) -> list[CrewDefinition]:
    """Bulk upsert crew definitions, keyed by (type, name)."""
    for doc in updates:
        await repo.upsert(doc)
    return updates
