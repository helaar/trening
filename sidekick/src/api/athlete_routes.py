import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from pymongo.asynchronous.database import AsyncDatabase

from analysis.memory_relevance import DayContext, select_relevant_memories_objects
from auth.dependencies import get_current_athlete_id
from database.athlete_repository import AthleteRepository
from database.memory_repository import MemoryRepository
from database.mongodb import get_db
from models.athlete import AthleteSettings, HeartRateSettings, SportSettings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/athlete", tags=["athlete"])


class AthleteSettingsPatch(AthleteSettings):
    """Partial update model for athlete settings — all fields optional."""

    cycling: SportSettings | None = None
    running: SportSettings | None = None
    heart_rate: HeartRateSettings | None = None
    autolap: str | None = None


async def get_athlete_repository(
    db: AsyncDatabase = Depends(get_db),
) -> AthleteRepository:
    return AthleteRepository(db)


async def get_memory_repository(
    db: AsyncDatabase = Depends(get_db),
) -> MemoryRepository:
    return MemoryRepository(db)


class MemoryView(BaseModel):
    """A single memory as surfaced to the athlete, mirroring the coach's tool fields."""

    memory_id: str
    scope: str
    category: str
    content: str
    confidence: float


@router.get("/{athlete_id}/settings", response_model=AthleteSettings)
async def get_settings(
    athlete_id: int,
    current_athlete_id: int = Depends(get_current_athlete_id),
    repo: AthleteRepository = Depends(get_athlete_repository),
) -> AthleteSettings:
    """Get athlete training settings."""
    if athlete_id != current_athlete_id:
        raise HTTPException(status_code=403, detail="You can only access your own data")
    settings = await repo.get_athlete_settings(athlete_id)
    if settings is None:
        raise HTTPException(status_code=404, detail="Athlete not found")
    return settings


@router.patch("/{athlete_id}/settings", response_model=AthleteSettings)
async def patch_settings(
    athlete_id: int,
    patch: AthleteSettingsPatch,
    current_athlete_id: int = Depends(get_current_athlete_id),
    repo: AthleteRepository = Depends(get_athlete_repository),
) -> AthleteSettings:
    """Partially update athlete training settings. Only provided fields are written."""
    if athlete_id != current_athlete_id:
        raise HTTPException(status_code=403, detail="You can only access your own data")

    # Build update dict from only the fields present in the request body
    settable = {
        "cycling",
        "running",
        "heart_rate",
        "autolap",
        "trainingpeaks_ical_url",
        "training_philosophy",
    }
    fields: dict = {}
    for field in patch.model_fields_set & settable:
        value = getattr(patch, field)
        fields[field] = value.model_dump() if hasattr(value, "model_dump") else value

    if not fields:
        # Nothing to update — just return current settings
        settings = await repo.get_athlete_settings(athlete_id)
        if settings is None:
            raise HTTPException(status_code=404, detail="Athlete not found")
        return settings

    logger.info(f"Patching settings for athlete {athlete_id}: fields={list(fields)}")
    athlete = await repo.patch_athlete_settings(athlete_id, fields)
    if athlete is None:
        raise HTTPException(status_code=404, detail="Athlete not found")
    return athlete.settings


@router.get("/{athlete_id}/memories", response_model=list[MemoryView])
async def get_memories(
    athlete_id: int,
    current_athlete_id: int = Depends(get_current_athlete_id),
    repo: MemoryRepository = Depends(get_memory_repository),
) -> list[MemoryView]:
    """The athlete's active memory bank, ranked the way the coach's tool sees it.

    Reuses the coach's ranking logic (``select_relevant_memories_objects``) so the
    ordering matches what the coach reasons over. First iteration uses a neutral
    ``DayContext`` — the situational boost (today's readiness/demand/race-phase)
    requires the daily-analysis pipeline and is a planned follow-up; confidence,
    importance, recency and the durable-core logic already apply here identically.
    """
    if athlete_id != current_athlete_id:
        raise HTTPException(status_code=403, detail="You can only access your own data")

    memories = await repo.get_active(athlete_id)
    today = datetime.now(timezone.utc).date().isoformat()
    ranked = select_relevant_memories_objects(memories, DayContext(), today)
    return [
        MemoryView(
            memory_id=m.memory_id,
            scope=m.scope.value,
            category=m.category.value,
            content=m.content,
            confidence=m.confidence,
        )
        for m in ranked
    ]


@router.get("/{athlete_id}/memories/suppressed", response_model=list[MemoryView])
async def get_suppressed_memories(
    athlete_id: int,
    current_athlete_id: int = Depends(get_current_athlete_id),
    repo: MemoryRepository = Depends(get_memory_repository),
) -> list[MemoryView]:
    """Memories the athlete or an admin has suppressed, most recently first.

    A flat undo list — no relevance ranking, since that logic exists to pick what
    the coach sees, not to rank things nobody currently acts on.
    """
    if athlete_id != current_athlete_id:
        raise HTTPException(status_code=403, detail="You can only access your own data")

    memories = await repo.get_inactive(athlete_id)
    memories.sort(key=lambda m: m.updated_at, reverse=True)
    return [
        MemoryView(
            memory_id=m.memory_id,
            scope=m.scope.value,
            category=m.category.value,
            content=m.content,
            confidence=m.confidence,
        )
        for m in memories
    ]


async def _get_owned_memory(athlete_id: int, memory_id: str, repo: MemoryRepository) -> None:
    memory = await repo.get_by_id(memory_id)
    if memory is None or memory.athlete_id != athlete_id:
        raise HTTPException(status_code=404, detail="Memory not found")


@router.delete("/{athlete_id}/memories/{memory_id}", status_code=204)
async def suppress_memory(
    athlete_id: int,
    memory_id: str,
    current_athlete_id: int = Depends(get_current_athlete_id),
    repo: MemoryRepository = Depends(get_memory_repository),
) -> None:
    """Suppress a memory (soft delete) — a dev tool for tuning coach prompts."""
    if athlete_id != current_athlete_id:
        raise HTTPException(status_code=403, detail="You can only access your own data")

    await _get_owned_memory(athlete_id, memory_id, repo)
    await repo.deactivate(memory_id)


@router.post("/{athlete_id}/memories/{memory_id}/restore", status_code=204)
async def restore_memory(
    athlete_id: int,
    memory_id: str,
    current_athlete_id: int = Depends(get_current_athlete_id),
    repo: MemoryRepository = Depends(get_memory_repository),
) -> None:
    """Undo a suppression, restoring a memory to the active set."""
    if athlete_id != current_athlete_id:
        raise HTTPException(status_code=403, detail="You can only access your own data")

    await _get_owned_memory(athlete_id, memory_id, repo)
    await repo.reactivate(memory_id)
