"""Admin API for inspecting captured LLM prompts/responses from crew runs.

NOTE: These endpoints are intentionally unauthenticated during the alpha phase.
Add authorization (admin-only) before any production rollout — entries contain
full athlete data (names, settings, workout notes) passed to the LLM.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from pymongo.asynchronous.database import AsyncDatabase

from database.mongodb import get_db
from database.prompt_log_repository import PromptLogRepository
from models.prompt_log import PromptLogEntry

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/admin/prompt-logs", tags=["admin"])


async def get_prompt_log_repo(db: AsyncDatabase = Depends(get_db)) -> PromptLogRepository:
    return PromptLogRepository(db)


@router.get("")
async def list_runs(
    athlete_id: int | None = None,
    limit: int = 50,
    repo: PromptLogRepository = Depends(get_prompt_log_repo),
) -> list[dict]:
    """Return recent crew runs with captured prompts, newest first."""
    return await repo.list_runs(athlete_id=athlete_id, limit=limit)


@router.get("/{run_id}", response_model=list[PromptLogEntry])
async def get_run(
    run_id: str,
    repo: PromptLogRepository = Depends(get_prompt_log_repo),
) -> list[PromptLogEntry]:
    """Return every captured LLM call for a run, in chronological order."""
    entries = await repo.get_run(run_id)
    if not entries:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")
    return entries
