"""Admin API for managing LLM prompt overrides stored in MongoDB.

NOTE: These endpoints are intentionally unauthenticated during the alpha phase.
Add authorization before any production rollout.
"""
import logging
from pathlib import Path

import yaml
from fastapi import APIRouter, Depends
from pymongo.asynchronous.database import AsyncDatabase

from database.mongodb import get_db
from database.prompt_repository import PromptRepository
from models.prompt_config import PromptConfig, PromptConfigUpdate

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/admin/prompts", tags=["admin"])

_CREW_DIR = Path(__file__).parent.parent / "crew"


def _load_yaml_defaults() -> dict[str, str]:
    """Return flat key->value dict of all prompts from agents.yaml and tasks.yaml."""
    defaults: dict[str, str] = {}

    def _flatten(prefix: str, obj: dict) -> None:
        for k, v in obj.items():
            full_key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                _flatten(full_key, v)
            elif isinstance(v, str):
                defaults[full_key] = v

    for filename in ("agents.yaml", "tasks.yaml"):
        path = _CREW_DIR / filename
        stem = filename.removesuffix(".yaml")
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        _flatten(stem, data)

    return defaults


async def get_prompt_repo(db: AsyncDatabase = Depends(get_db)) -> PromptRepository:
    return PromptRepository(db)


@router.get("", response_model=list[PromptConfig])
async def list_prompts(repo: PromptRepository = Depends(get_prompt_repo)) -> list[PromptConfig]:
    """Return all prompts: YAML defaults merged with DB overrides."""
    defaults = _load_yaml_defaults()
    overrides = {p.key: p for p in await repo.get_all()}

    result: list[PromptConfig] = []
    for key, value in defaults.items():
        if key in overrides:
            result.append(overrides[key])
        else:
            result.append(PromptConfig(key=key, value=value))

    # Include DB-only entries that have no YAML default (e.g. philosophy.*)
    for key, config in overrides.items():
        if key not in defaults:
            result.append(config)

    return result


@router.put("", response_model=list[PromptConfig])
async def save_prompts(
    updates: list[PromptConfigUpdate],
    repo: PromptRepository = Depends(get_prompt_repo),
) -> list[PromptConfig]:
    """Bulk upsert prompt overrides."""
    if not updates:
        return []
    return await repo.upsert_many([u.model_dump() for u in updates])
